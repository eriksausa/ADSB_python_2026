"""
Online IQ Stream Reader and Polyphase Resampler.
Replaces the legacy 1-second batch splicing loop (`for l_idx in range(k)`) with
continuous single-pass or memory-mapped sliding-window online reading for a single measurement file.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, Optional, Union
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly

from config import ReceiverConfig


@dataclass
class IQStreamChunk:
    """Represents a continuous online IQ window with exact sample/time offset tracking."""
    iq_raw: np.ndarray
    inphase_res: np.ndarray
    quadrature_res: np.ndarray
    envelope_res: np.ndarray
    start_time_s: float
    raw_sample_rate_hz: float
    target_sample_rate_hz: float
    raw_start_sample: int
    overlap_samples_res: int = 0

    def to_resampled_dict(self) -> Dict[str, np.ndarray]:
        """Compatibility dictionary matching legacy `resample_data()` output."""
        return {
            "Inphase": self.inphase_res,
            "Quadrature": self.quadrature_res,
            "Z": self.envelope_res,
        }


def resample_data(y_chunk: np.ndarray, new_fs: float, old_fs: float) -> Dict[str, np.ndarray]:
    """
    Resamples a complex I/Q time-domain signal from old_fs to new_fs using polyphase filtering.
    Returns a dictionary containing 'Inphase', 'Quadrature', and magnitude envelope 'Z'.
    """
    if np.isclose(new_fs, old_fs):
        inphase = np.real(y_chunk).astype(np.float64)
        quadrature = np.imag(y_chunk).astype(np.float64)
        return {
            "Inphase": inphase,
            "Quadrature": quadrature,
            "Z": np.abs(inphase + 1j * quadrature),
        }

    ratio = Fraction(new_fs / old_fs).limit_denominator(1000)
    p_up, q_down = ratio.numerator, ratio.denominator

    inphase = np.real(y_chunk).astype(np.float64)
    quadrature = np.imag(y_chunk).astype(np.float64)

    inphase_resampled = resample_poly(inphase, p_up, q_down)
    quadrature_resampled = resample_poly(quadrature, p_up, q_down)
    z_resampled = np.abs(inphase_resampled + 1j * quadrature_resampled)

    return {
        "Inphase": inphase_resampled,
        "Quadrature": quadrature_resampled,
        "Z": z_resampled,
    }


class OnlineIQStreamReader:
    """
    Continuous / online reader for a single RF measurement file (.mat, .npy, or in-memory array).
    Eliminates artificial 1-second batch splicing while preserving exact sample-accurate timestamps.
    """

    def __init__(
        self,
        source: Union[str, Path, np.ndarray],
        config: Optional[ReceiverConfig] = None,
        raw_sample_rate_hz: Optional[float] = None,
        start_time_s: Optional[float] = None,
    ):
        self.config = config or ReceiverConfig()
        self.source = source
        self.raw_sample_rate_hz = raw_sample_rate_hz or self.config.default_raw_sample_rate_hz
        self.target_sample_rate_hz = self.config.target_sample_rate_hz
        self.start_time_s = start_time_s if start_time_s is not None else self._infer_start_time(source)
        self._iq_buffer: Optional[np.ndarray] = None
        self._load_source()

    @staticmethod
    def _infer_start_time(source: Union[str, Path, np.ndarray]) -> float:
        """Extracts base timestamp from numeric filenames (e.g. '101000.mat' -> 101000.0) or defaults to 0.0."""
        if isinstance(source, (str, Path)):
            stem = Path(source).stem
            digits = "".join(ch for ch in stem[:6] if ch.isdigit())
            if len(digits) == 6:
                return float(digits)
        return 0.0

    def _load_source(self) -> None:
        """Initializes the underlying IQ buffer (using mmap for .npy or direct load for .mat)."""
        if isinstance(self.source, np.ndarray):
            self._iq_buffer = np.squeeze(self.source)
            return

        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Measurement file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npy":
            # Memory-mapped online reading: avoids loading massive arrays into RAM all at once
            self._iq_buffer = np.load(path, mmap_mode="r")
            if self._iq_buffer.ndim > 1:
                self._iq_buffer = np.squeeze(self._iq_buffer)
        elif suffix == ".mat":
            # Inspect metadata first
            try:
                vars_info = {name: shape for name, shape, _ in whosmat(str(path))}
            except Exception:
                vars_info = {}
            mat = loadmat(str(path))
            if "XDelta" in mat:
                xdelta = float(np.squeeze(mat["XDelta"]))
                if xdelta > 0:
                    self.raw_sample_rate_hz = 1.0 / xdelta
            if "Y" in mat:
                self._iq_buffer = np.squeeze(mat["Y"])
            else:
                raise KeyError(f"Expected 'Y' IQ array in {path.name}, found keys: {list(mat.keys())}")
        else:
            raise ValueError(f"Unsupported IQ measurement file extension: {suffix}")

    @property
    def total_samples(self) -> int:
        return 0 if self._iq_buffer is None else int(len(self._iq_buffer))

    @property
    def duration_s(self) -> float:
        return self.total_samples / self.raw_sample_rate_hz

    def read_continuous(self) -> IQStreamChunk:
        """
        Reads and resamples the entire single measurement in one continuous online pass
        (no 1-second loop splicing). Ideal when processing a single measurement file.
        """
        assert self._iq_buffer is not None
        resampled = resample_data(
            self._iq_buffer,
            new_fs=self.target_sample_rate_hz,
            old_fs=self.raw_sample_rate_hz,
        )
        return IQStreamChunk(
            iq_raw=self._iq_buffer,
            inphase_res=resampled["Inphase"],
            quadrature_res=resampled["Quadrature"],
            envelope_res=resampled["Z"],
            start_time_s=self.start_time_s,
            raw_sample_rate_hz=self.raw_sample_rate_hz,
            target_sample_rate_hz=self.target_sample_rate_hz,
            raw_start_sample=0,
            overlap_samples_res=0,
        )

    def stream_online(self, window_duration_s: Optional[float] = None) -> Iterator[IQStreamChunk]:
        """
        Generator that yields continuous IQ chunks with an automatic 120 us frame-overlap guard band.
        If `window_duration_s` is None (default for single measurement files), yields the entire
        measurement in a single continuous pass; otherwise streams sliding windows without dropping
        boundary frames.
        """
        assert self._iq_buffer is not None
        total = self.total_samples
        if total == 0:
            return

        if window_duration_s is None or int(window_duration_s * self.raw_sample_rate_hz) >= total:
            yield self.read_continuous()
            return

        step_raw = int(round(window_duration_s * self.raw_sample_rate_hz))
        overlap_raw = int(np.ceil(self.config.frame_duration_s * self.raw_sample_rate_hz))
        overlap_res = int(np.ceil(self.config.frame_duration_s * self.target_sample_rate_hz))

        cursor = 0
        while cursor < total:
            end_idx = min(total, cursor + step_raw + overlap_raw)
            iq_slice = self._iq_buffer[cursor:end_idx]
            resampled = resample_data(
                iq_slice,
                new_fs=self.target_sample_rate_hz,
                old_fs=self.raw_sample_rate_hz,
            )
            chunk_start_time = self.start_time_s + (cursor / self.raw_sample_rate_hz)

            yield IQStreamChunk(
                iq_raw=iq_slice,
                inphase_res=resampled["Inphase"],
                quadrature_res=resampled["Quadrature"],
                envelope_res=resampled["Z"],
                start_time_s=chunk_start_time,
                raw_sample_rate_hz=self.raw_sample_rate_hz,
                target_sample_rate_hz=self.target_sample_rate_hz,
                raw_start_sample=cursor,
                overlap_samples_res=overlap_res if end_idx < total else 0,
            )
            if end_idx >= total:
                break
            cursor += step_raw