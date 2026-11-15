"""
Physical-Layer RF Feature Extraction & Feature Group Definitions (`src/adsb_rf/features.py`).

Separates:
- Carrier Frequency Features (`CFO_Raw_Hz`, `Frequency_shift`)
- Hardware Transient & Symbol Clock Features (`Timing_Offset_PPM`, `Sub_Sample_Lag_Ns`,
  `Phase_Noise_Std`, `PA_Droop_dB`, `Pulse_Crest_Factor`, `Phase_Curvature_Rad`,
  `SNR_dB`, `Preamble_Ratio_dB`)
- Channel / Power Features (`Power_dBm`, `EIRP_dBm`)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .assumptions import DEFAULT_ASSUMPTIONS
from .dsp import compute_eirp_dbm, extract_waveform_transient_features, recover_cfo_unwrap_free

RF_FEATURE_NAMES: List[str] = [
    "CFO_Raw_Hz",            # 0: Raw Carrier Frequency Offset at receiver (Hz)
    "Frequency_shift",       # 1: Doppler-compensated Carrier Frequency Offset (Hz)
    "Timing_Offset_PPM",     # 2: Symbol clock crystal skew across 120 us frame (ppm)
    "Sub_Sample_Lag_Ns",     # 3: Fractional arrival phase / correlation peak asymmetry (ns)
    "Power_dBm",             # 4: Calibrated received power at antenna port (dBm)
    "EIRP_dBm",              # 5: Free-Space Path Loss compensated transmit power (dBm)
    "Phase_Noise_Std",       # 6: Unwrapped carrier phase residual standard deviation (rad)
    "PA_Droop_dB",           # 7: Power Amplifier envelope sag from first 30 us to last 30 us (dB)
    "Pulse_Crest_Factor",    # 8: Peak-to-average pulse envelope ratio (PA turn-on overshoot)
    "Phase_Curvature_Rad",   # 9: Quadratic phase chirp across 120 us burst (PA thermal pulling)
    "SNR_dB",                # 10: In-frame active pulse to off-chip noise floor ratio (dB)
    "Preamble_Ratio_dB",     # 11: Preamble-to-payload pulse amplitude balance (dB)
]

CSV_FEATURE_COLUMNS: List[str] = [
    "cfo_raw_hz",
    "freq_offset_hz",
    "timing_offset_ppm",
    "sub_sample_lag_ns",
    "rx_power_dbm",
    "eirp_dbm",
    "phase_noise_std",
    "pa_droop_db",
    "pulse_crest_factor",
    "phase_curvature_rad",
    "snr_db",
    "preamble_ratio_db",
]

# Explicit feature subsets for Ablation Studies
FEATURE_SUBSETS: Dict[str, List[int]] = {
    "cfo_only": [0, 1],
    "hardware_no_cfo": [2, 3, 6, 7, 8, 9, 10, 11],
    "hardware_plus_power_no_cfo": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "waveform_10_no_power": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
    "all_12_features": list(range(12)),
}

RF_PHYSICAL_STD_FLOOR = np.array(DEFAULT_ASSUMPTIONS.rf_physical_std_floor, dtype=np.float64)
RF_MAHALANOBIS_WEIGHTS = np.array(DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights, dtype=np.float64)


def _recover_missing_cfo_from_iq(frame: Dict[str, Any], fs_hz: float = 10e6) -> Tuple[float, float]:
    """Recovers `CFO_Raw_Hz` and `Phase_Noise_Std` from `IQ_res` using two-stage unwrap-free estimation."""
    iq_res = frame.get("IQ_res") if frame.get("IQ_res") is not None else frame.get("IQ_resampled")
    bits = frame.get("Bits")
    cfo_hz, pstd = recover_cfo_unwrap_free(iq_res, bits, fs_hz=fs_hz)
    existing_pstd = float(frame.get("Phase_Noise_Std", float("nan")))
    return cfo_hz, (pstd if np.isnan(existing_pstd) else existing_pstd)


def _extract_waveform_transient_features(
    iq_res: Optional[np.ndarray],
) -> Tuple[float, float, float, float, float]:
    """Backwards-compatible wrapper around `dsp.extract_waveform_transient_features`."""
    return extract_waveform_transient_features(iq_res)


def _backfill_aircraft_kinematics(
    frames: List[Dict[str, Any]],
    carrier_freq_hz: float = DEFAULT_ASSUMPTIONS.carrier_freq_hz,
    causal_only: bool = False,
) -> None:
    """
    1. Recovers any `NaN` `CFO_Raw_Hz` from `IQ_res` using `_recover_missing_cfo_from_iq(f)`.
    2. When `causal_only=True`, only uses past frames of the claimed aircraft (or direct frame
       measurements) to avoid future temporal leakage. When `causal_only=False` (enrollment mode),
       interpolates median Doppler and slant range across clean frames of each claimed aircraft.
    """
    for f in frames:
        cfo_raw = float(f.get("CFO_Raw_Hz", float("nan")))
        if np.isnan(cfo_raw):
            rec_cfo, rec_pstd = _recover_missing_cfo_from_iq(f)
            f["CFO_Raw_Hz"] = rec_cfo
            if np.isnan(f.get("Phase_Noise_Std", float("nan"))):
                f["Phase_Noise_Std"] = rec_pstd
            dop = float(f.get("Doppler", float("nan")))
            f["Frequency_shift"] = float(rec_cfo - dop) if not np.isnan(dop) else rec_cfo

    if causal_only:
        last_dop: Dict[str, float] = {}
        last_dist: Dict[str, float] = {}
        for f in frames:
            icao = str(f.get("ICAO_Hex", ""))
            dop = float(f.get("Doppler", float("nan")))
            dist_m = float(f.get("Distance_from_Rx", float("nan")))
            if not np.isnan(dop):
                last_dop[icao] = dop
            elif icao in last_dop:
                dop = last_dop[icao]
                f["Doppler"] = dop
            if not np.isnan(dist_m) and dist_m > 0:
                last_dist[icao] = dist_m
            elif icao in last_dist:
                dist_m = last_dist[icao]
                f["Distance_from_Rx"] = dist_m

            cfo_raw = float(f.get("CFO_Raw_Hz", float("nan")))
            if not np.isnan(cfo_raw):
                f["Frequency_shift"] = float(cfo_raw - dop) if not np.isnan(dop) else cfo_raw
            eirp = float(f.get("EIRP_dBm", float("nan")))
            rx_pwr = float(f.get("Power", float("nan")))
            if np.isnan(eirp) and not np.isnan(rx_pwr) and not np.isnan(dist_m) and dist_m > 0:
                f["EIRP_dBm"] = compute_eirp_dbm(rx_pwr, dist_m, carrier_freq_hz)
        return

    by_icao: Dict[str, List[Dict[str, Any]]] = {}
    for f in frames:
        icao = str(f.get("ICAO_Hex", ""))
        if icao:
            by_icao.setdefault(icao, []).append(f)

    for icao, ac_frames in by_icao.items():
        clean_ac = [f for f in ac_frames if not f.get("Rule_Anomaly", False)]
        ref_ac = clean_ac if clean_ac else ac_frames

        valid_dopplers = [
            float(f["Doppler"])
            for f in ref_ac
            if not np.isnan(f.get("Doppler", float("nan")))
        ]
        valid_dists = [
            float(f["Distance_from_Rx"])
            for f in ref_ac
            if not np.isnan(f.get("Distance_from_Rx", float("nan"))) and float(f.get("Distance_from_Rx", 0.0)) > 0
        ]

        med_doppler = float(np.median(valid_dopplers)) if valid_dopplers else float("nan")
        med_dist_m = float(np.median(valid_dists)) if valid_dists else float("nan")

        for f in ac_frames:
            cfo_raw = float(f.get("CFO_Raw_Hz", float("nan")))
            dop = float(f.get("Doppler", float("nan")))
            if np.isnan(dop) and not np.isnan(med_doppler):
                f["Doppler"] = med_doppler
                dop = med_doppler
            if not np.isnan(cfo_raw):
                f["Frequency_shift"] = float(cfo_raw - dop) if not np.isnan(dop) else cfo_raw

            eirp = float(f.get("EIRP_dBm", float("nan")))
            rx_pwr = float(f.get("Power", float("nan")))
            dist_m = float(f.get("Distance_from_Rx", float("nan")))
            if np.isnan(dist_m) and not np.isnan(med_dist_m):
                dist_m = med_dist_m
                f["Distance_from_Rx"] = dist_m
            if np.isnan(eirp) and not np.isnan(rx_pwr) and not np.isnan(dist_m) and dist_m > 0:
                f["EIRP_dBm"] = compute_eirp_dbm(rx_pwr, dist_m, carrier_freq_hz)


def extract_12_rf_features_from_frame(frame: Dict[str, Any]) -> np.ndarray:
    """
    Extracts the 12-dimensional physical-layer RF fingerprint vector for a single frame/burst.
    Uses pre-extracted transient features if present in `frame`, or computes them from `IQ_res`.
    """
    if "_cached_rf_vec12" in frame:
        return frame["_cached_rf_vec12"]

    cfo_raw = float(frame.get("CFO_Raw_Hz", frame.get("Frequency_shift", float("nan"))))
    freq_shift = float(frame.get("Frequency_shift", cfo_raw))
    if np.isnan(freq_shift):
        freq_shift = cfo_raw

    timing_ppm = float(frame.get("Timing_Offset_PPM", 0.0))
    sub_lag_ns = float(frame.get("Sub_Sample_Lag_Ns", 0.0))
    power_dbm = float(frame.get("Power", float("nan")))
    eirp_dbm = float(frame.get("EIRP_dBm", float("nan")))
    phase_noise = float(frame.get("Phase_Noise_Std", float("nan")))

    pa_droop_db = float(frame.get("PA_Droop_dB", float("nan")))
    crest_factor = float(frame.get("Pulse_Crest_Factor", float("nan")))
    phase_curv = float(frame.get("Phase_Curvature_Rad", float("nan")))
    snr_db = float(frame.get("SNR_dB", float("nan")))
    pre_ratio_db = float(frame.get("Preamble_Ratio_dB", float("nan")))

    if any(np.isnan(x) for x in (pa_droop_db, crest_factor, phase_curv, snr_db, pre_ratio_db)):
        iq_res = frame.get("IQ_res") if frame.get("IQ_res") is not None else frame.get("IQ_resampled")
        pa_droop_db, crest_factor, phase_curv, snr_db, pre_ratio_db = extract_waveform_transient_features(iq_res)
        frame["PA_Droop_dB"] = pa_droop_db
        frame["Pulse_Crest_Factor"] = crest_factor
        frame["Phase_Curvature_Rad"] = phase_curv
        frame["SNR_dB"] = snr_db
        frame["Preamble_Ratio_dB"] = pre_ratio_db

    vec = np.array(
        [
            cfo_raw,
            freq_shift,
            timing_ppm,
            sub_lag_ns,
            power_dbm,
            eirp_dbm,
            phase_noise,
            pa_droop_db,
            crest_factor,
            phase_curv,
            snr_db,
            pre_ratio_db,
        ],
        dtype=np.float64,
    )
    frame["_cached_rf_vec12"] = vec
    return vec


def extract_feature_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the `(N, 12)` physical-layer feature matrix directly from a DataFrame (`adsb_table.csv`).
    If any of the 5 transient columns are missing in older CSVs, fills with defaults.
    """
    n_rows = len(df)
    X = np.full((n_rows, 12), np.nan, dtype=np.float64)
    defaults = [0.0, 0.0, 0.0, 0.0, -75.0, 55.0, 0.35, 0.0, 1.2, 0.0, 20.0, 0.0]

    for j, col_name in enumerate(CSV_FEATURE_COLUMNS):
        if col_name in df.columns:
            X[:, j] = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float64)
        else:
            X[:, j] = defaults[j]

    # Fallback: if freq_offset_hz is NaN on non-position frames, use cfo_raw_hz (no label leakage!)
    missing_fshift = np.isnan(X[:, 1]) & ~np.isnan(X[:, 0])
    X[missing_fshift, 1] = X[missing_fshift, 0]
    return X
