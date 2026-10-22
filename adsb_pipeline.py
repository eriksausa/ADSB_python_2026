"""
Refactored Modular ADS-B RF Fingerprinting & Anomaly Detection Entrypoint (`adsb_pipeline.py`).
Re-exports all DSP, decoding, streaming, SQL database, ML anomaly detection, and visualization
components from their dedicated modules while providing backwards-compatible helper functions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from anomaly_ml import ADSBAnomalyDetector, RFFingerprintClassifier
from config import AnomalyThresholds, ReceiverConfig, parse_vsa_settings
from database import ADSBDatabase
from decoder import (
    ADSB_altitude,
    ADSB_callsign,
    ADSB_position,
    ADSB_velocity,
    NL,
    OnlineADSBDecoder,
    two_message_position,
)
from dsp import (
    build_blind_energy_mask_10mhz,
    calculate_amplitude,
    calculate_freq_shift,
    calculate_freq_shift_and_phase_noise,
    compute_eirp_dbm,
    compute_single_frame_doppler,
    decode_ppm,
    estimate_timing_offset,
    extract_blind_rf_features,
    find_valid_xcorr,
    is_valid,
    lldistkm,
    smooth_3,
)
from main import run_online_pipeline, synthesize_portfolio_iq_stream
from stream_reader import IQStreamChunk, OnlineIQStreamReader, resample_data
from visualization import (
    map_planes,
    plot_eye_diagram,
    plot_iq_new,
    plot_rf_fingerprint_clusters,
)


def power_statistics(messages: List[Dict[str, Any]]) -> None:
    """Calculates mean estimated transmit power (EIRP) across decoded position frames."""
    eirps = [
        m.get("EIRP_dBm", float("nan"))
        for m in messages
        if not np.isnan(m.get("EIRP_dBm", float("nan")))
    ]
    if not eirps:
        return
    print(f"Processed {len(eirps)} position frames.")
    print(f"Mean Estimated Transmit Power (EIRP): {np.nanmean(eirps):.2f} dBm")


def freq_statistics(messages: List[Dict[str, Any]]) -> None:
    """Prints average Carrier Frequency Offset (CFO) across decoded frames."""
    shifts = [
        m.get("Frequency_shift", float("nan"))
        for m in messages
        if not np.isnan(m.get("Frequency_shift", float("nan")))
    ]
    if shifts:
        print(f"Average Carrier Frequency Offset: {np.mean(shifts) / 1e3:.2f} kHz")


if __name__ == "__main__":
    uploaded_dir = Path("/working_dir/c_2aa6628f1b536aac/user_uploaded_files/ttl=63d/user_uploaded")
    run_online_pipeline(
        measurement_path=uploaded_dir / "receivedSignal (1).npy",
        vsa_setx_path=uploaded_dir / "vsa_settings.setx",
        output_dir=Path(__file__).resolve().parent,
    )