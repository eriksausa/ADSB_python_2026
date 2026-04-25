"""
Configuration and metadata parser for the modular ADS-B RF Fingerprinting & Anomaly Detection pipeline.
Supports Keysight 89600 VSA (.setx) XML setup files and receiver hardware calibration parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


@dataclass
class ReceiverConfig:
    """Hardware and DSP configuration for the ADS-B receiver station."""
    # Receiver Antenna Coordinates (TU Wien EI Building roof baseline)
    rx_lat: float = 48.196674
    rx_lon: float = 16.370998

    # RF Front-End Calibration (in dB)
    antenna_gain_db: float = 5.5
    cable_loss_db: float = 11.37
    reference_impedance_ohm: float = 50.0

    # Carrier & Sampling Parameters
    carrier_freq_hz: float = 1090e6
    target_sample_rate_hz: float = 10e6      # 10 MS/s target rate for Mode-S PPM (10 samples/bit)
    default_raw_sample_rate_hz: float = 12.8e6  # Keysight VSA baseline for 10 MHz span
    speed_of_light_mps: float = 299792458.0

    # Online Streaming Window Parameters (replaces 1-second batch slicing)
    frame_duration_s: float = 120e-6         # 120 us Mode-S Extended Squitter frame
    iq_slice_buffer_s: float = 5e-6          # 5 us high-res IQ padding on either side of frame
    correlation_prominence: float = 0.0001
    correlation_min_distance_samples: int = 250


@dataclass
class AnomalyThresholds:
    """Physics, protocol, and RF physical-layer thresholds calibrated for rooftop VSA captures."""
    # Kinematic plausibility bounds
    max_groundspeed_kt: float = 850.0        # Subsonic + short-dt CPR quantization margin
    max_altitude_ft: float = 60000.0         # Maximum plausible barometric altitude
    min_altitude_ft: float = -1000.0         # Minimum plausible barometric altitude
    max_vertical_rate_fpm: float = 10000.0   # Maximum plausible climb/descent rate (ft/min)
    max_altitude_rate_fpm: float = 15000.0   # Implied Δaltitude / Δt limit between frames

    # RF Physical-Layer plausibility bounds (matches Fig. 4.2 of TU Wien thesis: 20 to 70 dBm)
    min_eirp_dbm: float = 20.0               # Allows overhead collinear antenna null & fading
    max_eirp_dbm: float = 70.0               # Allows low-elevation gain + ground reflection
    max_abs_cfo_hz: float = 75000.0          # Max single-frame CFO jump from aircraft median (Hz)
    max_abs_timing_ppm: float = 650.0        # Allows 10 MS/s multipath correlation jitter

    # Machine Learning & RF Fingerprinting thresholds
    rf_mismatch_prob_threshold: float = 0.10 # Flag only when claimed aircraft prob is very low
    outlier_z_threshold: float = 4.5         # 4.5-sigma Mahalanobis gate for real rooftop noise
    isolation_forest_contamination: float = 0.02


def parse_vsa_settings(setx_path: str | Path, config: Optional[ReceiverConfig] = None) -> ReceiverConfig:
    """
    Parses a Keysight 89600 VSA (.setx) XML setup file to extract center frequency,
    span, and native sample rate (in VSA, Fs = 1.28 * Span).
    """
    cfg = config or ReceiverConfig()
    path = Path(setx_path)
    if not path.exists():
        return cfg

    try:
        tree = ET.parse(path)
        root = tree.getroot()
        input_center = None
        input_span = None

        for elem in root.iter():
            key = elem.attrib.get("key", "")
            val = elem.attrib.get("value", "")
            if key == "InputCenter" and val:
                input_center = float(val)
            elif key == "InputSpan" and val:
                input_span = float(val)

        if input_center is not None and input_center > 0:
            cfg.carrier_freq_hz = input_center
        if input_span is not None and input_span > 0:
            # Keysight 89600 VSA uses a 1.28 oversampling factor relative to frequency span
            cfg.default_raw_sample_rate_hz = input_span * 1.28
    except Exception as exc:
        print(f"[Config] Warning: Could not parse VSA settings from {path.name}: {exc}")

    return cfg