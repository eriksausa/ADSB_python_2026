"""
Centralized, Explicitly Cited Domain & Modeling Assumptions (`src/adsb_rf/assumptions.py`).

Consolidates all physical constants, receiver hardware calibration parameters, ICAO Annex 10 /
RTCA DO-260B protocol rules, DSP thresholds, and Machine Learning hyperparameters into
documented dataclasses with literature and hardware citations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class Assumptions:
    """
    Master reference of physical constants, DSP parameters, protocol tolerances, and ML
    hyperparameters used across the 1090 MHz ADS-B RF Fingerprinting pipeline.

    Every constant is explicitly documented and cited to prevent magic numbers.
    """

    # --- 1. Fundamental Physical & Geodetic Constants ---
    speed_of_light_mps: float = 299_792_458.0          # CODATA 2018 exact speed of light in vacuum (m/s)
    earth_mean_radius_km: float = 6371.0088            # IUGG / WGS-84 authalic mean spherical Earth radius (km)
    knots_to_mps: float = 0.514444444                  # Exact nautical mile (1852 m) / 3600 s conversion
    feet_to_meters: float = 0.3048                     # International foot definition (m)

    # --- 2. ICAO Annex 10 Vol. IV / RTCA DO-260B Mode-S DF17 Protocol Constants ---
    carrier_freq_hz: float = 1090e6                    # ICAO Annex 10 Vol. IV §3.1.2.1 Mode-S downlink frequency (1090 MHz)
    target_sample_rate_hz: float = 10e6                # 10 MS/s integer multiple of 2 MHz Manchester chip rate (5 samples/chip)
    vsa_oversampling_ratio: float = 1.28               # Keysight 89600 VSA FFT sample-rate factor: Fs = 1.28 * Span
    default_raw_sample_rate_hz: float = 12.8e6         # Keysight UXA N9040B baseline for 10 MHz frequency span (12.8 MS/s)
    frame_duration_s: float = 120e-6                   # RTCA DO-260B 120 us Extended Squitter (8 us preamble + 112 us payload)
    preamble_duration_s: float = 8e-6                  # 8 us Mode-S preamble (16 half-microsecond chips)
    payload_bits: int = 112                            # 112-bit Extended Squitter payload (56-bit ME + 24-bit CRC parity)
    crc24_generator_hex: int = 0xFFF409                # ICAO Annex 10 Vol. IV §3.1.2.3.3.2 CRC-24 polynomial G(x)
    cpr_lat_zones_nz: float = 15.0                     # RTCA DO-260B Compact Position Reporting (CPR) Nz = 15 (60 even / 59 odd zones)
    cpr_polar_boundary_deg: float = 87.0               # CPR single-longitude-zone polar latitude boundary (deg)
    cpr_quantization_deadband_km: float = 0.015        # 17-bit CPR grid deadband (~5.1 m lat, ~7.5 m lon -> 15 m combined tolerance)
    alt_quantization_deadband_ft: float = 25.0         # Gillham / Q-bit=1 barometric altitude quantization step (25 ft)

    # --- 3. TU Wien Rooftop Receiver Hardware Calibration (Šauša, TU Wien B.Sc. Thesis, 2019) ---
    rx_lat: float = 48.196674                          # TU Wien Institute of Telecommunications (Gußhausstraße 25, Vienna) roof latitude
    rx_lon: float = 16.370998                          # TU Wien Institute of Telecommunications roof longitude
    antenna_gain_db: float = 5.5                       # 1090 MHz vertical collinear rooftop antenna gain (dBi)
    cable_loss_db: float = 11.37                       # Measured coaxial feedline + front-end insertion loss at 1090 MHz (dB)
    reference_impedance_ohm: float = 50.0              # Keysight UXA N9040B RF input termination impedance (Ohms)

    # --- 4. DSP Detection & Feature Extraction Thresholds ---
    iq_slice_buffer_s: float = 5e-6                    # Guard buffer around 120 us frame for high-res I/Q slicing
    correlation_prominence: float = 1e-4               # Minimum preamble cross-correlation peak prominence
    correlation_min_distance_samples: int = 500        # 50 us minimum peak separation at 10 MS/s to reject multipath sidelobes
    blind_ppm_contrast_ratio: float = 2.2              # Minimum half-chip energy ratio (E_max / E_min) for uncorrupted slot mask
    blind_preamble_xcorr_min: float = 0.75             # Minimum normalized preamble correlation for CRC-failed burst gating
    blind_clean_slot_ratio_min: float = 0.80           # Minimum fraction of confident PPM slots required for blind attribution
    cfo_phase_delta_std_gate_rad: float = 0.45         # Legacy 1.0 us inter-slot phase spread threshold before unwrap-free fallback
    active_pulse_percentile: float = 55.0              # Envelope percentile separating active PPM pulses (~47% duty cycle) from noise
    noise_floor_percentile: float = 35.0               # Envelope percentile isolating off-chip baseline thermal noise

    # --- 5. Kinematic & Physical-Layer Anomaly Gate Thresholds ---
    max_groundspeed_kt: float = 850.0                  # Subsonic airliner upper bound + short-dt CPR quantization margin (kt)
    min_position_jump_km: float = 1.5                  # Minimum horizontal jump before flagging implied groundspeed (km)
    max_altitude_ft: float = 60000.0                   # Commercial aviation ceiling + safety margin (ft)
    min_altitude_ft: float = -1000.0                   # Dead Sea / high-pressure barometric floor per Mode-S encoding (ft)
    max_vertical_rate_fpm: float = 10000.0             # Maximum plausible commercial climb/descent rate (ft/min)
    max_altitude_rate_fpm: float = 15000.0             # Maximum implied delta-altitude / delta-t between consecutive frames (ft/min)
    min_eirp_dbm: float = 20.0                         # Rooftop lower EIRP bound accounting for overhead collinear null & banking fade (dBm)
    max_eirp_dbm: float = 70.0                         # Rooftop upper EIRP bound accounting for directive gain & constructive reflection (dBm)
    max_abs_cfo_hz: float = 450_000.0                  # Absolute Mode-S CFO sanity gate (Hz); ICAO Annex 10 allows +/- 1 MHz max
    max_cfo_jump_from_median_hz: float = 75_000.0      # Maximum single-frame CFO jump from aircraft's own median oscillator offset (Hz)
    max_abs_timing_ppm: float = 650.0                  # Symbol clock timing offset sanity gate under urban multipath (ppm)

    # --- 6. Machine Learning & Statistical Spoofing Detection Hyperparameters ---
    min_frames_enrolled: int = 5                       # Minimum clean frames per aircraft for 5-fold stratified evaluation (115 classes)
    min_frames_well_enrolled: int = 12                 # Minimum clean frames for well-enrolled benchmark tier (68 classes, ~70 classes)
    outlier_z_threshold: float = 4.5                   # Weighted Mahalanobis distance gate (sigma) for spoofing / outlier rejection
    secondary_mismatch_z: float = 3.2                  # Secondary Mahalanobis threshold when burst matches a different aircraft
    rf_mismatch_prob_threshold: float = 0.10           # Claimed-aircraft posterior probability floor before flagging mismatch
    sample_weight_damping_exp: float = 0.30            # Sub-linear class-imbalance damping exponent: w_c = clip((N_med/N_c)^0.30, 0.5, 2.5)
    ensemble_weights: Tuple[float, float, float] = (0.45, 0.35, 0.20)  # Soft-voting weights: (RandomForest, ExtraTrees, Mahalanobis)
    prior_log_weight: float = 0.15                     # Damped log-prior weight in Bayesian Mahalanobis posterior
    isolation_forest_contamination: float = 0.02       # Expected unsupervised outlier fraction in IsolationForest

    # Physical measurement resolution floors for the 12 features on a 10 MS/s receiver
    # Order matches RF_FEATURE_NAMES in features.py
    rf_physical_std_floor: Tuple[float, ...] = (
        1200.0,  # 0: CFO_Raw_Hz (Hz) - single-burst thermal & multipath frequency estimation floor
        1000.0,  # 1: Frequency_shift (Hz) - Doppler-compensated CFO resolution floor
        25.0,    # 2: Timing_Offset_PPM (ppm) - 0.015-sample parabolic lag floor over 600 samples at 10 MS/s
        8.0,     # 3: Sub_Sample_Lag_Ns (ns) - fractional arrival time interpolation floor (0.08 samples)
        1.5,     # 4: Power_dBm (dB) - short-term ricean/multipath amplitude fading floor
        2.0,     # 5: EIRP_dBm (dB) - FSPL + antenna elevation pattern variance floor
        0.15,    # 6: Phase_Noise_Std (rad) - pulse-to-pulse phase residual floor
        0.35,    # 7: PA_Droop_dB (dB) - power amplifier capacitor droop estimation floor
        0.10,    # 8: Pulse_Crest_Factor (ratio) - 95th-percentile to median envelope floor
        0.12,    # 9: Phase_Curvature_Rad (rad) - half-frame phase slope difference floor
        2.5,     # 10: SNR_dB (dB) - in-frame pulse-to-noise ratio variance floor
        0.35,    # 11: Preamble_Ratio_dB (dB) - preamble-to-payload amplitude balance floor
    )

    # Diagonal feature weights in the weighted Mahalanobis distance spoofing gate (sum = 1.00)
    rf_mahalanobis_weights: Tuple[float, ...] = (
        0.10,    # 0: CFO_Raw_Hz
        0.35,    # 1: Frequency_shift
        0.28,    # 2: Timing_Offset_PPM
        0.01,    # 3: Sub_Sample_Lag_Ns
        0.07,    # 4: Power_dBm
        0.10,    # 5: EIRP_dBm
        0.03,    # 6: Phase_Noise_Std
        0.02,    # 7: PA_Droop_dB
        0.01,    # 8: Pulse_Crest_Factor
        0.01,    # 9: Phase_Curvature_Rad
        0.01,    # 10: SNR_dB
        0.01,    # 11: Preamble_Ratio_dB
    )


DEFAULT_ASSUMPTIONS = Assumptions()


@dataclass
class ReceiverConfig:
    """Hardware and DSP configuration for the ADS-B receiver station, initialized from `Assumptions`."""

    rx_lat: float = DEFAULT_ASSUMPTIONS.rx_lat
    rx_lon: float = DEFAULT_ASSUMPTIONS.rx_lon
    antenna_gain_db: float = DEFAULT_ASSUMPTIONS.antenna_gain_db
    cable_loss_db: float = DEFAULT_ASSUMPTIONS.cable_loss_db
    reference_impedance_ohm: float = DEFAULT_ASSUMPTIONS.reference_impedance_ohm
    carrier_freq_hz: float = DEFAULT_ASSUMPTIONS.carrier_freq_hz
    target_sample_rate_hz: float = DEFAULT_ASSUMPTIONS.target_sample_rate_hz
    default_raw_sample_rate_hz: float = DEFAULT_ASSUMPTIONS.default_raw_sample_rate_hz
    speed_of_light_mps: float = DEFAULT_ASSUMPTIONS.speed_of_light_mps
    earth_mean_radius_km: float = DEFAULT_ASSUMPTIONS.earth_mean_radius_km
    frame_duration_s: float = DEFAULT_ASSUMPTIONS.frame_duration_s
    iq_slice_buffer_s: float = DEFAULT_ASSUMPTIONS.iq_slice_buffer_s
    correlation_prominence: float = DEFAULT_ASSUMPTIONS.correlation_prominence
    correlation_min_distance_samples: int = DEFAULT_ASSUMPTIONS.correlation_min_distance_samples


@dataclass
class AnomalyThresholds:
    """Physics, protocol, and RF physical-layer thresholds initialized from `Assumptions`."""

    max_groundspeed_kt: float = DEFAULT_ASSUMPTIONS.max_groundspeed_kt
    min_position_jump_km: float = DEFAULT_ASSUMPTIONS.min_position_jump_km
    max_altitude_ft: float = DEFAULT_ASSUMPTIONS.max_altitude_ft
    min_altitude_ft: float = DEFAULT_ASSUMPTIONS.min_altitude_ft
    max_vertical_rate_fpm: float = DEFAULT_ASSUMPTIONS.max_vertical_rate_fpm
    max_altitude_rate_fpm: float = DEFAULT_ASSUMPTIONS.max_altitude_rate_fpm
    min_eirp_dbm: float = DEFAULT_ASSUMPTIONS.min_eirp_dbm
    max_eirp_dbm: float = DEFAULT_ASSUMPTIONS.max_eirp_dbm
    max_abs_cfo_hz: float = DEFAULT_ASSUMPTIONS.max_cfo_jump_from_median_hz
    max_absolute_cfo_limit_hz: float = DEFAULT_ASSUMPTIONS.max_abs_cfo_hz
    max_abs_timing_ppm: float = DEFAULT_ASSUMPTIONS.max_abs_timing_ppm
    rf_mismatch_prob_threshold: float = DEFAULT_ASSUMPTIONS.rf_mismatch_prob_threshold
    outlier_z_threshold: float = DEFAULT_ASSUMPTIONS.outlier_z_threshold
    secondary_mismatch_z: float = DEFAULT_ASSUMPTIONS.secondary_mismatch_z
    isolation_forest_contamination: float = DEFAULT_ASSUMPTIONS.isolation_forest_contamination


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
            cfg.default_raw_sample_rate_hz = input_span * DEFAULT_ASSUMPTIONS.vsa_oversampling_ratio
    except Exception as exc:
        print(f"[Config] Warning: Could not parse VSA settings from {path.name}: {exc}")

    return cfg


def get_assumptions_table_rows() -> List[Dict[str, str]]:
    """Returns a structured table of domain and modeling assumptions with citations for UI/README display."""
    a = DEFAULT_ASSUMPTIONS
    return [
        {
            "Parameter": "Carrier Frequency & Sample Rate",
            "Value": f"{a.carrier_freq_hz/1e6:.0f} MHz | {a.default_raw_sample_rate_hz/1e6:.2f} -> {a.target_sample_rate_hz/1e6:.1f} MS/s",
            "Domain Justification & Citation": "ICAO Annex 10 Vol. IV §3.1.2.1 (1090 MHz Mode-S); Keysight 89600 VSA Fs = 1.28 * 10 MHz span.",
        },
        {
            "Parameter": "Mode-S CRC-24 Generator Polynomial",
            "Value": f"G(x) = 0x{a.crc24_generator_hex:06X} (degree 24)",
            "Domain Justification & Citation": "RTCA DO-260B §2.2.3 / ICAO Annex 10 modulo-2 parity check over 88 payload + 24 parity bits.",
        },
        {
            "Parameter": "Receiver Front-End Calibration",
            "Value": f"G_rx = +{a.antenna_gain_db:.1f} dBi, L_sys = {a.cable_loss_db:.2f} dB, Z_0 = {a.reference_impedance_ohm:.0f} Ohm",
            "Domain Justification & Citation": "TU Wien Institute of Telecommunications rooftop collinear antenna & coax insertion loss calibration (Šauša, 2019).",
        },
        {
            "Parameter": "CPR Position & Altitude Deadbands",
            "Value": f"15 m horizontal ({a.cpr_quantization_deadband_km:.3f} km), {a.alt_quantization_deadband_ft:.0f} ft vertical",
            "Domain Justification & Citation": "RTCA DO-260B 17-bit CPR grid resolution (~5.1 m lat, ~7.5 m lon) and Q-bit=1 25 ft Gillham/binary step.",
        },
        {
            "Parameter": "Rooftop EIRP Plausibility Window",
            "Value": f"[{a.min_eirp_dbm:.0f}, {a.max_eirp_dbm:.0f}] dBm",
            "Domain Justification & Citation": "Nominal Mode-S TX is 48.5–57 dBm (70–500 W, ICAO Annex 10); widened for collinear zenith null & ground reflection.",
        },
        {
            "Parameter": "Two-Stage Unwrap-Free CFO Estimator",
            "Value": "0.1 us coarse (+/- 5 MHz) + 1.0/1.5 us fine unit-phasor",
            "Domain Justification & Citation": "Eliminates +/- pi phase wrapping across 1 us pulse gaps when |CFO| > 250 kHz (Kay, 1989; Moose, 1994).",
        },
        {
            "Parameter": "Mahalanobis Spoofing Gate & Robust Spread",
            "Value": f"d_M > {a.outlier_z_threshold:.1f} sigma (MAD + P80-P20 robust scale)",
            "Domain Justification & Citation": "50% breakdown-point Median/MAD centroid resists training poisoning; physical std floors bound 10 MS/s quantization noise.",
        },
        {
            "Parameter": "Hybrid Ensemble Soft-Voting Weights",
            "Value": "45% RandomForest + 35% ExtraTrees + 20% Mahalanobis",
            "Domain Justification & Citation": "Combines nonlinear multi-class decision boundaries (RF/ET) with open-set Gaussian distance rejection (Mahalanobis).",
        },
    ]
