"""
Digital Signal Processing (DSP) and Physical-Layer RF Feature Extraction (`src/adsb_rf/dsp.py`).

Performs:
- Mode-S preamble cross-correlation & PPM bit slicing (`decode_ppm`)
- 24-bit CRC parity verification (`is_valid`, G(x) = 0xFFF409)
- Two-stage unwrap-free Carrier Frequency Offset (CFO) & phase noise estimation
- Sub-sample parabolic correlation lag & symbol clock crystal drift (`estimate_timing_offset`)
- 5 hardware transient waveform features (`extract_waveform_transient_features`)
- Calibrated RF power, Free-Space Path Loss EIRP, and spherical kinematic Doppler shift
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks

from .assumptions import DEFAULT_ASSUMPTIONS, ReceiverConfig
from .stream_reader import IQStreamChunk


def decode_ppm(envelope_frame: np.ndarray, sample_rate: float = 10e6) -> np.ndarray:
    """
    Decodes 112 Mode-S Pulse Position Modulation (PPM) payload bits from a
    resampled 120 us signal envelope (1,200 samples at 10 MHz).
    """
    samples_per_half_bit = int(sample_rate / (2 * 1e6))
    ppm_one = np.repeat(np.array([1.0, -1.0]), samples_per_half_bit)

    starting_point = int(DEFAULT_ASSUMPTIONS.preamble_duration_s * sample_rate)
    bit_length = int(sample_rate * 1e-6)

    payload_samples = envelope_frame[starting_point : starting_point + (112 * bit_length)]
    input_matrix = payload_samples.reshape(112, bit_length)

    correlated = np.dot(input_matrix, ppm_one)
    msg = (correlated > 0).astype(int)
    return msg.reshape(1, 112)


def is_valid(msg_bits: np.ndarray) -> bool:
    """
    Verifies 24-bit Mode-S CRC parity using modulo-2 polynomial division
    with generator polynomial G(x) = 0xFFF409 (ICAO Annex 10 Vol. IV / RTCA DO-260B).
    """
    poly = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        dtype=bool,
    )
    bits = np.squeeze(msg_bits).astype(bool).copy()
    if len(bits) < 112:
        return False

    for i in range(88):
        if bits[i]:
            bits[i : i + 25] = np.logical_xor(bits[i : i + 25], poly)

    parity_remainder = bits[88:112]
    return not np.any(parity_remainder)


def smooth_3(x: np.ndarray) -> np.ndarray:
    """3-point moving average filter matching MATLAB smooth(x, 3)."""
    return np.convolve(x, np.ones(3) / 3.0, mode="same")


def build_ideal_10mhz_mask(msg_bits: np.ndarray) -> np.ndarray:
    """Reconstructs the ideal 1200-sample (10 MHz) binary pulse mask for preamble + 112 PPM bits."""
    msg_flat = np.squeeze(msg_bits).astype(int)
    preamble_10mhz = np.repeat(
        np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]), 5
    )
    ppm_pairs = np.column_stack((msg_flat, 1 - msg_flat)).flatten()
    msg_10mhz = np.repeat(ppm_pairs, 5)
    return np.concatenate([preamble_10mhz, msg_10mhz])


def build_blind_energy_mask_10mhz(envelope_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a 1200-sample active pulse mask without trusting CRC-validated bits:
    1. Uses the fixed 8 us Mode-S preamble (`1010000101000000`) for samples 0..80.
    2. For each 1 us symbol slot in the payload, selects the stronger 0.5 us half-chip
       only if its energy contrast exceeds `blind_ppm_contrast_ratio` (2.2x).
    """
    preamble_10mhz = np.repeat(
        np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]), 5
    )
    env = np.abs(envelope_frame[:1200])
    payload_slots = env[80:1200].reshape(112, 10)
    first_half_energy = payload_slots[:, :5].sum(axis=1)
    second_half_energy = payload_slots[:, 5:].sum(axis=1)

    max_e = np.maximum(first_half_energy, second_half_energy)
    min_e = np.maximum(np.minimum(first_half_energy, second_half_energy), 1e-12)
    contrast_ratio = max_e / min_e

    confident_slots = contrast_ratio > DEFAULT_ASSUMPTIONS.blind_ppm_contrast_ratio
    decision_bits = (first_half_energy >= second_half_energy).astype(int)

    ppm_pairs = np.column_stack((decision_bits, 1 - decision_bits))
    ppm_pairs[~confident_slots, :] = 0
    payload_mask_10mhz = np.repeat(ppm_pairs.flatten(), 5)

    return np.concatenate([preamble_10mhz, payload_mask_10mhz]), confident_slots


def calculate_amplitude(rx: np.ndarray, msg_bits: np.ndarray) -> float:
    """
    Estimates true RF pulse amplitude by averaging flat pulse-top samples and
    subtracting the off-pulse noise floor.
    """
    bits_modulated = build_ideal_10mhz_mask(msg_bits)
    bits_neg = 1 - bits_modulated

    bits_3 = smooth_3(bits_modulated.astype(float))
    bits_3_neg = smooth_3(bits_neg.astype(float))

    bits_3[bits_3 < 0.9] = 0.0
    bits_3_neg[bits_3_neg < 0.9] = 0.0

    bits_3_neg[0] = 0.0
    bits_3[0] = 0.0
    if len(bits_3) > 0 and bits_3[-1] == 1.0:
        bits_3[-1] = 0.0

    rx_abs = np.abs(rx[: len(bits_modulated)])
    noise_blocks = (rx_abs * bits_3_neg).reshape(-1, 10).sum(axis=1) / 3.0
    noise_blocks = noise_blocks[noise_blocks != 0]
    noise = float(np.mean(noise_blocks)) if len(noise_blocks) > 0 else 0.0

    signal_blocks = (rx_abs * bits_3).reshape(-1, 10).sum(axis=1) / 3.0
    signal_blocks = signal_blocks[signal_blocks != 0]
    rx_mean = float(np.mean(signal_blocks)) if len(signal_blocks) > 0 else 0.0

    return float(max(rx_mean - noise, 0.0))


def recover_cfo_unwrap_free(
    iq_res: Optional[np.ndarray],
    bits: Optional[np.ndarray] = None,
    fs_hz: float = 10e6,
    custom_mask_10mhz: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Two-stage unwrap-free Carrier Frequency Offset (CFO in Hz) and phase noise (rad) estimator:
      - Stage A (Coarse Intra-Pulse Delay-and-Multiply, Unambiguous Range = +/- 5.0 MHz):
        Computes `iq[n+1] * conj(iq[n])` strictly inside active 0.5 us pulses (`dt = 0.1 us`).
      - Stage B (Fine 1.0 us & 1.5 us Unit-Phasor Pulse-Pair Refinement after Derotation):
        Derotates the 1,200-sample burst by `f_coarse` so the residual offset sits near 0 Hz
        (far from the +/- pi branch cut) and computes the weighted 1.0 us and 1.5 us slot phase slope.
    """
    if iq_res is None or len(iq_res) < 400:
        return 0.0, 0.5

    iq = np.asarray(iq_res[:1200], dtype=np.complex128)
    L = len(iq)

    if custom_mask_10mhz is not None and len(custom_mask_10mhz) >= L:
        mask_10mhz = custom_mask_10mhz[:L].astype(bool)
    elif bits is not None and len(np.squeeze(bits)) >= 112:
        mask_10mhz = build_ideal_10mhz_mask(bits)[:L].astype(bool)
    else:
        env = np.abs(iq)
        mask_10mhz = env >= np.percentile(env, DEFAULT_ASSUMPTIONS.active_pulse_percentile)

    same_pulse = mask_10mhz[:-1] & mask_10mhz[1:]
    if np.sum(same_pulse) < 10:
        return 0.0, 0.5

    conj_prod_1samp = iq[1:][same_pulse] * np.conj(iq[:-1][same_pulse])
    f_coarse_hz = float(np.angle(np.sum(conj_prod_1samp)) / (2.0 * np.pi * (1.0 / fs_hz)))

    t_vec = np.arange(L, dtype=np.float64) / fs_hz
    iq_derot = iq * np.exp(-1j * 2.0 * np.pi * f_coarse_hz * t_vec)

    n_slots = L // 10
    iq_slots = (iq_derot[: n_slots * 10] * mask_10mhz[: n_slots * 10]).reshape(n_slots, 10).sum(axis=1) / 5.0
    t_slots = (t_vec[: n_slots * 10] * mask_10mhz[: n_slots * 10]).reshape(n_slots, 10).sum(axis=1) / 5.0

    valid_slots = np.abs(iq_slots) > 1e-12
    rx_vals = iq_slots[valid_slots]
    t_vals = t_slots[valid_slots]

    if len(rx_vals) < 4:
        return f_coarse_hz, 0.5

    unit_phasors = rx_vals / np.abs(rx_vals)
    dt_slots = np.round(np.diff(t_vals), 8)
    idx1_0 = np.where(dt_slots == 1.0e-6)[0]
    idx1_5 = np.where(dt_slots == 1.5e-6)[0]

    f_fine_hz = 0.0
    phase_noise_std = 0.5
    if len(idx1_0) > 0:
        p1_0 = unit_phasors[idx1_0 + 1] * np.conj(unit_phasors[idx1_0])
        f_1_0 = float(np.angle(np.sum(p1_0)) / (2.0 * np.pi * 1.0e-6))
        phase_noise_std = float(np.std(np.angle(p1_0)))
        if len(idx1_5) > 0:
            p1_5 = unit_phasors[idx1_5 + 1] * np.conj(unit_phasors[idx1_5])
            f_1_5 = float(np.angle(np.sum(p1_5)) / (2.0 * np.pi * 1.5e-6))
            f_fine_hz = (f_1_0 * len(idx1_0) + f_1_5 * len(idx1_5)) / (len(idx1_0) + len(idx1_5))
        else:
            f_fine_hz = f_1_0

    return float(f_coarse_hz + f_fine_hz), phase_noise_std


def calculate_freq_shift_and_phase_noise(
    rx_complex: np.ndarray,
    msg_bits: np.ndarray,
    custom_mask_10mhz: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Estimates raw Carrier Frequency Offset (CFO in Hz) and unwrapped phase noise standard
    deviation (rad) from the 10 MHz complex IQ frame. Automatically falls back to the
    two-stage unwrap-free estimator when high CFO or multipath fading exceeds the legacy
    `std(phasedelta1_0) < 0.45` gate.
    """
    bits_modulated_10mhz = (
        custom_mask_10mhz
        if custom_mask_10mhz is not None
        else build_ideal_10mhz_mask(msg_bits)
    )
    t_10mhz = np.arange(1200) * 0.1e-6 + 0.05e-6
    rx_slice = rx_complex[:1200]

    rx_masked = rx_slice * bits_modulated_10mhz
    t_masked = t_10mhz * bits_modulated_10mhz

    rx_blocks = rx_masked.reshape(-1, 10).sum(axis=1) / 5.0
    t_blocks = t_masked.reshape(-1, 10).sum(axis=1) / 5.0

    if len(rx_blocks) > 0 and rx_blocks[0] != 0:
        rx_blocks = rx_blocks * np.exp(-1j * np.angle(rx_blocks[0]))

    t_blocks = t_blocks - t_blocks[0]

    valid_mask = rx_blocks != 0
    rx_meanvalues = rx_blocks[valid_mask]
    t_meanvalues = t_blocks[valid_mask]

    if len(rx_meanvalues) >= 2:
        time_diffs = np.round(np.diff(t_meanvalues), 8)
        idx1_0 = np.where(time_diffs == 1.0e-6)[0]
        idx1_5 = np.where(time_diffs == 1.5e-6)[0]

        if len(idx1_0) > 0:
            phasedelta1_0 = rx_meanvalues[idx1_0 + 1] / rx_meanvalues[idx1_0]
            phasedelta1_5 = (
                rx_meanvalues[idx1_5 + 1] / rx_meanvalues[idx1_5]
                if len(idx1_5) > 0
                else np.array([])
            )

            if np.std(phasedelta1_0) < 0.45:
                f_measured1_0 = np.angle(np.mean(phasedelta1_0)) / (2.0 * np.pi * 1.0e-6)
                if len(phasedelta1_5) > 0:
                    f_measured1_5 = np.angle(np.mean(phasedelta1_5)) / (2.0 * np.pi * 1.5e-6)
                    f_estimate_initial = (
                        f_measured1_0 * len(idx1_0) + f_measured1_5 * len(idx1_5)
                    ) / (len(idx1_0) + len(idx1_5))
                else:
                    f_estimate_initial = f_measured1_0

                # Guard against 2*pi branch-cut aliasing on 1.0 us (+/-500 kHz) and 1.5 us (+/-333 kHz)
                same_pulse = (bits_modulated_10mhz[:-1] == 1) & (bits_modulated_10mhz[1:] == 1)
                if np.sum(same_pulse) >= 10:
                    conj_1 = rx_slice[1:][same_pulse] * np.conj(rx_slice[:-1][same_pulse])
                    f_coarse_check = float(np.angle(np.sum(conj_1)) / (2.0 * np.pi * 0.1e-6))
                else:
                    f_coarse_check = float(f_estimate_initial)

                if abs(f_estimate_initial - f_coarse_check) < 150e3 and abs(f_coarse_check) < 300e3:
                    rx_fcorr = rx_slice * np.exp(-1j * 2.0 * np.pi * f_estimate_initial * t_10mhz)
                    active_mask = bits_modulated_10mhz == 1
                    rx_part = rx_fcorr[active_mask]
                    t_part = t_10mhz[active_mask]

                    rx_part_angle_unwrapped = np.unwrap(np.angle(rx_part))
                    p = np.polyfit(t_part, rx_part_angle_unwrapped, deg=1)
                    residuals = rx_part_angle_unwrapped - np.polyval(p, t_part)
                    phase_noise_std = float(np.std(residuals))

                    f_estimate_2 = p[0] / (2.0 * np.pi)
                    return float(f_estimate_initial + f_estimate_2), phase_noise_std

    # Unwrap-free recovery when legacy gate fails
    return recover_cfo_unwrap_free(rx_slice, msg_bits, fs_hz=10e6, custom_mask_10mhz=bits_modulated_10mhz)


def calculate_freq_shift(rx_complex: np.ndarray, msg_bits: np.ndarray) -> float:
    """Backwards-compatible wrapper returning raw CFO in Hz."""
    cfo_hz, _ = calculate_freq_shift_and_phase_noise(rx_complex, msg_bits)
    return cfo_hz


def _sub_sample_corr_lag(sig_segment: np.ndarray, template_segment: np.ndarray) -> float:
    """
    Computes sub-sample lag (in samples) between `sig_segment` and `template_segment`
    using zero-mean cross-correlation at lags [-1, 0, +1] and parabolic peak interpolation.
    """
    s = sig_segment - np.mean(sig_segment)
    t = template_segment - np.mean(template_segment)
    r_zero = float(np.dot(s, t))
    r_minus = float(np.dot(s[:-1], t[1:]))
    r_plus = float(np.dot(s[1:], t[:-1]))
    denom = r_minus - 2.0 * r_zero + r_plus
    if abs(denom) > 1e-12:
        return float(np.clip(0.5 * (r_minus - r_plus) / denom, -1.0, 1.0))
    return 0.0


def estimate_timing_offset(
    envelope_frame: np.ndarray,
    msg_bits: np.ndarray,
    corr_triplet: Optional[Tuple[float, float, float]] = None,
    sample_rate: float = 10e6,
    custom_mask_10mhz: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Extracts two physical-layer timing features:
    1. `timing_offset_ppm`: Symbol clock frequency skew (ppm) across the 120 us burst.
    2. `sub_sample_lag_ns`: Sub-sample arrival time offset (ns) around the preamble correlation peak.
    """
    sub_sample_lag_ns = 0.0
    if corr_triplet is not None:
        y_minus, y_zero, y_plus = corr_triplet
        denom = y_minus - 2.0 * y_zero + y_plus
        if abs(denom) > 1e-12:
            delta_samples = float(np.clip(0.5 * (y_minus - y_plus) / denom, -0.5, 0.5))
            sub_sample_lag_ns = delta_samples * (1e9 / sample_rate)

    ideal_mask = (
        custom_mask_10mhz.astype(float)
        if custom_mask_10mhz is not None
        else build_ideal_10mhz_mask(msg_bits).astype(float)
    )
    env = np.abs(envelope_frame[:1200]).astype(float)

    lag_first_samples = _sub_sample_corr_lag(env[:600], ideal_mask[:600])
    lag_second_samples = _sub_sample_corr_lag(env[600:1200], ideal_mask[600:1200])

    timing_offset_ppm = float(-((lag_second_samples - lag_first_samples) / 600.0) * 1e6)
    return timing_offset_ppm, sub_sample_lag_ns


def extract_waveform_transient_features(
    iq_res: Optional[np.ndarray],
) -> Tuple[float, float, float, float, float]:
    """
    Extracts 5 hardware-specific RF transmitter transient features directly from the 1,200-sample
    (10 MS/s, 120 us) complex baseband waveform `iq_res`:
      1. `pa_droop_db`: Power Amplifier (PA) storage-capacitor voltage sag from first 30 us to last 30 us (dB).
      2. `pulse_crest_factor`: 95th-percentile peak to median active pulse top ratio (PA turn-on overshoot).
      3. `phase_curvature_rad`: Quadratic phase chirp coefficient across the 120 us burst (PA thermal pulling).
      4. `snr_db`: In-frame active pulse to off-chip noise floor ratio (dB).
      5. `preamble_ratio_db`: Preamble-to-payload pulse amplitude balance (dB).
    """
    if iq_res is None or len(iq_res) < 400:
        return 0.0, 1.2, 0.0, 20.0, 0.0

    iq = np.asarray(iq_res[:1200], dtype=np.complex128)
    env = np.abs(iq)
    max_env = float(np.max(env))
    if max_env <= 1e-12:
        return 0.0, 1.2, 0.0, 0.0, 0.0

    q55 = float(np.percentile(env, DEFAULT_ASSUMPTIONS.active_pulse_percentile))
    q35 = float(np.percentile(env, DEFAULT_ASSUMPTIONS.noise_floor_percentile))
    active_mask = env >= q55
    noise_mask = env <= q35

    active_vals = env[active_mask]
    noise_vals = env[noise_mask]
    mean_sig = float(np.median(active_vals)) if len(active_vals) > 0 else max_env
    mean_noise = float(np.median(noise_vals)) if len(noise_vals) > 0 else (mean_sig * 0.05)

    snr_db = float(20.0 * np.log10(max(mean_sig, 1e-12) / max(mean_noise, 1e-12)))
    snr_db = float(np.clip(snr_db, 0.0, 60.0))

    head_env = env[:300]
    tail_env = env[-300:]
    head_active = head_env[head_env >= np.percentile(head_env, 60)]
    tail_active = tail_env[tail_env >= np.percentile(tail_env, 60)]
    head_amp = float(np.median(head_active)) if len(head_active) > 0 else mean_sig
    tail_amp = float(np.median(tail_active)) if len(tail_active) > 0 else mean_sig
    pa_droop_db = float(20.0 * np.log10(max(head_amp, 1e-12) / max(tail_amp, 1e-12)))
    pa_droop_db = float(np.clip(pa_droop_db, -6.0, 6.0))

    p95 = float(np.percentile(active_vals, 95)) if len(active_vals) > 0 else mean_sig
    pulse_crest_factor = float(np.clip(p95 / max(mean_sig, 1e-12), 1.0, 3.5))

    if len(env) >= 80:
        pre_pulses = np.concatenate([env[0:5], env[10:15], env[35:40], env[45:50]])
        pre_amp = float(np.median(pre_pulses))
        preamble_ratio_db = float(np.clip(20.0 * np.log10(max(pre_amp, 1e-12) / max(mean_sig, 1e-12)), -6.0, 6.0))
    else:
        preamble_ratio_db = 0.0

    half = len(iq) // 2
    iq_first = iq[:half]
    iq_second = iq[half:]
    act_1 = np.abs(iq_first[:-1]) >= q55
    act_2 = np.abs(iq_second[:-1]) >= q55
    if np.sum(act_1) >= 10 and np.sum(act_2) >= 10:
        dphi_1 = np.angle(iq_first[1:][act_1] * np.conj(iq_first[:-1][act_1]))
        dphi_2 = np.angle(iq_second[1:][act_2] * np.conj(iq_second[:-1][act_2]))
        phase_curvature_rad = float(np.clip((np.median(dphi_2) - np.median(dphi_1)) * 10.0, -2.0, 2.0))
    else:
        phase_curvature_rad = 0.0

    return pa_droop_db, pulse_crest_factor, phase_curvature_rad, snr_db, preamble_ratio_db


def extract_blind_rf_features(
    iq_resampled: np.ndarray,
    envelope_frame: np.ndarray,
    demod_bits: np.ndarray,
    corr_triplet: Optional[Tuple[float, float, float]] = None,
    sample_rate: float = 10e6,
) -> Dict[str, float]:
    """
    Extracts physical-layer RF fingerprint features from a CRC-failed burst by masking out
    low-contrast/collided bit slots.
    """
    blind_mask, confident_slots = build_blind_energy_mask_10mhz(envelope_frame)
    cfo_raw_hz, phase_noise_std = calculate_freq_shift_and_phase_noise(
        iq_resampled, demod_bits, custom_mask_10mhz=blind_mask
    )
    timing_offset_ppm, sub_sample_lag_ns = estimate_timing_offset(
        envelope_frame,
        demod_bits,
        corr_triplet=corr_triplet,
        sample_rate=sample_rate,
        custom_mask_10mhz=blind_mask,
    )
    pa_droop_db, crest_factor, phase_curv, snr_db, pre_ratio_db = extract_waveform_transient_features(
        iq_resampled
    )
    return {
        "CFO_Raw_Hz": float(cfo_raw_hz),
        "Phase_Noise_Std": float(phase_noise_std),
        "Timing_Offset_PPM": float(timing_offset_ppm),
        "Sub_Sample_Lag_Ns": float(sub_sample_lag_ns),
        "PA_Droop_dB": float(pa_droop_db),
        "Pulse_Crest_Factor": float(crest_factor),
        "Phase_Curvature_Rad": float(phase_curv),
        "SNR_dB": float(snr_db),
        "Preamble_Ratio_dB": float(pre_ratio_db),
        "Clean_Slot_Ratio": float(np.mean(confident_slots)),
    }


def lldistkm(latlon1: List[float], latlon2: List[float]) -> Tuple[float, float]:
    """
    Computes Haversine distance (`d1km`) and equirectangular Pythagorean distance (`d2km`) in km
    using the IUGG mean spherical Earth radius (`6371.0088 km`).
    """
    radius = DEFAULT_ASSUMPTIONS.earth_mean_radius_km
    lat1, lon1 = np.radians(latlon1[0]), np.radians(latlon1[1])
    lat2, lon2 = np.radians(latlon2[0]), np.radians(latlon2[1])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d1km = radius * c

    x = delta_lon * np.cos((lat1 + lat2) / 2.0)
    y = delta_lat
    d2km = radius * np.sqrt(x * x + y * y)
    return float(d1km), float(d2km)


def compute_eirp_dbm(
    rx_power_dbm: float,
    distance_m: float,
    carrier_freq_hz: float = DEFAULT_ASSUMPTIONS.carrier_freq_hz,
) -> float:
    """
    Calculates estimated Equivalent Isotropically Radiated Power (EIRP in dBm)
    by adding Free-Space Path Loss (FSPL) to the calibrated received power:
        FSPL(dB) = 20 * log10(4 * pi * d / lambda)
    """
    if np.isnan(rx_power_dbm) or np.isnan(distance_m) or distance_m <= 0:
        return float("nan")
    lambda_c = DEFAULT_ASSUMPTIONS.speed_of_light_mps / carrier_freq_hz
    fspl_db = 20.0 * np.log10((4.0 * np.pi * distance_m) / lambda_c)
    return float(rx_power_dbm + fspl_db)


def compute_single_frame_doppler(
    lat: float,
    lon: float,
    alt_ft: float,
    airspeed_kt: float,
    heading_deg: float,
    config: ReceiverConfig,
) -> float:
    """
    Computes expected kinematic Doppler frequency shift (Hz) at 1090 MHz for a single
    position frame paired with the aircraft's velocity vector.
    """
    if (
        np.isnan(lat)
        or np.isnan(lon)
        or np.isnan(airspeed_kt)
        or np.isnan(heading_deg)
        or airspeed_kt <= 0
        or alt_ft <= 0
    ):
        return float("nan")

    vel_m_s = airspeed_kt * DEFAULT_ASSUMPTIONS.knots_to_mps
    lats, lons = config.rx_lat, config.rx_lon

    y = np.sin(np.radians(lons - lon)) * np.cos(np.radians(lats))
    x = (
        np.cos(np.radians(lat)) * np.sin(np.radians(lats))
        - np.sin(np.radians(lat)) * np.cos(np.radians(lats)) * np.cos(np.radians(lons - lon))
    )
    plane_to_ant_rad = np.arctan2(y, x)
    ant_to_plane = np.degrees(plane_to_ant_rad) + 180.0
    if ant_to_plane < 0:
        ant_to_plane += 360.0

    doppler_shift = (
        -np.cos(np.radians(ant_to_plane - heading_deg))
        * (vel_m_s / config.speed_of_light_mps)
        * config.carrier_freq_hz
    )
    return float(doppler_shift)


def find_valid_xcorr(
    chunk: IQStreamChunk | dict,
    start_time: Optional[float] = None,
    sample_rate: Optional[float] = None,
    antenna_gain: Optional[float] = None,
    amplifier_cable: Optional[float] = None,
    config: Optional[ReceiverConfig] = None,
    capture_crc_failures: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detects Mode-S DF17 frames via extended preamble cross-correlation, verifies CRC-24 parity,
    and extracts calibrated Rx power, high-res IQ slices, unwrap-free CFO, phase noise,
    timing offset, and all 5 hardware transient features.
    """
    cfg = config or ReceiverConfig()

    if isinstance(chunk, IQStreamChunk):
        Z = chunk.envelope_res
        Inphase_res = chunk.inphase_res
        Quadrature_res = chunk.quadrature_res
        iq_raw = chunk.iq_raw
        start_time_s = chunk.start_time_s if start_time is None else start_time
        fs_res = chunk.target_sample_rate_hz if sample_rate is None else sample_rate
        fs_raw = chunk.raw_sample_rate_hz
    else:
        Z = chunk["Z"]
        Inphase_res = chunk["Inphase"]
        Quadrature_res = chunk["Quadrature"]
        iq_raw = None
        start_time_s = start_time or 0.0
        fs_res = sample_rate or cfg.target_sample_rate_hz
        fs_raw = cfg.default_raw_sample_rate_hz

    ant_gain = cfg.antenna_gain_db if antenna_gain is None else antenna_gain
    cable_loss = cfg.cable_loss_db if amplifier_cable is None else amplifier_cable

    points = len(Z)
    msg_length = int(fs_res * cfg.frame_duration_s)
    if points < msg_length:
        return ([], []) if capture_crc_failures else []

    preamble = np.array([1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1])
    ppm_df_17 = np.array([1, -1, -1, 1, -1, 1, -1, 1, 1, -1])
    preamble_ext_17 = np.concatenate([preamble, ppm_df_17])

    adjust_message = int(1e-6 * fs_res / 2)
    preamble_ext_17 = np.repeat(preamble_ext_17, adjust_message)
    template_norm = float(np.linalg.norm(preamble_ext_17))
    len_template = len(preamble_ext_17)

    ral = np.correlate(Z, preamble_ext_17, mode="valid")
    peaks, _ = find_peaks(
        ral,
        distance=cfg.correlation_min_distance_samples,
        prominence=cfg.correlation_prominence,
    )
    valid_peaks = peaks[peaks <= (points - msg_length)]

    valid_struct_list: List[Dict[str, Any]] = []
    valid_peak_indices: List[int] = []
    failed_candidates_raw: List[Tuple[int, np.ndarray]] = []
    pow_icao = 2 ** np.arange(23, -1, -1)
    pow_df = 2 ** np.arange(4, -1, -1)
    pow_tc = 2 ** np.arange(4, -1, -1)

    for peak_idx in valid_peaks:
        end_idx = peak_idx + msg_length
        mes_env = Z[peak_idx:end_idx]

        mes_bits_2d = decode_ppm(mes_env, fs_res)
        bits_112 = mes_bits_2d[0, :112] if mes_bits_2d.ndim == 2 else mes_bits_2d[:112]
        crc_ok = is_valid(bits_112)

        if not crc_ok:
            if capture_crc_failures:
                failed_candidates_raw.append((int(peak_idx), bits_112))
            continue

        valid_peak_indices.append(int(peak_idx))

        if np.isclose(fs_res, 10e6):
            avg_amplitude = calculate_amplitude(mes_env, mes_bits_2d)
        else:
            avg_amplitude = float("nan")

        if not np.isnan(avg_amplitude) and avg_amplitude > 0:
            avg_power = 10.0 * np.log10(
                (avg_amplitude ** 2) / (0.001 * cfg.reference_impedance_ohm)
            )
            avg_power -= (ant_gain + cable_loss)
        else:
            avg_power = float("nan")

        iq_resampled = Inphase_res[peak_idx:end_idx] + 1j * Quadrature_res[peak_idx:end_idx]
        rel_time_s = peak_idx / fs_res
        timestamp = start_time_s + rel_time_s

        if iq_raw is not None and len(iq_raw) > 0:
            start_raw = int(round(rel_time_s * fs_raw))
            end_raw = start_raw + int(round(cfg.frame_duration_s * fs_raw))
            buf_raw = int(round(cfg.iq_slice_buffer_s * fs_raw))
            s_buf = max(0, start_raw - buf_raw)
            e_buf = min(len(iq_raw), end_raw + buf_raw)
            iq_highres = np.asarray(iq_raw[s_buf:e_buf])
        else:
            iq_highres = iq_resampled

        corr_triplet = (
            (float(ral[peak_idx - 1]), float(ral[peak_idx]), float(ral[peak_idx + 1]))
            if 0 < peak_idx < len(ral) - 1
            else None
        )
        cfo_raw_hz, phase_noise_std = calculate_freq_shift_and_phase_noise(iq_resampled, bits_112)
        timing_offset_ppm, sub_sample_lag_ns = estimate_timing_offset(
            mes_env, bits_112, corr_triplet=corr_triplet, sample_rate=fs_res
        )
        pa_droop_db, crest_factor, phase_curv, snr_db, pre_ratio_db = extract_waveform_transient_features(
            iq_resampled
        )

        valid_struct_list.append(
            {
                "Bits": bits_112,
                "Timestamp": float(timestamp),
                "IQ_resampled": iq_resampled,
                "IQ": iq_highres,
                "Power": float(avg_power),
                "Amplitude": float(avg_amplitude),
                "CFO_Raw_Hz": float(cfo_raw_hz),
                "Phase_Noise_Std": float(phase_noise_std),
                "Timing_Offset_PPM": float(timing_offset_ppm),
                "Sub_Sample_Lag_Ns": float(sub_sample_lag_ns),
                "PA_Droop_dB": float(pa_droop_db),
                "Pulse_Crest_Factor": float(crest_factor),
                "Phase_Curvature_Rad": float(phase_curv),
                "SNR_dB": float(snr_db),
                "Preamble_Ratio_dB": float(pre_ratio_db),
                "CRC_Valid": True,
            }
        )

    failed_struct_list: List[Dict[str, Any]] = []
    if capture_crc_failures:
        accepted_failed_peaks: List[int] = []
        for peak_idx, bits_112 in failed_candidates_raw:
            if any(abs(peak_idx - vp) < msg_length for vp in valid_peak_indices):
                continue
            if any(abs(peak_idx - fp) < msg_length for fp in accepted_failed_peaks):
                continue

            end_idx = peak_idx + msg_length
            mes_env = Z[peak_idx:end_idx]

            seg_head = mes_env[:len_template] - np.mean(mes_env[:len_template])
            seg_norm = float(np.linalg.norm(seg_head))
            if seg_norm < 1e-9:
                continue
            norm_xcorr = float(np.dot(seg_head, preamble_ext_17) / (seg_norm * template_norm))
            if norm_xcorr < DEFAULT_ASSUMPTIONS.blind_preamble_xcorr_min:
                continue

            iq_resampled = Inphase_res[peak_idx:end_idx] + 1j * Quadrature_res[peak_idx:end_idx]
            corr_triplet = (
                (float(ral[peak_idx - 1]), float(ral[peak_idx]), float(ral[peak_idx + 1]))
                if 0 < peak_idx < len(ral) - 1
                else None
            )
            blind_feats = extract_blind_rf_features(
                iq_resampled, mes_env, bits_112, corr_triplet=corr_triplet, sample_rate=fs_res
            )
            if (
                blind_feats["Clean_Slot_Ratio"] < DEFAULT_ASSUMPTIONS.blind_clean_slot_ratio_min
                or np.isnan(blind_feats["CFO_Raw_Hz"])
            ):
                continue

            accepted_failed_peaks.append(peak_idx)
            avg_amplitude = calculate_amplitude(mes_env, bits_112)
            if not np.isnan(avg_amplitude) and avg_amplitude > 0:
                avg_power = 10.0 * np.log10(
                    (avg_amplitude ** 2) / (0.001 * cfg.reference_impedance_ohm)
                ) - (ant_gain + cable_loss)
            else:
                avg_power = float("nan")

            rel_time_s = peak_idx / fs_res
            timestamp = start_time_s + rel_time_s
            raw_icao = int(np.dot(bits_112[8:32], pow_icao))
            raw_df = int(np.dot(bits_112[0:5], pow_df))
            raw_tc = int(np.dot(bits_112[32:37], pow_tc))

            failed_struct_list.append(
                {
                    "Bits": bits_112,
                    "Timestamp": float(timestamp),
                    "IQ_resampled": iq_resampled,
                    "IQ": iq_resampled,
                    "Power": float(avg_power),
                    "Amplitude": float(avg_amplitude),
                    "CFO_Raw_Hz": blind_feats["CFO_Raw_Hz"],
                    "Phase_Noise_Std": blind_feats["Phase_Noise_Std"],
                    "Timing_Offset_PPM": blind_feats["Timing_Offset_PPM"],
                    "Sub_Sample_Lag_Ns": blind_feats["Sub_Sample_Lag_Ns"],
                    "PA_Droop_dB": blind_feats["PA_Droop_dB"],
                    "Pulse_Crest_Factor": blind_feats["Pulse_Crest_Factor"],
                    "Phase_Curvature_Rad": blind_feats["Phase_Curvature_Rad"],
                    "SNR_dB": blind_feats["SNR_dB"],
                    "Preamble_Ratio_dB": blind_feats["Preamble_Ratio_dB"],
                    "Clean_Slot_Ratio": blind_feats["Clean_Slot_Ratio"],
                    "Corrupted_ICAO_Hex": f"{raw_icao:06X}",
                    "Corrupted_DF": raw_df,
                    "Corrupted_TC": raw_tc,
                    "CRC_Valid": False,
                }
            )

    if capture_crc_failures:
        return valid_struct_list, failed_struct_list
    return valid_struct_list
