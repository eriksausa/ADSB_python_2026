"""
Digital Signal Processing (DSP) and Physical-Layer RF Feature Extraction Module.
Performs Mode-S preamble cross-correlation, PPM demodulation, 24-bit CRC validation,
calibrated RF power/EIRP estimation, Carrier Frequency Offset (CFO) estimation,
Symbol Clock / Timing Offset extraction, and BLIND RF Fingerprinting for CRC-failed bursts.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks

from config import ReceiverConfig
from stream_reader import IQStreamChunk


def decode_ppm(envelope_frame: np.ndarray, sample_rate: float = 10e6) -> np.ndarray:
    """
    Decodes 112 Mode-S Pulse Position Modulation (PPM) payload bits from a
    resampled 120 us signal envelope (1,200 samples at 10 MHz).
    """
    samples_per_half_bit = int(sample_rate / (2 * 1e6))
    ppm_one = np.repeat(np.array([1.0, -1.0]), samples_per_half_bit)

    starting_point = int(8e-6 * sample_rate)  # Skip 8 us preamble (80 samples at 10 MHz)
    bit_length = int(sample_rate * 1e-6)      # 10 samples per 1 us bit

    payload_samples = envelope_frame[starting_point : starting_point + (112 * bit_length)]
    input_matrix = payload_samples.reshape(112, bit_length)

    correlated = np.dot(input_matrix, ppm_one)
    msg = (correlated > 0).astype(int)
    return msg.reshape(1, 112)


def is_valid(msg_bits: np.ndarray) -> bool:
    """
    Verifies 24-bit Mode-S CRC parity using modulo-2 polynomial division
    with generator polynomial G(x) = 0xFFF409.
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
    Constructs a 1200-sample active pulse mask WITHOUT trusting CRC-validated bits:
    1. Uses the fixed, known 8 us Mode-S preamble (`1010000101000000`) for samples 0..80.
    2. For each 1 us symbol slot in the payload, selects the stronger 0.5 us half-chip
       only if its energy contrast exceeds 2.2x the opposite half-chip (rejecting collided
       or noise-corrupted symbols automatically).
    Returns:
        - `full_mask_10mhz`: 1200-sample mask (1 = clean active pulse, 0 = noise/corrupted)
        - `confident_slot_mask`: 112-element boolean array indicating uncorrupted bit slots
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

    confident_slots = contrast_ratio > 2.2
    decision_bits = (first_half_energy >= second_half_energy).astype(int)

    ppm_pairs = np.column_stack((decision_bits, 1 - decision_bits))
    ppm_pairs[~confident_slots, :] = 0
    payload_mask_10mhz = np.repeat(ppm_pairs.flatten(), 5)

    return np.concatenate([preamble_10mhz, payload_mask_10mhz]), confident_slots


def calculate_amplitude(rx: np.ndarray, msg_bits: np.ndarray) -> float:
    """
    Estimates the true RF pulse amplitude by averaging peak signal pulse levels
    and subtracting the surrounding noise floor level.
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


def calculate_freq_shift_and_phase_noise(
    rx_complex: np.ndarray,
    msg_bits: np.ndarray,
    custom_mask_10mhz: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Estimates raw Carrier Frequency Offset (CFO in Hz) and unwrapped phase noise standard
    deviation (rad) from the 10 MHz complex IQ frame. Supports both CRC-verified masks and
    outlier-rejected blind energy masks for CRC-failed bursts.
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

    if len(rx_meanvalues) < 2:
        return float("nan"), float("nan")

    time_diffs = np.round(np.diff(t_meanvalues), 8)
    idx1_0 = np.where(time_diffs == 1.0e-6)[0]
    idx1_5 = np.where(time_diffs == 1.5e-6)[0]

    if len(idx1_0) == 0:
        return float("nan"), float("nan")

    phasedelta1_0 = rx_meanvalues[idx1_0 + 1] / rx_meanvalues[idx1_0]
    phasedelta1_5 = (
        rx_meanvalues[idx1_5 + 1] / rx_meanvalues[idx1_5]
        if len(idx1_5) > 0
        else np.array([])
    )

    phase_std = float(np.std(np.angle(phasedelta1_0)))
    threshold = 0.45
    if np.std(phasedelta1_0) < threshold:
        f_measured1_0 = np.angle(np.mean(phasedelta1_0)) / (2.0 * np.pi * 1.0e-6)
        if len(phasedelta1_5) > 0:
            f_measured1_5 = np.angle(np.mean(phasedelta1_5)) / (2.0 * np.pi * 1.5e-6)
            f_estimate_initial = (
                f_measured1_0 * len(idx1_0) + f_measured1_5 * len(idx1_5)
            ) / (len(idx1_0) + len(idx1_5))
        else:
            f_estimate_initial = f_measured1_0

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

    return float("nan"), phase_std


def calculate_freq_shift(rx_complex: np.ndarray, msg_bits: np.ndarray) -> float:
    """Backwards-compatible wrapper returning raw CFO in Hz."""
    cfo_hz, _ = calculate_freq_shift_and_phase_noise(rx_complex, msg_bits)
    return cfo_hz


def _sub_sample_corr_lag(sig_segment: np.ndarray, template_segment: np.ndarray) -> float:
    """
    Computes the sub-sample lag (in samples) between `sig_segment` and `template_segment`
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
    Extracts two physical-layer timing features for RF Fingerprinting:
    1. `timing_offset_ppm`: Symbol clock frequency skew (parts per million) across the 120 us
       frame by comparing the matched-filter sub-sample lag of the first half of the frame
       (0..60 us) vs. the second half (60..120 us) against the pulse mask.
    2. `sub_sample_lag_ns`: Sub-sample arrival time offset (in nanoseconds) via parabolic
       interpolation around the preamble correlation peak.
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


def extract_blind_rf_features(
    iq_resampled: np.ndarray,
    envelope_frame: np.ndarray,
    demod_bits: np.ndarray,
    corr_triplet: Optional[Tuple[float, float, float]] = None,
    sample_rate: float = 10e6,
) -> Dict[str, float]:
    """
    Extracts physical-layer RF fingerprint features (CFO, Symbol Clock Timing Offset,
    Phase Noise, and Uncorrupted Slot Ratio) from a CRC-FAILED burst by masking out
    low-contrast/collided bit slots so burst noise does not corrupt the oscillator estimate.
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
    return {
        "CFO_Raw_Hz": float(cfo_raw_hz),
        "Phase_Noise_Std": float(phase_noise_std),
        "Timing_Offset_PPM": float(timing_offset_ppm),
        "Sub_Sample_Lag_Ns": float(sub_sample_lag_ns),
        "Clean_Slot_Ratio": float(np.mean(confident_slots)),
    }


def lldistkm(latlon1: List[float], latlon2: List[float]) -> Tuple[float, float]:
    """
    Computes Haversine distance (d1km) and equirectangular Pythagorean distance (d2km) in km.
    """
    radius = 6371.0
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


def compute_eirp_dbm(rx_power_dbm: float, distance_m: float, carrier_freq_hz: float = 1090e6) -> float:
    """
    Calculates estimated Equivalent Isotropically Radiated Power (EIRP in dBm)
    by adding Free-Space Path Loss (FSPL) to the calibrated received power.
    """
    if np.isnan(rx_power_dbm) or np.isnan(distance_m) or distance_m <= 0:
        return float("nan")
    lambda_c = 299792458.0 / carrier_freq_hz
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
    position frame paired with the aircraft's current velocity vector.
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

    vel_m_s = airspeed_kt * 0.514444
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
    Detects Mode-S DF17 frames via cross-correlation, verifies CRC-24 parity,
    and extracts calibrated Rx power, high-res IQ slices, raw CFO, phase noise,
    and timing offset features for RF fingerprinting.

    When `capture_crc_failures=True`, also extracts blind physical-layer RF features
    from genuine Mode-S RF bursts that pass preamble correlation and SNR gating but fail
    the 24-bit CRC parity check, allowing the ML RF Fingerprinting classifier to identify
    which aircraft transmitted them.
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
    msg_length = int(fs_res * cfg.frame_duration_s)  # 1,200 samples at 10 MHz
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

    # Pass 1: Extract all CRC-valid Mode-S frames
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
                "CRC_Valid": True,
            }
        )

    # Pass 2: Filter & extract genuine Mode-S RF bursts that failed CRC-24 (rejecting thermal noise peaks)
    failed_struct_list: List[Dict[str, Any]] = []
    if capture_crc_failures:
        accepted_failed_peaks: List[int] = []
        for peak_idx, bits_112 in failed_candidates_raw:
            # Skip sidelobes overlapping within 120 us (1200 samples) of any CRC-valid frame
            if any(abs(peak_idx - vp) < msg_length for vp in valid_peak_indices):
                continue
            if any(abs(peak_idx - fp) < msg_length for fp in accepted_failed_peaks):
                continue

            end_idx = peak_idx + msg_length
            mes_env = Z[peak_idx:end_idx]

            # Normalized preamble correlation check (distinguishes real Mode-S preamble from noise)
            seg_head = mes_env[:len_template] - np.mean(mes_env[:len_template])
            seg_norm = float(np.linalg.norm(seg_head))
            if seg_norm < 1e-9:
                continue
            norm_xcorr = float(np.dot(seg_head, preamble_ext_17) / (seg_norm * template_norm))
            if norm_xcorr < 0.75:
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
                blind_feats["Clean_Slot_Ratio"] < 0.80
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