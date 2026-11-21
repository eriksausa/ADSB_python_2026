"""
Digital Signal Processing (DSP) and Physical-Layer RF Feature Extraction (`src/adsb_rf/dsp.py`).

Implements the physical-layer signal processing chain for 1090 MHz Mode-S Extended Squitter
(DF=17) telemetry sampled at 10.0 MS/s (10 samples/us; 5 samples per 0.5 us PPM half-chip):
  1. Mode-S Pulse Position Modulation (PPM) bit slicing and modulo-2 CRC-24 verification.
  2. Two-stage unwrap-free Carrier Frequency Offset (CFO) and phase-noise estimation.
  3. Sub-sample parabolic correlation lag, symbol clock drift (ppm), and PA transients.
  4. Spherical-geometry kinematic Doppler shift and Free-Space Path Loss (FSPL) EIRP.
  5. Streaming extended-preamble cross-correlation detector (`find_valid_xcorr`).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks

from .assumptions import DEFAULT_ASSUMPTIONS, ReceiverConfig
from .stream_reader import IQStreamChunk

# Fixed 8 us Mode-S preamble (16 chips @ 2 MHz -> repeated 5x for 80 samples @ 10 MS/s).
_PREAMBLE_CHIPS_16 = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=int)

# ICAO Annex 10 Vol. IV §3.1.2.3.3 CRC-24 generator polynomial G(x) = 0x1FFF409 (25 taps).
_CRC24_POLY = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    dtype=bool,
)


# =============================================================================
# 1. PPM DEMODULATION, ACTIVE-CHIP MASKS & CRC-24 PARITY CHECK
# =============================================================================

def decode_ppm(envelope_frame: np.ndarray, sample_rate: float = 10e6) -> np.ndarray:
    """
    Decodes 112 Mode-S PPM payload bits from a 120 us burst envelope (1,200 samples @ 10 MS/s).
    Each 1.0 us symbol is correlated against [+1 ... -1 ...] so bit=1 if first half > second half.
    """
    samples_per_half_bit = int(sample_rate / (2 * 1e6))
    ppm_one = np.repeat(np.array([1.0, -1.0]), samples_per_half_bit)

    starting_point = int(DEFAULT_ASSUMPTIONS.preamble_duration_s * sample_rate)
    bit_length = int(sample_rate * 1e-6)

    payload_samples = envelope_frame[starting_point : starting_point + (112 * bit_length)]
    # Reshape 1,120 payload samples -> (112 symbols, 10 samples/symbol) and correlate in one BLAS call.
    input_matrix = payload_samples.reshape(112, bit_length)
    correlated = np.dot(input_matrix, ppm_one)
    return (correlated > 0).astype(int).reshape(1, 112)


def is_valid(msg_bits: np.ndarray) -> bool:
    """
    Verifies 24-bit Mode-S CRC parity via modulo-2 polynomial division over GF(2)
    with G(x) = 0x1FFF409 (ICAO Annex 10 Vol. IV / RTCA DO-260B).
    """
    bits = np.squeeze(msg_bits).astype(bool).copy()
    if len(bits) < 112:
        return False

    for i in range(88):
        if bits[i]:
            bits[i : i + 25] = np.logical_xor(bits[i : i + 25], _CRC24_POLY)

    return not np.any(bits[88:112])


def smooth_3(x: np.ndarray) -> np.ndarray:
    """3-point boxcar moving average isolating flat pulse tops from 0.1 us rise/fall edges."""
    return np.convolve(x, np.ones(3) / 3.0, mode="same")


def build_ideal_10mhz_mask(msg_bits: np.ndarray) -> np.ndarray:
    """Reconstructs the 1,200-sample ideal binary OOK pulse mask for preamble + 112 PPM bits."""
    msg_flat = np.squeeze(msg_bits).astype(int)
    preamble_10mhz = np.repeat(_PREAMBLE_CHIPS_16, 5)
    ppm_pairs = np.column_stack((msg_flat, 1 - msg_flat)).flatten()
    return np.concatenate([preamble_10mhz, np.repeat(ppm_pairs, 5)])


def build_blind_energy_mask_10mhz(envelope_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a 1,200-sample active-chip mask without trusting CRC-validated bits:
      - Samples 0..80 use the fixed 8 us Mode-S preamble pattern.
      - Each 1 us payload symbol keeps only the stronger 0.5 us half-chip when its energy
        contrast exceeds `blind_ppm_contrast_ratio` (2.2x), gating out collided/faded bits.
    """
    preamble_10mhz = np.repeat(_PREAMBLE_CHIPS_16, 5)
    env = np.abs(envelope_frame[:1200])
    payload_slots = env[80:1200].reshape(112, 10)
    first_half = payload_slots[:, :5].sum(axis=1)
    second_half = payload_slots[:, 5:].sum(axis=1)

    contrast_ratio = np.maximum(first_half, second_half) / np.maximum(np.minimum(first_half, second_half), 1e-12)
    confident_slots = contrast_ratio > DEFAULT_ASSUMPTIONS.blind_ppm_contrast_ratio
    decision_bits = (first_half >= second_half).astype(int)

    ppm_pairs = np.column_stack((decision_bits, 1 - decision_bits))
    ppm_pairs[~confident_slots, :] = 0
    return np.concatenate([preamble_10mhz, np.repeat(ppm_pairs.flatten(), 5)]), confident_slots


def calculate_amplitude(rx: np.ndarray, msg_bits: np.ndarray) -> float:
    """
    Estimates pulse envelope amplitude by averaging interior flat-top samples
    (`smooth_3 >= 0.9`) and subtracting the off-pulse thermal noise baseline.
    """
    bits_modulated = build_ideal_10mhz_mask(msg_bits)
    bits_3 = smooth_3(bits_modulated.astype(float))
    bits_3_neg = smooth_3((1 - bits_modulated).astype(float))

    bits_3[bits_3 < 0.9] = 0.0
    bits_3_neg[bits_3_neg < 0.9] = 0.0
    bits_3_neg[0] = 0.0
    bits_3[0] = 0.0
    if len(bits_3) > 0 and bits_3[-1] == 1.0:
        bits_3[-1] = 0.0

    rx_abs = np.abs(rx[: len(bits_modulated)])
    noise_blocks = (rx_abs * bits_3_neg).reshape(-1, 10).sum(axis=1) / 3.0
    signal_blocks = (rx_abs * bits_3).reshape(-1, 10).sum(axis=1) / 3.0

    noise = float(np.mean(noise_blocks[noise_blocks != 0])) if np.any(noise_blocks != 0) else 0.0
    sig = float(np.mean(signal_blocks[signal_blocks != 0])) if np.any(signal_blocks != 0) else 0.0
    return float(max(sig - noise, 0.0))


def _amplitude_to_rx_power_dbm(
    avg_amplitude: float,
    ant_gain_db: float,
    cable_loss_db: float,
    z0_ohm: float,
) -> float:
    """Converts linear envelope amplitude (V) to calibrated antenna-port power in dBm."""
    if np.isnan(avg_amplitude) or avg_amplitude <= 0.0:
        return float("nan")
    pwr_dbm = 10.0 * np.log10((avg_amplitude ** 2) / (0.001 * z0_ohm))
    return float(pwr_dbm - (ant_gain_db + cable_loss_db))


# =============================================================================
# 2. TWO-STAGE UNWRAP-FREE CARRIER FREQUENCY OFFSET (CFO) & PHASE NOISE
# =============================================================================

def _integrate_active_slots_10mhz(
    rx_10mhz: np.ndarray,
    mask_10mhz: np.ndarray,
    t_10mhz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherently integrates the active 5-sample (0.5 us) half-chip within each 1.0 us symbol
    slot (`n_slots <= 120`), returning `(active_phasors, active_slot_timestamps)`.
    """
    n_slots = min(len(rx_10mhz), len(mask_10mhz), len(t_10mhz)) // 10
    if n_slots == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.float64)

    n_trim = n_slots * 10
    m = mask_10mhz[:n_trim]
    rx_slots = (rx_10mhz[:n_trim] * m).reshape(n_slots, 10).sum(axis=1) / 5.0
    t_slots = (t_10mhz[:n_trim] * m).reshape(n_slots, 10).sum(axis=1) / 5.0

    valid = np.abs(rx_slots) > 1e-12
    return rx_slots[valid], t_slots[valid]


def recover_cfo_unwrap_free(
    iq_res: Optional[np.ndarray],
    bits: Optional[np.ndarray] = None,
    fs_hz: float = 10e6,
    custom_mask_10mhz: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Two-stage unwrap-free Carrier Frequency Offset (CFO, Hz) & phase-noise (rad) estimator:
      - Stage A (Coarse Intra-Pulse Delay-and-Multiply, unambiguous range = +/- 5.0 MHz):
            f_coarse = arg(sum r[n+1] * conj(r[n])) / (2 * pi * T_s)   (m[n] == m[n+1] == 1)
      - Stage B (Derotated Fine 1.0 us & 1.5 us Unit-Phasor Refinement):
        Removes f_coarse so residual frequency sits near 0 Hz (far from +/- pi branch cuts),
        then weights 1.0 us and 1.5 us active-slot conjugate products for sub-kHz precision.
    """
    if iq_res is None or len(iq_res) < 400:
        return 0.0, 0.5

    iq = np.asarray(iq_res[:1200], dtype=np.complex128)
    n_len = len(iq)

    if custom_mask_10mhz is not None and len(custom_mask_10mhz) >= n_len:
        mask_10mhz = custom_mask_10mhz[:n_len].astype(bool)
    elif bits is not None and len(np.squeeze(bits)) >= 112:
        mask_10mhz = build_ideal_10mhz_mask(bits)[:n_len].astype(bool)
    else:
        env = np.abs(iq)
        mask_10mhz = env >= np.percentile(env, DEFAULT_ASSUMPTIONS.active_pulse_percentile)

    same_pulse = mask_10mhz[:-1] & mask_10mhz[1:]
    if np.sum(same_pulse) < 10:
        return 0.0, 0.5

    # Stage A: adjacent sample lag dt = 0.1 us inside active 0.5 us pulses (+/- 5 MHz Nyquist).
    conj_1samp = iq[1:][same_pulse] * np.conj(iq[:-1][same_pulse])
    f_coarse_hz = float(np.angle(np.sum(conj_1samp)) / (2.0 * np.pi * (1.0 / fs_hz)))

    # Stage B: derotate entire burst and refine across 1.0 us and 1.5 us symbol gaps.
    t_vec = np.arange(n_len, dtype=np.float64) / fs_hz
    iq_derot = iq * np.exp(-1j * 2.0 * np.pi * f_coarse_hz * t_vec)
    rx_vals, t_vals = _integrate_active_slots_10mhz(iq_derot, mask_10mhz, t_vec)

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
    Estimates raw CFO (Hz) and unwrapped phase noise std (rad) from a 1,200-sample burst.
    Uses fast linear phase regression when inter-slot phase spread < 0.45 rad and within the
    +/- 300 kHz aliasing margin; automatically falls back to `recover_cfo_unwrap_free` otherwise.
    """
    mask_10mhz = (
        custom_mask_10mhz[:1200]
        if custom_mask_10mhz is not None
        else build_ideal_10mhz_mask(msg_bits)
    )
    t_10mhz = np.arange(1200, dtype=np.float64) * 0.1e-6 + 0.05e-6
    rx_slice = rx_complex[:1200]

    rx_vals, t_vals = _integrate_active_slots_10mhz(rx_slice, mask_10mhz, t_10mhz)
    if len(rx_vals) >= 2:
        time_diffs = np.round(np.diff(t_vals), 8)
        idx1_0 = np.where(time_diffs == 1.0e-6)[0]
        idx1_5 = np.where(time_diffs == 1.5e-6)[0]

        if len(idx1_0) > 0:
            phasedelta1_0 = rx_vals[idx1_0 + 1] / rx_vals[idx1_0]
            phasedelta1_5 = (
                rx_vals[idx1_5 + 1] / rx_vals[idx1_5] if len(idx1_5) > 0 else np.array([])
            )

            if np.std(phasedelta1_0) < DEFAULT_ASSUMPTIONS.cfo_phase_delta_std_gate_rad:
                f_1_0 = np.angle(np.mean(phasedelta1_0)) / (2.0 * np.pi * 1.0e-6)
                if len(phasedelta1_5) > 0:
                    f_1_5 = np.angle(np.mean(phasedelta1_5)) / (2.0 * np.pi * 1.5e-6)
                    f_init = (f_1_0 * len(idx1_0) + f_1_5 * len(idx1_5)) / (len(idx1_0) + len(idx1_5))
                else:
                    f_init = f_1_0

                # Guard against 2*pi aliasing on 1.0 us (+/-500 kHz) and 1.5 us (+/-333.3 kHz) gaps.
                same_pulse = (mask_10mhz[:-1] == 1) & (mask_10mhz[1:] == 1)
                f_coarse_check = (
                    float(
                        np.angle(np.sum(rx_slice[1:][same_pulse] * np.conj(rx_slice[:-1][same_pulse])))
                        / (2.0 * np.pi * 0.1e-6)
                    )
                    if np.sum(same_pulse) >= 10
                    else float(f_init)
                )

                if abs(f_init - f_coarse_check) < 150e3 and abs(f_coarse_check) < 300e3:
                    rx_fcorr = rx_slice * np.exp(-1j * 2.0 * np.pi * f_init * t_10mhz)
                    active_mask = mask_10mhz == 1
                    unwrapped = np.unwrap(np.angle(rx_fcorr[active_mask]))
                    t_active = t_10mhz[active_mask]

                    slope, intercept = np.polyfit(t_active, unwrapped, deg=1)
                    phase_noise_std = float(np.std(unwrapped - (slope * t_active + intercept)))
                    return float(f_init + slope / (2.0 * np.pi)), phase_noise_std

    return recover_cfo_unwrap_free(rx_slice, msg_bits, fs_hz=10e6, custom_mask_10mhz=mask_10mhz)


# =============================================================================
# 3. PARABOLIC SYMBOL CLOCK DRIFT & POWER-AMPLIFIER TRANSIENT FEATURES
# =============================================================================

def _sub_sample_corr_lag(sig_segment: np.ndarray, template_segment: np.ndarray) -> float:
    """
    Parabolic sub-sample peak lag interpolation (in samples) via zero-mean cross-correlation
    evaluated at integer lags [-1, 0, +1]:
        delta = (R[-1] - R[+1]) / (2 * (R[-1] - 2*R[0] + R[+1]))
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
    Computes:
      1. `timing_offset_ppm`: 2 MHz symbol-clock crystal skew measured as lag drift between
         the first 60 us (samples 0..600) and second 60 us (samples 600..1200) of the burst.
      2. `sub_sample_lag_ns`: Sub-sample preamble arrival offset (ns) around the peak.
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

    lag_first = _sub_sample_corr_lag(env[:600], ideal_mask[:600])
    lag_second = _sub_sample_corr_lag(env[600:1200], ideal_mask[600:1200])
    timing_offset_ppm = float(-((lag_second - lag_first) / 600.0) * 1e6)
    return timing_offset_ppm, sub_sample_lag_ns


def extract_waveform_transient_features(
    iq_res: Optional[np.ndarray],
) -> Tuple[float, float, float, float, float]:
    """
    Extracts 5 hardware-specific RF transmitter transient features from `iq_res` (1,200 samples):
      1. `pa_droop_db`: Power Amplifier storage-capacitor voltage sag (first 30 us vs last 30 us, dB).
      2. `pulse_crest_factor`: 95th-percentile peak to median active pulse top ratio (PA overshoot).
      3. `phase_curvature_rad`: Half-burst frequency chirp from thermal PA pulling (rad).
      4. `snr_db`: In-frame active pulse median vs 35th-percentile noise floor ratio (dB).
      5. `preamble_ratio_db`: Fixed preamble pulse amplitude relative to payload median (dB).
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
    active_vals = env[env >= q55]
    noise_vals = env[env <= q35]

    mean_sig = float(np.median(active_vals)) if len(active_vals) > 0 else max_env
    mean_noise = float(np.median(noise_vals)) if len(noise_vals) > 0 else (mean_sig * 0.05)
    snr_db = float(np.clip(20.0 * np.log10(max(mean_sig, 1e-12) / max(mean_noise, 1e-12)), 0.0, 60.0))

    # Power amplifier supply capacitor sag across the 120 us transmit pulse train.
    head_env, tail_env = env[:300], env[-300:]
    head_act = head_env[head_env >= np.percentile(head_env, 60)]
    tail_act = tail_env[tail_env >= np.percentile(tail_env, 60)]
    head_amp = float(np.median(head_act)) if len(head_act) > 0 else mean_sig
    tail_amp = float(np.median(tail_act)) if len(tail_act) > 0 else mean_sig
    pa_droop_db = float(np.clip(20.0 * np.log10(max(head_amp, 1e-12) / max(tail_amp, 1e-12)), -6.0, 6.0))

    p95 = float(np.percentile(active_vals, 95)) if len(active_vals) > 0 else mean_sig
    pulse_crest_factor = float(np.clip(p95 / max(mean_sig, 1e-12), 1.0, 3.5))

    if len(env) >= 80:
        pre_pulses = np.concatenate([env[0:5], env[10:15], env[35:40], env[45:50]])
        pre_amp = float(np.median(pre_pulses))
        preamble_ratio_db = float(np.clip(20.0 * np.log10(max(pre_amp, 1e-12) / max(mean_sig, 1e-12)), -6.0, 6.0))
    else:
        preamble_ratio_db = 0.0

    # Half-burst intra-pulse frequency step capturing VCO/PA thermal frequency pulling.
    half = len(iq) // 2
    iq1, iq2 = iq[:half], iq[half:]
    act1, act2 = np.abs(iq1[:-1]) >= q55, np.abs(iq2[:-1]) >= q55
    if np.sum(act1) >= 10 and np.sum(act2) >= 10:
        dphi1 = np.angle(iq1[1:][act1] * np.conj(iq1[:-1][act1]))
        dphi2 = np.angle(iq2[1:][act2] * np.conj(iq2[:-1][act2]))
        phase_curvature_rad = float(np.clip((np.median(dphi2) - np.median(dphi1)) * 10.0, -2.0, 2.0))
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
    Extracts physical-layer RF fingerprint features from a CRC-failed burst using the
    high-contrast blind energy mask (excluding collided/low-SNR bit slots).
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


# =============================================================================
# 4. GREAT-CIRCLE GEODESY, FREE-SPACE PATH LOSS (EIRP) & SPHERICAL DOPPLER
# =============================================================================

def lldistkm(latlon1: List[float], latlon2: List[float]) -> Tuple[float, float]:
    """
    Computes Haversine (`d1km`) and equirectangular (`d2km`) distance in km using the
    IUGG / WGS-84 mean spherical Earth radius (`6371.0088 km`).
    """
    radius = DEFAULT_ASSUMPTIONS.earth_mean_radius_km
    lat1, lon1 = np.radians(latlon1[0]), np.radians(latlon1[1])
    lat2, lon2 = np.radians(latlon2[0]), np.radians(latlon2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    d1km = radius * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))
    d2km = radius * np.sqrt((dlon * np.cos((lat1 + lat2) / 2.0)) ** 2 + dlat ** 2)
    return float(d1km), float(d2km)


def compute_eirp_dbm(
    rx_power_dbm: float,
    distance_m: float,
    carrier_freq_hz: float = DEFAULT_ASSUMPTIONS.carrier_freq_hz,
) -> float:
    """
    Computes Equivalent Isotropically Radiated Power (EIRP in dBm) via Free-Space Path Loss:
        FSPL(dB) = 20 * log10(4 * pi * d / lambda_c)
    """
    if np.isnan(rx_power_dbm) or np.isnan(distance_m) or distance_m <= 0:
        return float("nan")
    lambda_c = DEFAULT_ASSUMPTIONS.speed_of_light_mps / carrier_freq_hz
    return float(rx_power_dbm + 20.0 * np.log10((4.0 * np.pi * distance_m) / lambda_c))


def compute_single_frame_doppler(
    lat: float,
    lon: float,
    alt_ft: float,
    airspeed_kt: float,
    heading_deg: float,
    config: ReceiverConfig,
) -> float:
    """
    Computes expected radial Doppler shift (Hz) at 1090 MHz from the spherical bearing
    between aircraft `(lat, lon, airspeed_kt, heading_deg)` and receiver `(rx_lat, rx_lon)`:
        f_D = -(v / c) * f_c * cos(bearing_ant_to_ac - heading)
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
    ant_to_plane = np.degrees(np.arctan2(y, x)) + 180.0
    if ant_to_plane < 0:
        ant_to_plane += 360.0

    return float(
        -np.cos(np.radians(ant_to_plane - heading_deg))
        * (vel_m_s / config.speed_of_light_mps)
        * config.carrier_freq_hz
    )


# =============================================================================
# 5. EXTENDED PREAMBLE MATCHED FILTER & STREAMING FRAME DETECTOR
# =============================================================================

def _extract_valid_frame_rf_features(
    iq_resampled: np.ndarray,
    mes_env: np.ndarray,
    bits_112: np.ndarray,
    corr_triplet: Optional[Tuple[float, float, float]],
    fs_res: float,
) -> Dict[str, float]:
    """Extracts CFO, phase noise, clock drift, lag, and 5 PA transient features for one frame."""
    cfo_raw_hz, phase_noise_std = calculate_freq_shift_and_phase_noise(iq_resampled, bits_112)
    timing_ppm, sub_lag_ns = estimate_timing_offset(
        mes_env, bits_112, corr_triplet=corr_triplet, sample_rate=fs_res
    )
    pa_droop, crest, curv, snr_db, pre_ratio = extract_waveform_transient_features(iq_resampled)
    return {
        "CFO_Raw_Hz": float(cfo_raw_hz),
        "Phase_Noise_Std": float(phase_noise_std),
        "Timing_Offset_PPM": float(timing_ppm),
        "Sub_Sample_Lag_Ns": float(sub_lag_ns),
        "PA_Droop_dB": float(pa_droop),
        "Pulse_Crest_Factor": float(crest),
        "Phase_Curvature_Rad": float(curv),
        "SNR_dB": float(snr_db),
        "Preamble_Ratio_dB": float(pre_ratio),
    }


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
    Detects Mode-S DF=17 frames via extended preamble matched filtering (16 preamble chips +
    10 DF=17 header chips), verifies CRC-24 parity, and extracts all physical-layer RF features.
    """
    cfg = config or ReceiverConfig()

    if isinstance(chunk, IQStreamChunk):
        Z, Inphase_res, Quadrature_res = chunk.envelope_res, chunk.inphase_res, chunk.quadrature_res
        iq_raw = chunk.iq_raw
        start_time_s = chunk.start_time_s if start_time is None else start_time
        fs_res = chunk.target_sample_rate_hz if sample_rate is None else sample_rate
        fs_raw = chunk.raw_sample_rate_hz
    else:
        Z, Inphase_res, Quadrature_res = chunk["Z"], chunk["Inphase"], chunk["Quadrature"]
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

    # Bipolar matched-filter template: 8 us preamble + DF=17 header (10001 -> 5 PPM symbols).
    preamble_bipolar = np.where(_PREAMBLE_CHIPS_16 == 1, 1, -1)
    ppm_df_17 = np.array([1, -1, -1, 1, -1, 1, -1, 1, 1, -1])
    samples_per_chip = int(1e-6 * fs_res / 2)
    template = np.repeat(np.concatenate([preamble_bipolar, ppm_df_17]), samples_per_chip)
    template_norm = float(np.linalg.norm(template))
    len_template = len(template)

    ral = np.correlate(Z, template, mode="valid")
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
    pow_5 = 2 ** np.arange(4, -1, -1)

    for peak_idx in valid_peaks:
        end_idx = peak_idx + msg_length
        mes_env = Z[peak_idx:end_idx]
        mes_bits_2d = decode_ppm(mes_env, fs_res)
        bits_112 = mes_bits_2d[0, :112] if mes_bits_2d.ndim == 2 else mes_bits_2d[:112]

        if not is_valid(bits_112):
            if capture_crc_failures:
                failed_candidates_raw.append((int(peak_idx), bits_112))
            continue

        valid_peak_indices.append(int(peak_idx))
        avg_amp = calculate_amplitude(mes_env, mes_bits_2d) if np.isclose(fs_res, 10e6) else float("nan")
        avg_pwr = _amplitude_to_rx_power_dbm(avg_amp, ant_gain, cable_loss, cfg.reference_impedance_ohm)

        iq_res = Inphase_res[peak_idx:end_idx] + 1j * Quadrature_res[peak_idx:end_idx]
        rel_time_s = peak_idx / fs_res
        if iq_raw is not None and len(iq_raw) > 0:
            s_raw = int(round(rel_time_s * fs_raw))
            buf_raw = int(round(cfg.iq_slice_buffer_s * fs_raw))
            iq_highres = np.asarray(
                iq_raw[max(0, s_raw - buf_raw) : min(len(iq_raw), s_raw + int(round(cfg.frame_duration_s * fs_raw)) + buf_raw)]
            )
        else:
            iq_highres = iq_res

        corr_triplet = (
            (float(ral[peak_idx - 1]), float(ral[peak_idx]), float(ral[peak_idx + 1]))
            if 0 < peak_idx < len(ral) - 1
            else None
        )
        rf_feats = _extract_valid_frame_rf_features(iq_res, mes_env, bits_112, corr_triplet, fs_res)
        valid_struct_list.append(
            {
                "Bits": bits_112,
                "Timestamp": float(start_time_s + rel_time_s),
                "IQ_resampled": iq_res,
                "IQ": iq_highres,
                "Power": avg_pwr,
                "Amplitude": float(avg_amp),
                "CRC_Valid": True,
                **rf_feats,
            }
        )

    failed_struct_list: List[Dict[str, Any]] = []
    if capture_crc_failures:
        accepted_failed_peaks: List[int] = []
        for peak_idx, bits_112 in failed_candidates_raw:
            if any(abs(peak_idx - p) < msg_length for p in valid_peak_indices + accepted_failed_peaks):
                continue

            end_idx = peak_idx + msg_length
            mes_env = Z[peak_idx:end_idx]
            seg_head = mes_env[:len_template] - np.mean(mes_env[:len_template])
            seg_norm = float(np.linalg.norm(seg_head))
            if seg_norm < 1e-9 or (np.dot(seg_head, template) / (seg_norm * template_norm)) < DEFAULT_ASSUMPTIONS.blind_preamble_xcorr_min:
                continue

            iq_res = Inphase_res[peak_idx:end_idx] + 1j * Quadrature_res[peak_idx:end_idx]
            corr_triplet = (
                (float(ral[peak_idx - 1]), float(ral[peak_idx]), float(ral[peak_idx + 1]))
                if 0 < peak_idx < len(ral) - 1
                else None
            )
            blind_feats = extract_blind_rf_features(
                iq_res, mes_env, bits_112, corr_triplet=corr_triplet, sample_rate=fs_res
            )
            if blind_feats["Clean_Slot_Ratio"] < DEFAULT_ASSUMPTIONS.blind_clean_slot_ratio_min or np.isnan(blind_feats["CFO_Raw_Hz"]):
                continue

            accepted_failed_peaks.append(peak_idx)
            avg_amp = calculate_amplitude(mes_env, bits_112)
            failed_struct_list.append(
                {
                    "Bits": bits_112,
                    "Timestamp": float(start_time_s + peak_idx / fs_res),
                    "IQ_resampled": iq_res,
                    "IQ": iq_res,
                    "Power": _amplitude_to_rx_power_dbm(avg_amp, ant_gain, cable_loss, cfg.reference_impedance_ohm),
                    "Amplitude": float(avg_amp),
                    "Corrupted_ICAO_Hex": f"{int(np.dot(bits_112[8:32], pow_icao)):06X}",
                    "Corrupted_DF": int(np.dot(bits_112[0:5], pow_5)),
                    "Corrupted_TC": int(np.dot(bits_112[32:37], pow_5)),
                    "CRC_Valid": False,
                    **blind_feats,
                }
            )

    return (valid_struct_list, failed_struct_list) if capture_crc_failures else valid_struct_list
