"""
Synthetic Mode-S DF17 Complex I/Q Stream Generator (`src/adsb_rf/synthesizer.py`).
Provides a deterministic, calibrated multi-aircraft 12.8 MS/s Mode-S stream when raw
multi-gigabyte VSA captures (`receivedSignals.npy`, `101000.mat`) are excluded by `.gitignore`.
"""

from typing import Tuple
import numpy as np

from .assumptions import DEFAULT_ASSUMPTIONS, ReceiverConfig
from .decoder import NL


def _compute_crc24(bits_88: np.ndarray) -> np.ndarray:
    """Computes the 24-bit Mode-S parity field for an 88-bit DF17 payload."""
    poly = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        dtype=int,
    )
    reg = np.concatenate([bits_88.astype(int), np.zeros(24, dtype=int)])
    for i in range(88):
        if reg[i] == 1:
            reg[i : i + 25] ^= poly
    return reg[88:112]


def _int_to_bits(val: int, width: int) -> np.ndarray:
    return np.array([(val >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)


def _encode_cpr_local(lat: float, lon: float, even_odd: int) -> Tuple[int, int]:
    """Encodes latitude and longitude into 17-bit local CPR integers."""
    dlat = 360.0 / (60.0 - even_odd)
    yz = int(np.floor((2 ** 17) * ((lat % dlat) / dlat) + 0.5)) % (2 ** 17)
    nl = NL(lat)
    dlon = 360.0 if (nl - even_odd) <= 0 else 360.0 / (nl - even_odd)
    xz = int(np.floor((2 ** 17) * ((lon % dlon) / dlon) + 0.5)) % (2 ** 17)
    return yz, xz


def _build_df17_callsign_bits(icao: int, callsign: str, ec: int = 3) -> np.ndarray:
    """Constructs a complete 112-bit CRC-valid DF17 Aircraft Identification frame (TC=4)."""
    df_ca = _int_to_bits((17 << 3) | 5, 8)
    icao_bits = _int_to_bits(icao, 24)
    tc_bits = _int_to_bits(4, 5)
    ec_bits = _int_to_bits(ec, 3)

    padded = callsign.upper().ljust(8)[:8]
    char_bits_list = []
    for ch in padded:
        if "A" <= ch <= "Z":
            code = ord(ch) - 64
        elif "0" <= ch <= "9":
            code = ord(ch)
        else:
            code = 32
        char_bits_list.append(_int_to_bits(code, 6))

    payload_56 = np.concatenate([tc_bits, ec_bits] + char_bits_list)
    bits_88 = np.concatenate([df_ca, icao_bits, payload_56])
    crc_24 = _compute_crc24(bits_88)
    return np.concatenate([bits_88, crc_24])


def _build_df17_position_bits(icao: int, alt_ft: float, lat: float, lon: float, even_odd: int) -> np.ndarray:
    """Constructs a complete 112-bit CRC-valid DF17 Airborne Position frame (TC=11)."""
    df_ca = _int_to_bits((17 << 3) | 5, 8)
    icao_bits = _int_to_bits(icao, 24)
    tc_ss_nic = _int_to_bits((11 << 3) | 0, 8)

    n_alt = int(round((alt_ft + 1000.0) / 25.0))
    alt_11 = _int_to_bits(max(0, min(n_alt, 2047)), 11)
    alt_12 = np.concatenate([alt_11[:7], np.array([1]), alt_11[7:]])

    t_cpr = np.array([0, int(even_odd)], dtype=int)
    lat_cpr, lon_cpr = _encode_cpr_local(lat, lon, even_odd)
    pos_34 = np.concatenate([_int_to_bits(lat_cpr, 17), _int_to_bits(lon_cpr, 17)])

    bits_88 = np.concatenate([df_ca, icao_bits, tc_ss_nic, alt_12, t_cpr, pos_34])
    crc_24 = _compute_crc24(bits_88)
    return np.concatenate([bits_88, crc_24])


def _build_df17_velocity_bits(icao: int, v_ew_kt: int, v_ns_kt: int, v_rate_fpm: int = 0) -> np.ndarray:
    """Constructs a complete 112-bit CRC-valid DF17 Airborne Velocity Subtype 1 frame (TC=19)."""
    df_ca = _int_to_bits((17 << 3) | 5, 8)
    icao_bits = _int_to_bits(icao, 24)
    tc_bits = _int_to_bits(19, 5)
    sub_bits = _int_to_bits(1, 3)
    ic_res_nac = _int_to_bits(0, 5)

    ew_sign = np.array([1 if v_ew_kt < 0 else 0], dtype=int)
    ew_mag = _int_to_bits(abs(v_ew_kt) + 1, 10)
    ns_sign = np.array([1 if v_ns_kt < 0 else 0], dtype=int)
    ns_mag = _int_to_bits(abs(v_ns_kt) + 1, 10)

    vr_src = np.array([0], dtype=int)
    vr_sign = np.array([1 if v_rate_fpm < 0 else 0], dtype=int)
    vr_mag = _int_to_bits(int(round(abs(v_rate_fpm) / 64.0)) + 1, 9)
    pad = np.zeros(10, dtype=int)

    payload_56 = np.concatenate(
        [tc_bits, sub_bits, ic_res_nac, ew_sign, ew_mag, ns_sign, ns_mag, vr_src, vr_sign, vr_mag, pad]
    )
    bits_88 = np.concatenate([df_ca, icao_bits, payload_56])
    crc_24 = _compute_crc24(bits_88)
    return np.concatenate([bits_88, crc_24])


def synthesize_portfolio_iq_stream(
    config: ReceiverConfig,
    raw_fs: float = DEFAULT_ASSUMPTIONS.default_raw_sample_rate_hz,
) -> np.ndarray:
    """
    Synthesizes a continuous complex IQ measurement at `raw_fs` (12.8 MS/s) containing:
    - 24 legitimate CRC-valid frames from 3 aircraft (`4401A2`, `3C6541`, `4B1809`),
    - 3 CRC-valid security/data-validity anomalies (RF Spoofing, Teleportation Jump, Extreme CFO/EIRP),
    - 3 CRC-failed bursts for blind RF attribution testing.
    """
    rng = np.random.default_rng(42)
    duration_s = 0.058
    total_samples = int(round(duration_s * raw_fs))
    iq_stream = (rng.normal(0, 3.0e-6, total_samples) + 1j * rng.normal(0, 3.0e-6, total_samples)).astype(
        np.complex128
    )

    aircraft_specs = [
        {
            "icao": 0x4401A2,
            "callsign": "AUA101",
            "cfo_hz": +14200.0,
            "timing_ppm": +65.0,
            "amp": 0.00095,
            "lat0": 48.2450,
            "lon0": 16.4800,
            "alt0": 31000.0,
            "v_ew": -220,
            "v_ns": -140,
        },
        {
            "icao": 0x3C6541,
            "callsign": "DLH402",
            "cfo_hz": -18500.0,
            "timing_ppm": -70.0,
            "amp": 0.00072,
            "lat0": 48.1200,
            "lon0": 16.2500,
            "alt0": 36000.0,
            "v_ew": 250,
            "v_ns": 110,
        },
        {
            "icao": 0x4B1809,
            "callsign": "SWR219",
            "cfo_hz": +3800.0,
            "timing_ppm": -10.0,
            "amp": 0.00125,
            "lat0": 48.2100,
            "lon0": 16.3100,
            "alt0": 24000.0,
            "v_ew": 180,
            "v_ns": -160,
        },
    ]

    events = []
    t_cursor = 0.0012
    dt_step = 0.0016

    for ac in aircraft_specs:
        cs_bits = _build_df17_callsign_bits(ac["icao"], ac["callsign"])
        events.append((t_cursor, cs_bits, ac["cfo_hz"], ac["timing_ppm"], ac["amp"]))
        t_cursor += dt_step

        vel_bits = _build_df17_velocity_bits(ac["icao"], ac["v_ew"], ac["v_ns"], 0)
        events.append((t_cursor, vel_bits, ac["cfo_hz"], ac["timing_ppm"], ac["amp"]))
        t_cursor += dt_step

        dlat_step = (ac["v_ns"] * DEFAULT_ASSUMPTIONS.knots_to_mps * dt_step) / 111320.0
        dlon_step = (ac["v_ew"] * DEFAULT_ASSUMPTIONS.knots_to_mps * dt_step) / (
            111320.0 * np.cos(np.radians(ac["lat0"]))
        )

        for k in range(6):
            eo = k % 2
            lat_k = ac["lat0"] + k * dlat_step
            lon_k = ac["lon0"] + k * dlon_step
            pos_bits = _build_df17_position_bits(ac["icao"], ac["alt0"], lat_k, lon_k, eo)
            cfo_jitter = float(rng.normal(0.0, 140.0))
            ppm_jitter = float(rng.normal(0.0, 2.0))
            events.append(
                (t_cursor, pos_bits, ac["cfo_hz"] + cfo_jitter, ac["timing_ppm"] + ppm_jitter, ac["amp"])
            )
            t_cursor += dt_step

    spoof_bits = _build_df17_position_bits(0x4401A2, 31000.0, 48.2450, 16.4800, 0)
    events.append((t_cursor, spoof_bits, -28500.0, -95.0, 0.00095))
    t_cursor += 0.0018

    jump_bits = _build_df17_position_bits(0x3C6541, 36000.0, 48.5200, 16.6500, 1)
    events.append((t_cursor, jump_bits, -18400.0, -69.0, 0.00072))
    t_cursor += 0.0018

    bad_alt_bits = _build_df17_position_bits(0x4B1809, 50000.0, 48.2100, 16.3100, 0)
    events.append((t_cursor, bad_alt_bits, +64000.0, -11.0, 0.0095))
    t_cursor += 0.0018

    for idx, ac in enumerate(aircraft_specs):
        corrupted_bits = _build_df17_position_bits(ac["icao"], ac["alt0"], ac["lat0"], ac["lon0"], idx % 2).copy()
        corrupted_bits[12] ^= 1
        corrupted_bits[19] ^= 1
        corrupted_bits[55] ^= 1
        corrupted_bits[72] ^= 1
        corrupted_bits[94] ^= 1
        events.append((t_cursor, corrupted_bits, ac["cfo_hz"] + 90.0, ac["timing_ppm"] - 1.5, ac["amp"]))
        t_cursor += 0.0018

    preamble_chips = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    for t_start, bits_112, cfo_hz, timing_ppm, amp in events:
        ppm_chips = np.column_stack((bits_112, 1 - bits_112)).flatten().astype(float)
        all_chips = np.concatenate([preamble_chips, ppm_chips])

        start_idx = int(round(t_start * raw_fs))
        n_frame_samples = int(round(120e-6 * raw_fs))
        if start_idx + n_frame_samples >= total_samples:
            continue

        t_vec = np.arange(n_frame_samples) / raw_fs
        chip_pos = t_vec * 2e6 * (1.0 + timing_ppm * 1e-6)
        envelope = amp * np.interp(chip_pos, np.arange(len(all_chips)), all_chips, left=0.0, right=0.0)

        carrier = np.exp(1j * (2.0 * np.pi * cfo_hz * t_vec + 0.3))
        iq_stream[start_idx : start_idx + n_frame_samples] += envelope * carrier

    return iq_stream
