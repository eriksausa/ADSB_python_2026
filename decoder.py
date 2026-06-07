"""
Mode-S / ADS-B Bitstream Decoder and Online Stateful Aircraft Tracker.
Decodes DF17 Callsigns (TC 1-4), Airborne Positions (TC 9-18) with both local and
globally unambiguous Even/Odd CPR, and Airborne Velocities (TC 19), while computing
Doppler shifts, residual Carrier Frequency Offsets (CFO), EIRP, and kinematic deltas online.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from config import ReceiverConfig
from dsp import compute_eirp_dbm, compute_single_frame_doppler, lldistkm


def NL(lat: float) -> int:
    """
    Computes the Number of Longitude zones (NL) for a given latitude in degrees
    according to ICAO Annex 10 / RTCA DO-260B Compact Position Reporting (CPR).
    """
    if np.isnan(lat):
        return 1
    abs_lat = np.abs(lat)
    if abs_lat >= 89.0:
        return 1
    if abs_lat == 0.0:
        return 59

    nz = 15.0
    a = 1.0 - np.cos(np.pi / (2.0 * nz))
    b = np.cos(np.radians(lat)) ** 2
    ratio = 1.0 - a / b
    if ratio <= -1.0:
        return 1
    nl = np.floor((2.0 * np.pi) / np.arccos(np.clip(ratio, -1.0, 1.0)))
    return int(nl)


def ADSB_callsign(bits_51: np.ndarray) -> str:
    """
    Decodes an 8-character flight callsign and aircraft emitter category from bits 38..88.
    """
    a = np.squeeze(bits_51).astype(int)
    pow_3 = 2 ** np.arange(2, -1, -1)
    pow_6 = 2 ** np.arange(5, -1, -1)

    ec = int(np.dot(a[0:3], pow_3))
    char_bits = a[3:51].reshape(8, 6)
    char_codes = np.dot(char_bits, pow_6)

    chars = []
    for code in char_codes:
        if 1 <= code <= 26:
            chars.append(chr(int(code) + 64))
        elif code == 32 or (48 <= code <= 57):
            chars.append(chr(int(code)))
        else:
            chars.append(" ")

    callsign = "".join(chars).strip()
    if ec == 5:
        callsign += " Heavy"
    elif ec == 4:
        callsign += " High vortex aircraft"
    return callsign


def ADSB_altitude(bits_12: np.ndarray) -> float:
    """Decodes 12-bit Mode-S barometric altitude (feet)."""
    a = np.squeeze(bits_12).astype(int)
    q_bit = a[7]
    alt_bits = np.concatenate([a[0:7], a[8:12]])
    pow_11 = 2 ** np.arange(10, -1, -1)
    alt = float(np.dot(alt_bits, pow_11))
    return (alt * 25.0) - 1000.0 if q_bit == 1 else (alt * 1000.0) - 1000.0


def ADSB_position(
    bits_36: np.ndarray, rx_lat: float = 48.196674, rx_lon: float = 16.370998
) -> List[float]:
    """
    Locally unambiguous CPR position decoding relative to receiver antenna coordinates.
    Returns [Latitude (deg), Longitude (deg), Even_Odd (0/1), Great-Circle Ground Distance (km)].
    """
    a = np.squeeze(bits_36).astype(int)
    e_o = int(a[1])

    dlat = 360.0 / (60.0 - e_o)
    pow_17 = 2 ** np.arange(16, -1, -1)
    lat_c = float(np.dot(a[2:19], pow_17))
    lon_c = float(np.dot(a[19:36], pow_17))

    j = np.floor(rx_lat / dlat) + np.floor(
        (np.fmod(rx_lat, dlat) / dlat) - (lat_c / (2 ** 17)) + 0.5
    )
    lat = dlat * (j + (lat_c / (2 ** 17)))

    nl_lat = NL(lat)
    dlon = 360.0 if (nl_lat - e_o) <= 0 else 360.0 / (nl_lat - e_o)

    m = np.floor(rx_lon / dlon) + np.floor(
        (np.fmod(rx_lon, dlon) / dlon) - (lon_c / (2 ** 17)) + 0.5
    )
    lon = dlon * (m + (lon_c / (2 ** 17)))

    cos_arg = (
        np.cos(np.radians(rx_lat)) * np.cos(np.radians(lat)) * np.cos(np.radians(rx_lon - lon))
        + np.sin(np.radians(rx_lat)) * np.sin(np.radians(lat))
    )
    dist_km = 6378.0 * np.arccos(np.clip(cos_arg, -1.0, 1.0))
    return [float(lat), float(lon), e_o, float(dist_km)]


def two_message_position(even_frame: Dict[str, Any], odd_frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Globally unambiguous CPR airborne position decoding from an Even/Odd frame pair.
    Returns dict with global_lat, global_lon, and cpr_zone_valid flag.
    """
    e_bits = np.squeeze(even_frame["Bits"]).astype(int)
    o_bits = np.squeeze(odd_frame["Bits"]).astype(int)

    pow_17 = 2 ** np.arange(16, -1, -1)
    cpr_lat_even = float(np.dot(e_bits[54:71], pow_17)) / (2 ** 17)
    cpr_lat_odd = float(np.dot(o_bits[54:71], pow_17)) / (2 ** 17)
    cpr_lon_even = float(np.dot(e_bits[71:88], pow_17)) / (2 ** 17)
    cpr_lon_odd = float(np.dot(o_bits[71:88], pow_17)) / (2 ** 17)

    j = np.floor(59.0 * cpr_lat_even - 60.0 * cpr_lat_odd + 0.5)
    dlat_e = 6.0
    dlat_o = 360.0 / 59.0

    lat_e = dlat_e * (np.fmod(j, 60.0) + cpr_lat_even)
    lat_o = dlat_o * (np.fmod(j, 59.0) + cpr_lat_odd)

    if lat_e > 270.0:
        lat_e -= 360.0
    if lat_o > 270.0:
        lat_o -= 360.0

    zone_valid = NL(lat_e) == NL(lat_o)
    e_ts = even_frame.get("Timestamp", 0.0)
    o_ts = odd_frame.get("Timestamp", 0.0)

    if e_ts >= o_ts:
        lat = lat_e
        ni = max(NL(lat_e), 1)
        dlon = 360.0 / ni
        m = np.floor(cpr_lon_even * (NL(lat_e) - 1) - cpr_lon_odd * NL(lat_e) + 0.5)
        lon = dlon * (np.fmod(m, ni) + cpr_lon_even)
    else:
        lat = lat_o
        ni = max(NL(lat_o) - 1, 1)
        dlon = 360.0 / ni
        m = np.floor(cpr_lon_even * (NL(lat_e) - 1) - cpr_lon_odd * NL(lat_o) + 0.5)
        lon = dlon * (np.fmod(m, ni) + cpr_lon_odd)

    if lon > 180.0:
        lon -= 360.0

    return {
        "global_lat": float(lat),
        "global_lon": float(lon),
        "cpr_zone_valid": bool(zone_valid),
    }


def ADSB_velocity(bits_51: np.ndarray) -> List[float]:
    """
    Decodes Subtype 1 (Groundspeed) and Subtype 3 (Airspeed / Heading) velocity frames.
    Returns [Speed (knots), Heading (deg), Vertical Rate (ft/min)].
    """
    a = np.squeeze(bits_51).astype(int)
    pow_3 = 2 ** np.arange(2, -1, -1)
    pow_10 = 2 ** np.arange(9, -1, -1)
    pow_9 = 2 ** np.arange(8, -1, -1)

    sub_type = int(np.dot(a[0:3], pow_3))
    vel_vec = [float("nan"), float("nan"), float("nan")]

    if sub_type == 1:
        ew_sign = a[8]
        ns_sign = a[19]
        v_ew = int(np.dot(a[9:19], pow_10))
        v_ns = int(np.dot(a[20:30], pow_10))

        v_we = -1.0 * (v_ew - 1) if ew_sign == 1 else float(v_ew - 1)
        v_sn = -1.0 * (v_ns - 1) if ns_sign == 1 else float(v_ns - 1)

        vel_vec[0] = float(np.sqrt(v_sn ** 2 + v_we ** 2))
        head = float(np.degrees(np.arctan2(v_we, v_sn)))
        if head < 0:
            head += 360.0
        vel_vec[1] = head

        v_rate_sign = a[31]
        v_rate = (int(np.dot(a[32:41], pow_9)) - 1) * 64.0
        if v_rate_sign == 1:
            v_rate *= -1.0
        vel_vec[2] = float(v_rate)

    elif sub_type == 3:
        s_h = a[8]
        vel_vec[1] = float(np.dot(a[9:19], pow_10)) * (360.0 / 1024.0) if s_h else float("nan")
        vel_vec[0] = float(np.dot(a[20:30], pow_10))

        v_rate_sign = a[31]
        v_rate = (int(np.dot(a[32:41], pow_9)) - 1) * 64.0
        if v_rate_sign == 1:
            v_rate *= -1.0
        vel_vec[2] = float(v_rate)

    return vel_vec


@dataclass
class AircraftOnlineState:
    """In-memory state per ICAO address for online Doppler, CPR, and kinematic jump tracking."""
    icao_hex: str
    callsign: Optional[str] = None
    last_airspeed_kt: float = float("nan")
    last_heading_deg: float = float("nan")
    last_vertical_fpm: float = float("nan")
    last_even_frame: Optional[Dict[str, Any]] = None
    last_odd_frame: Optional[Dict[str, Any]] = None
    last_pos_timestamp: Optional[float] = None
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    last_alt_ft: Optional[float] = None
    last_frame_timestamp: Optional[float] = None
    frame_count: int = 0


class OnlineADSBDecoder:
    """
    Stateful online Mode-S / ADS-B decoder.
    Processes raw detected frames one-by-one as they arrive from the stream reader,
    maintaining per-aircraft velocity and CPR state to compute Doppler, CFO, EIRP,
    and kinematic jump metrics immediately without batch post-processing.
    """

    # 17-bit CPR quantization grid resolution (~5.1 m lat, ~7.5 m lon -> 15 m combined tolerance)
    CPR_QUANTIZATION_TOLERANCE_KM: float = 0.0
    ALT_QUANTIZATION_TOLERANCE_FT: float = 0.0

    def __init__(self, config: Optional[ReceiverConfig] = None):
        self.config = config or ReceiverConfig()
        self.states: Dict[str, AircraftOnlineState] = {}
        self._pow_df = 2 ** np.arange(4, -1, -1)
        self._pow_icao = 2 ** np.arange(23, -1, -1)
        self._pow_tc = 2 ** np.arange(4, -1, -1)

    def decode_frame(self, raw_struct: Dict[str, Any]) -> Dict[str, Any]:
        """Decodes a single CRC-validated Mode-S frame and updates online aircraft state."""
        bits = np.squeeze(raw_struct["Bits"]).astype(int)
        df = int(np.dot(bits[0:5], self._pow_df))
        icao = int(np.dot(bits[8:32], self._pow_icao))
        icao_hex = f"{icao:06X}"
        tc = int(np.dot(bits[32:37], self._pow_tc))
        timestamp = float(raw_struct["Timestamp"])

        state = self.states.setdefault(icao_hex, AircraftOnlineState(icao_hex=icao_hex))
        state.frame_count += 1

        inter_arrival_s = (
            timestamp - state.last_frame_timestamp
            if state.last_frame_timestamp is not None
            else float("nan")
        )
        state.last_frame_timestamp = timestamp

        # Default telemetry fields
        callsign = state.callsign
        altitude_ft = float("nan")
        lat = float("nan")
        lon = float("nan")
        even_odd: Optional[int] = None
        slant_distance_m = float("nan")
        doppler_hz = float("nan")
        cfo_raw_hz = float(raw_struct.get("CFO_Raw_Hz", float("nan")))
        freq_offset_hz = cfo_raw_hz
        eirp_dbm = float("nan")
        airspeed_kt = state.last_airspeed_kt
        heading_deg = state.last_heading_deg
        vertical_fpm = state.last_vertical_fpm
        cpr_zone_valid = True
        implied_groundspeed_kt = float("nan")
        implied_alt_rate_fpm = float("nan")
        pos_jump_km = float("nan")
        msg_payload: Any = None

        if tc < 5:
            callsign = ADSB_callsign(bits[37:88])
            state.callsign = callsign
            msg_payload = callsign

        elif 8 < tc < 19:
            altitude_ft = ADSB_altitude(bits[40:52])
            pos = ADSB_position(bits[52:88], self.config.rx_lat, self.config.rx_lon)
            lat, lon, even_odd, ground_dist_km = pos[0], pos[1], int(pos[2]), pos[3]

            # Update Even/Odd CPR state and check global 2-message position consistency
            if even_odd == 0:
                state.last_even_frame = raw_struct
            else:
                state.last_odd_frame = raw_struct

            if state.last_even_frame is not None and state.last_odd_frame is not None:
                cpr_res = two_message_position(state.last_even_frame, state.last_odd_frame)
                cpr_zone_valid = cpr_res["cpr_zone_valid"]

            dist_m = ground_dist_km * 1000.0
            alt_m = max(altitude_ft, 0.0) * 0.3048
            slant_distance_m = float(np.sqrt(dist_m ** 2 + alt_m ** 2))
            eirp_dbm = compute_eirp_dbm(
                float(raw_struct["Power"]), slant_distance_m, self.config.carrier_freq_hz
            )

            # Online kinematic jump tracking relative to previous position of this aircraft
            if (
                state.last_pos_timestamp is not None
                and state.last_lat is not None
                and state.last_lon is not None
            ):
                dt_pos = timestamp - state.last_pos_timestamp
                if dt_pos > 1e-4:
                    d_km, _ = lldistkm([state.last_lat, state.last_lon], [lat, lon])
                    pos_jump_km = d_km
                    # Subtract 17-bit CPR quantization deadband (15 m) so sub-grid rounding isn't mistaken for motion
                    eff_km = max(0.0, d_km - self.CPR_QUANTIZATION_TOLERANCE_KM)
                    implied_groundspeed_kt = float((eff_km / dt_pos) * 1943.844)
                    if state.last_alt_ft is not None and not np.isnan(altitude_ft):
                        eff_alt_diff = max(
                            0.0, abs(altitude_ft - state.last_alt_ft) - self.ALT_QUANTIZATION_TOLERANCE_FT
                        )
                        implied_alt_rate_fpm = float(eff_alt_diff / (dt_pos / 60.0))

            state.last_pos_timestamp = timestamp
            state.last_lat = lat
            state.last_lon = lon
            state.last_alt_ft = altitude_ft

            # Online Doppler & Doppler-compensated Carrier Frequency Offset (CFO)
            doppler_hz = compute_single_frame_doppler(
                lat, lon, altitude_ft, airspeed_kt, heading_deg, self.config
            )
            if not np.isnan(cfo_raw_hz) and not np.isnan(doppler_hz):
                freq_offset_hz = float(cfo_raw_hz - doppler_hz)

            msg_payload = {
                "Altitude": altitude_ft,
                "Latitude": lat,
                "Longitude": lon,
                "Even_Odd": even_odd,
                "Distance_from_Rx": slant_distance_m,
                "Doppler": doppler_hz,
                "Frequency_shift": freq_offset_hz,
            }

        elif tc == 19:
            v_h = ADSB_velocity(bits[37:88])
            airspeed_kt, heading_deg, vertical_fpm = v_h[0], v_h[1], v_h[2]
            state.last_airspeed_kt = airspeed_kt
            state.last_heading_deg = heading_deg
            state.last_vertical_fpm = vertical_fpm

            if state.last_lat is not None and state.last_lon is not None and state.last_alt_ft is not None:
                doppler_hz = compute_single_frame_doppler(
                    state.last_lat,
                    state.last_lon,
                    state.last_alt_ft,
                    airspeed_kt,
                    heading_deg,
                    self.config,
                )
                if not np.isnan(cfo_raw_hz) and not np.isnan(doppler_hz):
                    freq_offset_hz = float(cfo_raw_hz - doppler_hz)

            msg_payload = {
                "Airspeed": airspeed_kt,
                "Heading": heading_deg,
                "Vertical": vertical_fpm,
            }

        return {
            "Bits": bits,
            "Timestamp": timestamp,
            "Power": float(raw_struct["Power"]),
            "Amplitude": float(raw_struct["Amplitude"]),
            "EIRP_dBm": float(eirp_dbm),
            "DF": df,
            "ICAO": icao,
            "ICAO_Hex": icao_hex,
            "TC": tc,
            "Call_sign": callsign,
            "Altitude": altitude_ft,
            "Latitude": lat,
            "Longitude": lon,
            "Even_Odd": even_odd,
            "Distance_from_Rx": slant_distance_m,
            "Airspeed": airspeed_kt,
            "Heading": heading_deg,
            "Vertical": vertical_fpm,
            "Doppler": doppler_hz,
            "CFO_Raw_Hz": cfo_raw_hz,
            "Frequency_shift": freq_offset_hz,
            "Timing_Offset_PPM": float(raw_struct.get("Timing_Offset_PPM", 0.0)),
            "Sub_Sample_Lag_Ns": float(raw_struct.get("Sub_Sample_Lag_Ns", 0.0)),
            "Phase_Noise_Std": float(raw_struct.get("Phase_Noise_Std", float("nan"))),
            "Inter_Arrival_s": float(inter_arrival_s),
            "CPR_Zone_Valid": bool(cpr_zone_valid),
            "Pos_Jump_km": float(pos_jump_km),
            "Implied_Groundspeed_kt": float(implied_groundspeed_kt),
            "Implied_Alt_Rate_fpm": float(implied_alt_rate_fpm),
            "Message": msg_payload,
            "IQ": raw_struct.get("IQ"),
            "IQ_res": raw_struct.get("IQ_resampled"),
        }