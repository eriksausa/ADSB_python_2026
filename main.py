"""
Main Entry Point & Portfolio Demonstration Runner for the Modular ADS-B Pipeline.
Connects:
1. `config.py` (VSA XML settings & receiver calibration)
2. `stream_reader.py` (Continuous / online single-measurement IQ reading — no 1-second batch loop)
3. `dsp.py` (Preamble cross-correlation, CRC-24, amplitude/EIRP, CFO, symbol timing offset,
   and BLIND RF feature extraction for CRC-failed bursts)
4. `decoder.py` (Online stateful Mode-S decoder with real-time Doppler & CPR position pairing)
5. `database.py` (Local DuckDB analytical SQL storage, aircraft RF profiles, & `crc_failed_frames` table)
6. `anomaly_ml.py` (Supervised RF Fingerprinting classifier + Kinematic anomaly detection +
   Blind attribution of CRC-failed bursts)
7. `visualization.py` (Interactive HTML map, 2D RF Fingerprint cluster plot, IQ/Eye diagrams)
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from anomaly_ml import ADSBAnomalyDetector
from config import AnomalyThresholds, ReceiverConfig, parse_vsa_settings
from database import ADSBDatabase
from decoder import OnlineADSBDecoder, NL
from dsp import find_valid_xcorr
from stream_reader import OnlineIQStreamReader
from visualization import (
    map_planes,
    plot_eye_diagram,
    plot_iq_new,
    plot_rf_fingerprint_clusters,
)


# =============================================================================
# MODE-S DF17 RF STREAM SYNTHESIZER (WITH CRC-VALID, ANOMALOUS & CRC-FAILED BURSTS)
# =============================================================================

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
    config: ReceiverConfig, raw_fs: float = 12.8e6
) -> np.ndarray:
    """
    Synthesizes a continuous complex IQ measurement at `raw_fs` (12.8 MS/s) containing:
    - 24 legitimate CRC-valid frames from 3 aircraft (`4401A2`, `3C6541`, `4B1809`),
    - 3 CRC-valid security/data-validity anomalies (RF Spoofing, Teleportation Jump, Extreme CFO/EIRP),
    - 3 CRC-FAILED bursts (where mid-frame bit errors / noise corrupt both the 24-bit ICAO address
      and payload bits so CRC-24 fails, testing whether Blind RF Fingerprinting can still identify
      which aircraft transmitted each burst!).
    """
    rng = np.random.default_rng(42)
    duration_s = 0.058  # 58 ms continuous capture (~742,400 samples at 12.8 MS/s)
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

        dlat_step = (ac["v_ns"] * 0.514444 * dt_step) / 111320.0
        dlon_step = (ac["v_ew"] * 0.514444 * dt_step) / (111320.0 * np.cos(np.radians(ac["lat0"])))

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

    # Inject 3 security/data-validity anomalies (CRC-valid):
    spoof_bits = _build_df17_position_bits(0x4401A2, 31000.0, 48.2450, 16.4800, 0)
    events.append((t_cursor, spoof_bits, -28500.0, -95.0, 0.00095))
    t_cursor += 0.0018

    jump_bits = _build_df17_position_bits(0x3C6541, 36000.0, 48.5200, 16.6500, 1)
    events.append((t_cursor, jump_bits, -18400.0, -69.0, 0.00072))
    t_cursor += 0.0018

    bad_alt_bits = _build_df17_position_bits(0x4B1809, 50000.0, 48.2100, 16.3100, 0)
    events.append((t_cursor, bad_alt_bits, +64000.0, -11.0, 0.0095))
    t_cursor += 0.0018

    # Inject 3 CRC-FAILED bursts (one from each aircraft) where payload AND ICAO bits are corrupted
    # so standard Mode-S decoding fails CRC and reads a bogus ICAO address, while Blind RF Fingerprinting
    # recovers the true transmitting aircraft!
    for idx, ac in enumerate(aircraft_specs):
        corrupted_bits = _build_df17_position_bits(ac["icao"], ac["alt0"], ac["lat0"], ac["lon0"], idx % 2).copy()
        # Flip 2 bits inside the 24-bit ICAO address field (bits 8..31) + 3 bits in payload so CRC-24 fails
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


# =============================================================================
# END-TO-END ONLINE PIPELINE EXECUTION
# =============================================================================

def run_online_pipeline(
    measurement_path: Optional[str | Path] = None,
    vsa_setx_path: Optional[str | Path] = None,
    db_path: str | Path = "adsb_rf_telemetry.duckdb",
    output_dir: str | Path = ".",
) -> dict:
    """
    Executes the complete online ADS-B RF Fingerprinting, Anomaly Detection & Blind CRC Recovery workflow.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = parse_vsa_settings(vsa_setx_path) if vsa_setx_path else ReceiverConfig()
    thresholds = AnomalyThresholds()

    source_label = str(measurement_path) if measurement_path else "synthesized_12.8MSps_capture"
    use_synth = True
    if measurement_path and Path(measurement_path).exists():
        probe_reader = OnlineIQStreamReader(measurement_path, config=config)
        if probe_reader.total_samples >= int(config.frame_duration_s * probe_reader.raw_sample_rate_hz):
            reader = probe_reader
            use_synth = False
        else:
            print(
                f"[StreamReader] Note: '{Path(measurement_path).name}' contains {probe_reader.total_samples} samples "
                f"({probe_reader.duration_s * 1e6:.1f} us < 120 us single Mode-S frame). "
                "Synthesizing continuous 12.8 MS/s multi-aircraft RF stream for full pipeline run..."
            )

    if use_synth:
        synth_iq = synthesize_portfolio_iq_stream(config, raw_fs=config.default_raw_sample_rate_hz)
        reader = OnlineIQStreamReader(
            synth_iq,
            config=config,
            raw_sample_rate_hz=config.default_raw_sample_rate_hz,
            start_time_s=101000.0,
        )

    print(
        f"[StreamReader] Online source ready: {reader.total_samples:,} IQ samples "
        f"at {reader.raw_sample_rate_hz / 1e6:.2f} MS/s ({reader.duration_s * 1e3:.1f} ms continuous capture)"
    )

    db_file = out_dir / Path(db_path).name
    if db_file.exists():
        db_file.unlink()
    db = ADSBDatabase(db_path=db_file)
    meas_id = db.register_measurement(
        source_file=source_label,
        carrier_freq_hz=config.carrier_freq_hz,
        raw_sample_rate_hz=reader.raw_sample_rate_hz,
        target_sample_rate_hz=config.target_sample_rate_hz,
        rx_lat=config.rx_lat,
        rx_lon=config.rx_lon,
        total_samples=reader.total_samples,
        duration_s=reader.duration_s,
    )

    decoder = OnlineADSBDecoder(config=config)
    decoded_frames: List[dict] = []
    crc_failed_bursts: List[dict] = []

    # Continuous Online Streaming (captures both CRC-valid frames AND CRC-failed bursts)
    for chunk in reader.stream_online(window_duration_s=None):
        valid_cands, failed_cands = find_valid_xcorr(
            chunk, config=config, capture_crc_failures=True
        )
        for cand in valid_cands:
            decoded = decoder.decode_frame(cand)
            db.insert_frame_online(decoded, measurement_id=meas_id)
            decoded_frames.append(decoded)
        crc_failed_bursts.extend(failed_cands)

    print(
        f"[OnlineDecoder] Ingested {len(decoded_frames)} CRC-valid frames and captured "
        f"{len(crc_failed_bursts)} CRC-failed bursts for blind RF attribution."
    )

    # Run ML RF Fingerprinting Classifier, Anomaly Detector, and Blind CRC-Failed Attribution
    detector = ADSBAnomalyDetector(thresholds=thresholds)
    ml_summary = detector.analyze_frames(decoded_frames, crc_failed_bursts=crc_failed_bursts)
    db.update_ml_predictions(decoded_frames)
    db.insert_crc_failed_bursts(crc_failed_bursts, measurement_id=meas_id)

    profiles_df = db.query_df("SELECT * FROM aircraft_rf_profiles ORDER BY frame_count DESC")
    anomalies_df = db.query_df(
        "SELECT frame_id, timestamp, icao_hex, predicted_icao_hex, rf_match_prob, "
        "freq_offset_hz, timing_offset_ppm, implied_groundspeed_kt, anomaly_reasons "
        "FROM adsb_frames WHERE is_anomaly = 1 ORDER BY timestamp ASC"
    )
    crc_failed_df = db.query_df(
        "SELECT burst_id, timestamp, corrupted_icao_hex, predicted_icao_hex, predicted_callsign, "
        "icao_bits_corrupted, rf_confidence, mahalanobis_dist, freq_offset_hz, timing_offset_ppm "
        "FROM crc_failed_frames ORDER BY timestamp ASC"
    )
    full_df = db.export_csv(str(out_dir / "adsb_table.csv"))

    # Generate Portfolio Visualizations
    map_path = map_planes(decoded_frames, output_html=out_dir / "adsb_map.html", rx_lat=config.rx_lat, rx_lon=config.rx_lon)
    cluster_plot_path = plot_rf_fingerprint_clusters(
        full_df, crc_failed_df=crc_failed_df, save_path=out_dir / "rf_fingerprint_clusters.png"
    )
    iq_plot_path = plot_iq_new(decoded_frames, fs=reader.raw_sample_rate_hz, save_path=out_dir / "iq_spectrum_sample.png")
    eye_plot_path = (
        plot_eye_diagram(decoded_frames[0]["IQ_res"], msps=10.0, save_path=out_dir / "eye_diagram_sample.png")
        if decoded_frames
        else None
    )

   #print("\n=== AIRCRAFT RF FINGERPRINT PROFILES (SQL AGGREGATION) ===")
   #print(profiles_df.to_string(index=False))

   #print("\n=== DETECTED ANOMALIES & SPOOFING EVENTS (CRC-VALID FRAMES) ===")
    #if not anomalies_df.empty:
        #print(anomalies_df.to_string(index=False))

    #print("\n=== BLIND RF ATTRIBUTION OF CRC-FAILED BURSTS (NO PAYLOAD DECODING) ===")
    #if not crc_failed_df.empty:
    #    print(crc_failed_df.to_string(index=False))

    print(
        f"\n[ML Summary] CRC-Valid Frames: {ml_summary['total_frames']} | "
        f"Clean RF Classification Accuracy: {ml_summary['rf_classification_accuracy']*100:.1f}% | "
        f"Anomalies Flagged: {ml_summary['anomalies_detected']} | "
        f"CRC-Failed Bursts Attributed: {ml_summary['crc_failed_attributed']}"
    )

    db.close()
    return {
        "summary": ml_summary,
        "profiles_df": profiles_df,
        "anomalies_df": anomalies_df,
        "crc_failed_df": crc_failed_df,
        "map_path": map_path,
        "cluster_plot_path": cluster_plot_path,
        "iq_plot_path": iq_plot_path,
        "eye_plot_path": eye_plot_path,
    }


if __name__ == "__main__":
    uploaded_dir = Path(__file__).resolve().parent  # Update this path if needed
    npy_file = uploaded_dir / "receivedSignals.npy"
    setx_file = uploaded_dir / "vsa_settings.setx"

    run_online_pipeline(
        measurement_path=npy_file if npy_file.exists() else None,
        vsa_setx_path=setx_file if setx_file.exists() else None,
        output_dir=Path(__file__).resolve().parent,
    )