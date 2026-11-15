"""
End-to-End Online Pipeline Runner (`src/adsb_rf/pipeline.py`).
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from .assumptions import AnomalyThresholds, ReceiverConfig, parse_vsa_settings
from .database import ADSBDatabase
from .decoder import OnlineADSBDecoder
from .dsp import find_valid_xcorr
from .models import ADSBAnomalyDetector
from .stream_reader import OnlineIQStreamReader
from .synthesizer import synthesize_portfolio_iq_stream
from .visualization import (
    map_planes,
    plot_eye_diagram,
    plot_iq_new,
    plot_rf_fingerprint_clusters,
)


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

    source_label = Path(measurement_path).name if measurement_path else "synthesized_12.8MSps_capture"
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

    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_planes(
        decoded_frames, output_html=outputs_dir / "adsb_map.html", rx_lat=config.rx_lat, rx_lon=config.rx_lon
    )
    cluster_plot_path = plot_rf_fingerprint_clusters(
        full_df, crc_failed_df=crc_failed_df, save_path=outputs_dir / "rf_fingerprint_clusters.png"
    )
    iq_plot_path = plot_iq_new(
        decoded_frames, fs=reader.raw_sample_rate_hz, save_path=outputs_dir / "iq_spectrum_sample.png"
    )
    eye_plot_path = (
        plot_eye_diagram(decoded_frames[0]["IQ_res"], msps=10.0, save_path=outputs_dir / "eye_diagram_sample.png")
        if decoded_frames
        else None
    )

    print(
        f"\n[ML Summary] CRC-Valid Frames: {ml_summary['total_frames']} | "
        f"In-Sample Clean Fit: {ml_summary['rf_classification_accuracy']*100:.1f}% | "
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
