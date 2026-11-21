"""
Relational SQL Database Module for ADS-B RF Fingerprinting & Telemetry Storage
(`src/adsb_rf/database.py`).

Uses DuckDB when installed with automatic fallback to Python's built-in `sqlite3`.
Sanitizes recorded source filenames to basenames only (preventing username/path leakage)
and persists all 12 physical-layer RF features alongside CRC-failed blind attributions.
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    import sqlite3
    HAS_DUCKDB = False


def _clean_sql_val(val: Any) -> Any:
    """Converts NaN/inf and NumPy scalars into native SQL-safe Python primitives."""
    if val is None:
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _clean_bool(val: Any, default: bool = False) -> bool:
    return bool(default if val is None else val)


def _sanitize_source_label(source_file: str | Path) -> str:
    """Strips directory paths so only the file basename or synthetic label is persisted."""
    raw = str(source_file).strip()
    return Path(raw.replace("\\", "/")).name if raw else "unknown_source"


# Declarative field mapping: (sql_column, sql_type, frame_dict_key, sanitizer)
_FRAME_SCHEMA: List[Tuple[str, str, str, Callable[[Any], Any]]] = [
    ("timestamp", "DOUBLE", "Timestamp", _clean_sql_val),
    ("icao", "INTEGER", "ICAO", _clean_sql_val),
    ("icao_hex", "VARCHAR(8)", "ICAO_Hex", _clean_sql_val),
    ("df", "INTEGER", "DF", _clean_sql_val),
    ("tc", "INTEGER", "TC", _clean_sql_val),
    ("callsign", "VARCHAR(16)", "Call_sign", _clean_sql_val),
    ("altitude_ft", "DOUBLE", "Altitude", _clean_sql_val),
    ("latitude", "DOUBLE", "Latitude", _clean_sql_val),
    ("longitude", "DOUBLE", "Longitude", _clean_sql_val),
    ("even_odd", "INTEGER", "Even_Odd", _clean_sql_val),
    ("distance_m", "DOUBLE", "Distance_from_Rx", _clean_sql_val),
    ("airspeed_kt", "DOUBLE", "Airspeed", _clean_sql_val),
    ("heading_deg", "DOUBLE", "Heading", _clean_sql_val),
    ("vertical_rate_fpm", "DOUBLE", "Vertical", _clean_sql_val),
    ("rx_power_dbm", "DOUBLE", "Power", _clean_sql_val),
    ("amplitude", "DOUBLE", "Amplitude", _clean_sql_val),
    ("eirp_dbm", "DOUBLE", "EIRP_dBm", _clean_sql_val),
    ("doppler_hz", "DOUBLE", "Doppler", _clean_sql_val),
    ("cfo_raw_hz", "DOUBLE", "CFO_Raw_Hz", _clean_sql_val),
    ("freq_offset_hz", "DOUBLE", "Frequency_shift", _clean_sql_val),
    ("timing_offset_ppm", "DOUBLE", "Timing_Offset_PPM", _clean_sql_val),
    ("sub_sample_lag_ns", "DOUBLE", "Sub_Sample_Lag_Ns", _clean_sql_val),
    ("phase_noise_std", "DOUBLE", "Phase_Noise_Std", _clean_sql_val),
    ("pa_droop_db", "DOUBLE", "PA_Droop_dB", _clean_sql_val),
    ("pulse_crest_factor", "DOUBLE", "Pulse_Crest_Factor", _clean_sql_val),
    ("phase_curvature_rad", "DOUBLE", "Phase_Curvature_Rad", _clean_sql_val),
    ("snr_db", "DOUBLE", "SNR_dB", _clean_sql_val),
    ("preamble_ratio_db", "DOUBLE", "Preamble_Ratio_dB", _clean_sql_val),
    ("inter_arrival_s", "DOUBLE", "Inter_Arrival_s", _clean_sql_val),
    ("cpr_zone_valid", "BOOLEAN", "CPR_Zone_Valid", lambda v: _clean_bool(v, True)),
    ("pos_jump_km", "DOUBLE", "Pos_Jump_km", _clean_sql_val),
    ("implied_groundspeed_kt", "DOUBLE", "Implied_Groundspeed_kt", _clean_sql_val),
    ("implied_alt_rate_fpm", "DOUBLE", "Implied_Alt_Rate_fpm", _clean_sql_val),
    ("predicted_icao_hex", "VARCHAR(8)", "Predicted_ICAO_Hex", _clean_sql_val),
    ("rf_match_prob", "DOUBLE", "RF_Match_Prob", _clean_sql_val),
    ("anomaly_score", "DOUBLE", "Anomaly_Score", _clean_sql_val),
    ("is_anomaly", "BOOLEAN", "Is_Anomaly", lambda v: _clean_bool(v, False)),
    ("anomaly_reasons", "VARCHAR", "Anomaly_Reasons", lambda v: _clean_sql_val(v if v is not None else "")),
]

_FRAME_INSERT_SQL = (
    "INSERT INTO adsb_frames VALUES (" + ", ".join(["?"] * (2 + len(_FRAME_SCHEMA))) + ")"
)


class ADSBDatabase:
    """Analytical SQL store for decoded ADS-B telemetry, RF fingerprints, and anomaly verdicts."""

    def __init__(self, db_path: str | Path = "adsb_rf_telemetry.duckdb"):
        self.db_path = Path(db_path)
        self.engine_name = "duckdb" if HAS_DUCKDB else "sqlite3 (duckdb-compatible mode)"
        self.conn = duckdb.connect(str(self.db_path)) if HAS_DUCKDB else sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id INTEGER PRIMARY KEY,
                source_file VARCHAR,
                carrier_freq_hz DOUBLE,
                raw_sample_rate_hz DOUBLE,
                target_sample_rate_hz DOUBLE,
                rx_lat DOUBLE,
                rx_lon DOUBLE,
                total_samples BIGINT,
                duration_s DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        frame_cols = ",\n                ".join(f"{col} {dtype}" for col, dtype, _, _ in _FRAME_SCHEMA)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS adsb_frames (
                frame_id INTEGER PRIMARY KEY,
                measurement_id INTEGER,
                {frame_cols}
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crc_failed_frames (
                burst_id INTEGER PRIMARY KEY,
                measurement_id INTEGER,
                timestamp DOUBLE,
                corrupted_icao_hex VARCHAR(8),
                predicted_icao_hex VARCHAR(8),
                predicted_callsign VARCHAR(16),
                icao_bits_corrupted BOOLEAN,
                rf_confidence DOUBLE,
                mahalanobis_dist DOUBLE,
                cfo_raw_hz DOUBLE,
                freq_offset_hz DOUBLE,
                timing_offset_ppm DOUBLE,
                rx_power_dbm DOUBLE,
                eirp_dbm DOUBLE,
                phase_noise_std DOUBLE,
                clean_slot_ratio DOUBLE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS aircraft_rf_profiles (
                icao_hex VARCHAR(8) PRIMARY KEY,
                callsign VARCHAR(16),
                frame_count INTEGER,
                crc_recovered_count INTEGER,
                mean_freq_offset_hz DOUBLE,
                std_freq_offset_hz DOUBLE,
                mean_timing_offset_ppm DOUBLE,
                std_timing_offset_ppm DOUBLE,
                mean_eirp_dbm DOUBLE,
                std_eirp_dbm DOUBLE,
                mean_phase_noise_rad DOUBLE,
                anomaly_count INTEGER,
                first_seen DOUBLE,
                last_seen DOUBLE
            );
            """
        )
        self.conn.commit()
        self._next_frame_id = self._get_max_id("adsb_frames", "frame_id") + 1
        self._next_burst_id = self._get_max_id("crc_failed_frames", "burst_id") + 1
        self._next_meas_id = self._get_max_id("measurements", "measurement_id") + 1

    def _get_max_id(self, table: str, col: str) -> int:
        res = self.conn.cursor().execute(f"SELECT COALESCE(MAX({col}), 0) FROM {table}").fetchone()
        return int(res[0]) if res and res[0] is not None else 0

    def register_measurement(
        self,
        source_file: str,
        carrier_freq_hz: float,
        raw_sample_rate_hz: float,
        target_sample_rate_hz: float,
        rx_lat: float,
        rx_lon: float,
        total_samples: int,
        duration_s: float,
    ) -> int:
        """Inserts a measurement session (with sanitized basename) and returns `measurement_id`."""
        meas_id = self._next_meas_id
        self._next_meas_id += 1
        self.conn.cursor().execute(
            """
            INSERT INTO measurements (
                measurement_id, source_file, carrier_freq_hz, raw_sample_rate_hz,
                target_sample_rate_hz, rx_lat, rx_lon, total_samples, duration_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meas_id,
                _sanitize_source_label(source_file),
                float(carrier_freq_hz),
                float(raw_sample_rate_hz),
                float(target_sample_rate_hz),
                float(rx_lat),
                float(rx_lon),
                int(total_samples),
                float(duration_s),
            ),
        )
        self.conn.commit()
        return meas_id

    def insert_frame_online(self, frame: Dict[str, Any], measurement_id: int = 1) -> int:
        """Inserts one decoded ADS-B frame during online streaming and returns `frame_id`."""
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        frame["frame_id"] = frame_id
        row = (frame_id, measurement_id) + tuple(fn(frame.get(key)) for _, _, key, fn in _FRAME_SCHEMA)
        self.conn.cursor().execute(_FRAME_INSERT_SQL, row)
        self.conn.commit()
        return frame_id

    def insert_crc_failed_bursts(self, failed_bursts: List[Dict[str, Any]], measurement_id: int = 1) -> None:
        """Persists CRC-failed bursts and their blind RF Fingerprinting aircraft attributions."""
        rows = []
        for b in failed_bursts:
            burst_id = self._next_burst_id
            self._next_burst_id += 1
            b["burst_id"] = burst_id
            rows.append(
                (
                    burst_id,
                    measurement_id,
                    _clean_sql_val(b.get("Timestamp")),
                    _clean_sql_val(b.get("Corrupted_ICAO_Hex")),
                    _clean_sql_val(b.get("Predicted_ICAO_Hex")),
                    _clean_sql_val(b.get("Predicted_Callsign")),
                    bool(b.get("ICAO_Bits_Were_Corrupted", False)),
                    _clean_sql_val(b.get("RF_Confidence")),
                    _clean_sql_val(b.get("Mahalanobis_Dist")),
                    _clean_sql_val(b.get("CFO_Raw_Hz")),
                    _clean_sql_val(b.get("Frequency_shift")),
                    _clean_sql_val(b.get("Timing_Offset_PPM")),
                    _clean_sql_val(b.get("Power")),
                    _clean_sql_val(b.get("EIRP_dBm")),
                    _clean_sql_val(b.get("Phase_Noise_Std")),
                    _clean_sql_val(b.get("Clean_Slot_Ratio")),
                )
            )
        if rows:
            self.conn.cursor().executemany(
                "INSERT INTO crc_failed_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()
        self.refresh_aircraft_profiles()

    def update_ml_predictions(self, frames: List[Dict[str, Any]]) -> None:
        """Batch-updates ML classifications, anomaly flags, and backfilled physical features."""
        rows = [
            (
                _clean_sql_val(f.get("Distance_from_Rx")),
                _clean_sql_val(f.get("EIRP_dBm")),
                _clean_sql_val(f.get("Doppler")),
                _clean_sql_val(f.get("CFO_Raw_Hz")),
                _clean_sql_val(f.get("Frequency_shift")),
                _clean_sql_val(f.get("Phase_Noise_Std")),
                _clean_sql_val(f.get("PA_Droop_dB")),
                _clean_sql_val(f.get("Pulse_Crest_Factor")),
                _clean_sql_val(f.get("Phase_Curvature_Rad")),
                _clean_sql_val(f.get("SNR_dB")),
                _clean_sql_val(f.get("Preamble_Ratio_dB")),
                _clean_sql_val(f.get("Predicted_ICAO_Hex")),
                _clean_sql_val(f.get("RF_Match_Prob")),
                _clean_sql_val(f.get("Anomaly_Score")),
                bool(f.get("Is_Anomaly", False)),
                _clean_sql_val(f.get("Anomaly_Reasons", "")),
                int(f["frame_id"]),
            )
            for f in frames
            if "frame_id" in f
        ]
        if rows:
            self.conn.cursor().executemany(
                """
                UPDATE adsb_frames
                SET distance_m = COALESCE(?, distance_m),
                    eirp_dbm = COALESCE(?, eirp_dbm),
                    doppler_hz = COALESCE(?, doppler_hz),
                    cfo_raw_hz = COALESCE(?, cfo_raw_hz),
                    freq_offset_hz = COALESCE(?, freq_offset_hz),
                    phase_noise_std = COALESCE(?, phase_noise_std),
                    pa_droop_db = COALESCE(?, pa_droop_db),
                    pulse_crest_factor = COALESCE(?, pulse_crest_factor),
                    phase_curvature_rad = COALESCE(?, phase_curvature_rad),
                    snr_db = COALESCE(?, snr_db),
                    preamble_ratio_db = COALESCE(?, preamble_ratio_db),
                    predicted_icao_hex = ?,
                    rf_match_prob = ?,
                    anomaly_score = ?,
                    is_anomaly = ?,
                    anomaly_reasons = ?
                WHERE frame_id = ?
                """,
                rows,
            )
            self.conn.commit()
        self.refresh_aircraft_profiles()

    def refresh_aircraft_profiles(self) -> pd.DataFrame:
        """Recomputes per-aircraft summary rows in `aircraft_rf_profiles` from `adsb_frames`."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM aircraft_rf_profiles;")

        df_frames = self.query_df("SELECT * FROM adsb_frames ORDER BY timestamp ASC")
        if df_frames.empty:
            self.conn.commit()
            return pd.DataFrame()

        df_crc_failed = self.query_df("SELECT predicted_icao_hex FROM crc_failed_frames")
        crc_counts = df_crc_failed["predicted_icao_hex"].value_counts().to_dict() if not df_crc_failed.empty else {}

        profile_rows = []
        for icao_hex, grp in df_frames.groupby("icao_hex"):
            clean_mask = ~grp["is_anomaly"].astype(bool)
            clean_grp = grp[clean_mask] if clean_mask.any() else grp
            callsigns = grp["callsign"].dropna()
            profile_rows.append(
                (
                    str(icao_hex),
                    str(callsigns.iloc[-1]) if not callsigns.empty else None,
                    int(len(grp)),
                    int(crc_counts.get(str(icao_hex), 0)),
                    _clean_sql_val(clean_grp["freq_offset_hz"].mean()),
                    _clean_sql_val(clean_grp["freq_offset_hz"].std(ddof=0)),
                    _clean_sql_val(clean_grp["timing_offset_ppm"].mean()),
                    _clean_sql_val(clean_grp["timing_offset_ppm"].std(ddof=0)),
                    _clean_sql_val(clean_grp["eirp_dbm"].mean()),
                    _clean_sql_val(clean_grp["eirp_dbm"].std(ddof=0)),
                    _clean_sql_val(clean_grp["phase_noise_std"].mean()),
                    int(grp["is_anomaly"].astype(bool).sum()),
                    _clean_sql_val(grp["timestamp"].min()),
                    _clean_sql_val(grp["timestamp"].max()),
                )
            )

        cur.executemany(
            "INSERT INTO aircraft_rf_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            profile_rows,
        )
        self.conn.commit()
        return self.query_df("SELECT * FROM aircraft_rf_profiles ORDER BY frame_count DESC")

    def query_df(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Executes a SQL query and returns results as a Pandas DataFrame."""
        cur = self.conn.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        return pd.DataFrame(cur.fetchall(), columns=cols)

    def export_csv(self, filename: str = "adsb_table.csv") -> pd.DataFrame:
        """Exports the full `adsb_frames` table to CSV ordered by timestamp."""
        df = self.query_df("SELECT * FROM adsb_frames ORDER BY timestamp ASC")
        df.to_csv(filename, index=False)
        return df

    def close(self) -> None:
        self.conn.close()
