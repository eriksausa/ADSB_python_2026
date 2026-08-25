"""
Relational SQL Database Module for ADS-B RF Fingerprinting and Telemetry Storage.
Uses local DuckDB (`duckdb`) as the primary analytical SQL engine with an automatic,
zero-friction fallback to `sqlite3` when running in minimal environments, and clean
SQL abstraction ready to upgrade to PostgreSQL / MotherDuck.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    import sqlite3
    HAS_DUCKDB = False


def _clean_sql_val(val: Any) -> Any:
    """Converts NaN/inf/NumPy scalars into standard Python/SQL types (None, float, int, str)."""
    if val is None:
        return None
    if isinstance(val, (float, int)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if hasattr(val, "item"):
        scalar = val.item()
        if isinstance(scalar, float) and (math.isnan(scalar) or math.isinf(scalar)):
            return None
        return scalar
    return val


class ADSBDatabase:
    """
    Analytical SQL storage layer for online ADS-B telemetry, physical-layer RF fingerprints,
    ML anomaly detection results, and blind RF attribution of CRC-failed bursts.
    """

    def __init__(self, db_path: str | Path = "adsb_rf_telemetry.duckdb"):
        self.db_path = Path(db_path)
        self.engine_name = "duckdb" if HAS_DUCKDB else "sqlite3 (duckdb-compatible mode)"
        if HAS_DUCKDB:
            self.conn = duckdb.connect(str(self.db_path))
        else:
            self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Creates normalized tables for measurements, valid frames, CRC-failed frames, and aircraft RF profiles."""
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

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS adsb_frames (
                frame_id INTEGER PRIMARY KEY,
                measurement_id INTEGER,
                timestamp DOUBLE,
                icao INTEGER,
                icao_hex VARCHAR(8),
                df INTEGER,
                tc INTEGER,
                callsign VARCHAR(16),
                altitude_ft DOUBLE,
                latitude DOUBLE,
                longitude DOUBLE,
                even_odd INTEGER,
                distance_m DOUBLE,
                airspeed_kt DOUBLE,
                heading_deg DOUBLE,
                vertical_rate_fpm DOUBLE,
                rx_power_dbm DOUBLE,
                amplitude DOUBLE,
                eirp_dbm DOUBLE,
                doppler_hz DOUBLE,
                cfo_raw_hz DOUBLE,
                freq_offset_hz DOUBLE,
                timing_offset_ppm DOUBLE,
                sub_sample_lag_ns DOUBLE,
                phase_noise_std DOUBLE,
                inter_arrival_s DOUBLE,
                cpr_zone_valid BOOLEAN,
                pos_jump_km DOUBLE,
                implied_groundspeed_kt DOUBLE,
                implied_alt_rate_fpm DOUBLE,
                predicted_icao_hex VARCHAR(8),
                rf_match_prob DOUBLE,
                anomaly_score DOUBLE,
                is_anomaly BOOLEAN,
                anomaly_reasons VARCHAR
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
        cur = self.conn.cursor()
        res = cur.execute(f"SELECT COALESCE(MAX({col}), 0) FROM {table}").fetchone()
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
        """Inserts a measurement session record and returns its `measurement_id`."""
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
                str(source_file),
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
        """Inserts a single decoded ADS-B frame during online streaming and returns `frame_id`."""
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        frame["frame_id"] = frame_id

        row = (
            frame_id,
            measurement_id,
            _clean_sql_val(frame.get("Timestamp")),
            _clean_sql_val(frame.get("ICAO")),
            _clean_sql_val(frame.get("ICAO_Hex")),
            _clean_sql_val(frame.get("DF")),
            _clean_sql_val(frame.get("TC")),
            _clean_sql_val(frame.get("Call_sign")),
            _clean_sql_val(frame.get("Altitude")),
            _clean_sql_val(frame.get("Latitude")),
            _clean_sql_val(frame.get("Longitude")),
            _clean_sql_val(frame.get("Even_Odd")),
            _clean_sql_val(frame.get("Distance_from_Rx")),
            _clean_sql_val(frame.get("Airspeed")),
            _clean_sql_val(frame.get("Heading")),
            _clean_sql_val(frame.get("Vertical")),
            _clean_sql_val(frame.get("Power")),
            _clean_sql_val(frame.get("Amplitude")),
            _clean_sql_val(frame.get("EIRP_dBm")),
            _clean_sql_val(frame.get("Doppler")),
            _clean_sql_val(frame.get("CFO_Raw_Hz")),
            _clean_sql_val(frame.get("Frequency_shift")),
            _clean_sql_val(frame.get("Timing_Offset_PPM")),
            _clean_sql_val(frame.get("Sub_Sample_Lag_Ns")),
            _clean_sql_val(frame.get("Phase_Noise_Std")),
            _clean_sql_val(frame.get("Inter_Arrival_s")),
            bool(frame.get("CPR_Zone_Valid", True)),
            _clean_sql_val(frame.get("Pos_Jump_km")),
            _clean_sql_val(frame.get("Implied_Groundspeed_kt")),
            _clean_sql_val(frame.get("Implied_Alt_Rate_fpm")),
            _clean_sql_val(frame.get("Predicted_ICAO_Hex")),
            _clean_sql_val(frame.get("RF_Match_Prob")),
            _clean_sql_val(frame.get("Anomaly_Score")),
            bool(frame.get("Is_Anomaly", False)),
            _clean_sql_val(frame.get("Anomaly_Reasons", "")),
        )

        self.conn.cursor().execute(
            """
            INSERT INTO adsb_frames VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            row,
        )
        self.conn.commit()
        return frame_id

    def insert_crc_failed_bursts(
        self, failed_bursts: List[Dict[str, Any]], measurement_id: int = 1
    ) -> None:
        """Persists CRC-failed bursts and their blind RF Fingerprinting aircraft attributions."""
        cur = self.conn.cursor()
        for b in failed_bursts:
            burst_id = self._next_burst_id
            self._next_burst_id += 1
            b["burst_id"] = burst_id
            cur.execute(
                """
                INSERT INTO crc_failed_frames VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
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
                ),
            )
        self.conn.commit()
        self.refresh_aircraft_profiles()

    def update_ml_predictions(self, frames: List[Dict[str, Any]]) -> None:
        """Updates ML RF fingerprinting classifications and anomaly flags in `adsb_frames`."""
        cur = self.conn.cursor()
        for f in frames:
            if "frame_id" not in f:
                continue
            cur.execute(
                """
                UPDATE adsb_frames
                SET predicted_icao_hex = ?,
                    rf_match_prob = ?,
                    anomaly_score = ?,
                    is_anomaly = ?,
                    anomaly_reasons = ?
                WHERE frame_id = ?
                """,
                (
                    _clean_sql_val(f.get("Predicted_ICAO_Hex")),
                    _clean_sql_val(f.get("RF_Match_Prob")),
                    _clean_sql_val(f.get("Anomaly_Score")),
                    bool(f.get("Is_Anomaly", False)),
                    _clean_sql_val(f.get("Anomaly_Reasons", "")),
                    int(f["frame_id"]),
                ),
            )
        self.conn.commit()
        self.refresh_aircraft_profiles()

    def refresh_aircraft_profiles(self) -> pd.DataFrame:
        """
        Recomputes per-aircraft RF fingerprint summary statistics in SQL
        (mean/std of Carrier Frequency Offset, Timing Offset, EIRP, anomaly counts, and
        count of CRC-failed bursts attributed to each aircraft).
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM aircraft_rf_profiles;")

        df_frames = self.query_df("SELECT * FROM adsb_frames ORDER BY timestamp ASC")
        if df_frames.empty:
            return pd.DataFrame()

        df_crc_failed = self.query_df("SELECT * FROM crc_failed_frames")

        for icao_hex, grp in df_frames.groupby("icao_hex"):
            clean_grp = grp[~grp["is_anomaly"].astype(bool)] if (~grp["is_anomaly"].astype(bool)).any() else grp
            callsigns = grp["callsign"].dropna()
            callsign = str(callsigns.iloc[-1]) if not callsigns.empty else None

            crc_rec_count = 0
            if not df_crc_failed.empty:
                crc_rec_count = int((df_crc_failed["predicted_icao_hex"] == str(icao_hex)).sum())

            row = (
                str(icao_hex),
                callsign,
                int(len(grp)),
                crc_rec_count,
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
            cur.execute(
                """
                INSERT INTO aircraft_rf_profiles VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                row,
            )

        self.conn.commit()
        return self.query_df("SELECT * FROM aircraft_rf_profiles ORDER BY frame_count DESC")

    def query_df(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Executes an analytical SQL query and returns the result as a Pandas DataFrame."""
        cur = self.conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def export_csv(self, filename: str = "adsb_table.csv") -> pd.DataFrame:
        """Exports the full `adsb_frames` table to CSV (replacing legacy `create_table()`)."""
        df = self.query_df("SELECT * FROM adsb_frames ORDER BY timestamp ASC")
        df.to_csv(filename, index=False)
        return df

    def close(self) -> None:
        self.conn.close()