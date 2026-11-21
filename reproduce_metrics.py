"""
Reproducible Benchmark, Class Census & Ablation Runner (`reproduce_metrics.py`).

Loads the exported 2,809-frame TU Wien rooftop telemetry table (`adsb_table.csv`) and prints:
  1. Reconciliation Table (`format_reconciliation_table`): In-Sample Clean Fit (100.0%) vs.
     Exported CSV Match (86.54%) vs. Leak-Free OOS Accuracies across both class tiers
     (>= 5 clean frames = 115 classes; >= 12 clean frames = 72 classes, ~70 classes).
  2. Feature Group Ablation Study (`CFO Only` vs. `Hardware/Clock Only` vs. `Full 12-Feature`).
  3. Ensemble Member Ablation Study (`Mahalanobis` vs. `RandomForest` vs. `ExtraTrees` vs. `Ensemble`).
  4. Out-of-Fold Backtest Residual Quantiles used for simulator uncertainty intervals.

Usage:
    python reproduce_metrics.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from adsb_rf.evaluation import (
    compute_reconciliation_and_headline_metrics,
    format_reconciliation_table,
    run_ablation_suite,
)


def main(csv_path: str | Path = ROOT_DIR / "adsb_table.csv") -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry CSV not found at {path}. Run `python main.py` first to generate `adsb_table.csv`."
        )

    df = pd.read_csv(path)
    m = compute_reconciliation_and_headline_metrics(df)
    t5, res = m["tier5"], m["residual_uncertainty"]

    print("=" * 94)
    print("1090 MHz ADS-B PHYSICAL-LAYER RF FINGERPRINTING — REPRODUCIBLE BENCHMARK & ABLATION REPORT")
    print("=" * 94)
    print(f"Source Telemetry File : {path.name} (TU Wien Rooftop Keysight UXA N9040B Capture, 20.97 s @ 12.8 MS/s)")
    print(f"Total CRC-Valid Frames: {m['total_frames']:,} frames across {m['total_icaos']} distinct ICAO addresses")
    print(f"Clean Inlier Frames   : {m['clean_frames']:,} frames across {m['clean_icaos']} distinct ICAO addresses (is_anomaly == 0)")
    print(f"Flagged Anomalies     : {m['anom_frames']:,} frames (Rule_Anomaly or Mahalanobis d_M > 4.5 sigma)")
    print("-" * 94)

    print("\n1. RECONCILIATION: IN-SAMPLE CLEAN FIT (100.0%) vs. EXPORTED CSV MATCH (86.5%) vs. LEAK-FREE OOS")
    print("-" * 94)
    print(format_reconciliation_table(m).to_string(index=False))

    abl5 = run_ablation_suite(df, min_frames=5, precomputed_full_res=t5)
    print(f"\n2. FEATURE GROUP ABLATION: CFO vs. HARDWARE TRANSIENT & CLOCK FEATURES ({t5['n_classes']} Classes, {t5['n_frames']:,} Frames)")
    print("-" * 94)
    print(abl5["feature_ablation_df"].to_string(index=False))

    print(f"\n3. ENSEMBLE MEMBER ABLATION: MAHALANOBIS vs. RANDOM FOREST vs. EXTRA TREES ({t5['n_classes']} Classes, {t5['n_frames']:,} Frames)")
    print("-" * 94)
    print(abl5["ensemble_ablation_df"].to_string(index=False))

    print("\n4. EMPIRICAL OUT-OF-FOLD BACKTEST RESIDUAL QUANTILES (USED FOR SIMULATOR UNCERTAINTY BANDS)")
    print("-" * 94)
    print(
        f"  * Out-of-Fold Mahalanobis Distance (d_M) : Median = {res['mahal_median']:.2f} sigma | "
        f"90th Pct = {res['mahal_p90']:.2f} sigma | 95th Pct = {res['mahal_p95']:.2f} sigma"
    )
    print(f"  * Out-of-Fold |CFO Error| 90th Pct       : +/- {res['cfo_err_p90_khz']:.2f} kHz")
    print(f"  * Out-of-Fold |Timing Drift Error| 90th  : +/- {res['ppm_err_p90']:.1f} ppm")
    print("=" * 94)


if __name__ == "__main__":
    main()
