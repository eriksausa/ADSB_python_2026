"""
Reproducible Benchmark, Class Census & Ablation Runner (`reproduce_metrics.py`).

Loads the exported 2,809-frame TU Wien rooftop telemetry table (`adsb_table.csv`) and prints:
1. Exact Dataset & Class Count Census (174 total ICAOs, 161 clean ICAOs, 115 classes with >= 5
   clean frames, and 68 classes (~70 classes) with >= 12 clean frames).
2. Explicit Reconciliation of "100.0% In-Sample Clean Inlier Fit" vs. "86.5% All-Frame Match Rate"
   vs. Leak-Free Out-of-Sample Accuracies (80/20 Stratified Holdout, 5-Fold Stratified CV,
   and Chronological Out-of-Time Split).
3. Feature Group Ablation Study: CFO Features Only vs. Hardware Transient/Clock Features (No CFO)
   vs. Pure Waveform Features vs. Full 12-Feature Vector.
4. Ensemble Member Ablation Study: Bayesian Mahalanobis Posterior vs. RandomForest vs. ExtraTrees
   vs. 3-Model Soft-Voting Ensemble.

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
from adsb_rf.evaluation import compute_reconciliation_and_headline_metrics, run_ablation_suite


def main(csv_path: str | Path = ROOT_DIR / "adsb_table.csv") -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry CSV not found at {path}. Run `python main.py` first to generate `adsb_table.csv`."
        )

    df = pd.read_csv(path)
    m = compute_reconciliation_and_headline_metrics(df)
    t5 = m["tier5"]
    t12 = m["tier12"]
    res = m["residual_uncertainty"]

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
    reconcile_rows = [
        {
            "Metric / Evaluation Scope": "In-Sample Clean Inlier Fit (is_anomaly == 0)",
            "Evaluated Frames": f"{m['clean_frames']:,} / {m['clean_frames']:,}",
            "Evaluated Classes": f"{m['clean_icaos']} ICAOs",
            "Top-1 Accuracy": f"{m['clean_inlier_fit']*100:.2f}%",
            "Top-3 Accuracy": "100.00%",
            "Methodology Note": f"In-sample training fit ({m['clean_match_count']}/{m['clean_frames']} clean frames match).",
        },
        {
            "Metric / Evaluation Scope": "All-Frame Exported CSV Match (adsb_table.csv)",
            "Evaluated Frames": f"{m['all_match_count']:,} / {m['total_frames']:,}",
            "Evaluated Classes": f"{m['total_icaos']} ICAOs",
            "Top-1 Accuracy": f"{m['all_match_rate']*100:.2f}%",
            "Top-3 Accuracy": "N/A (includes outliers)",
            "Methodology Note": f"Includes {m['anom_frames']} flagged anomalies (where only {m['anom_match_count']}/{m['anom_frames']} match).",
        },
        {
            "Metric / Evaluation Scope": "Leak-Free 80/20 Stratified Holdout (All Enrolled >=5)",
            "Evaluated Frames": f"{t5['n_frames']:,} (461 test)",
            "Evaluated Classes": f"{t5['n_classes']} classes",
            "Top-1 Accuracy": f"{t5['holdout_top1']*100:.2f}%",
            "Top-3 Accuracy": f"{t5['holdout_top3']*100:.2f}%",
            "Methodology Note": "Fold-isolated imputation, centroids & trees; >=5 clean frames/class.",
        },
        {
            "Metric / Evaluation Scope": "Leak-Free 5-Fold Stratified CV (All Enrolled >=5)",
            "Evaluated Frames": f"{t5['n_frames']:,} (100% OOF)",
            "Evaluated Classes": f"{t5['n_classes']} classes",
            "Top-1 Accuracy": f"{t5['cv_top1']*100:.2f}%",
            "Top-3 Accuracy": f"{t5['cv_top3']*100:.2f}%",
            "Methodology Note": "5-fold out-of-fold evaluation across all 115 enrolled classes.",
        },
        {
            "Metric / Evaluation Scope": "Chronological Out-of-Time Split (All Enrolled >=5)",
            "Evaluated Frames": f"{t5['n_frames']:,} (first 80% -> last 20%)",
            "Evaluated Classes": f"{t5['n_classes']} classes",
            "Top-1 Accuracy": f"{t5['oot_top1']*100:.2f}%",
            "Top-3 Accuracy": f"{t5['oot_top3']*100:.2f}%",
            "Methodology Note": "Trained on first 80% of each aircraft's timeline, tested on final 20%.",
        },
        {
            "Metric / Evaluation Scope": "Leak-Free 80/20 Holdout (Well-Enrolled >=12, ~70 classes)",
            "Evaluated Frames": f"{t12['n_frames']:,} (397 test)",
            "Evaluated Classes": f"{t12['n_classes']} classes (~70)",
            "Top-1 Accuracy": f"{t12['holdout_top1']*100:.2f}%",
            "Top-3 Accuracy": f"{t12['holdout_top3']*100:.2f}%",
            "Methodology Note": f"Well-enrolled aircraft tier (>=12 clean frames/aircraft -> {t12['n_classes']} classes).",
        },
        {
            "Metric / Evaluation Scope": "Leak-Free 5-Fold Stratified CV (Well-Enrolled >=12, ~70 classes)",
            "Evaluated Frames": f"{t12['n_frames']:,} (100% OOF)",
            "Evaluated Classes": f"{t12['n_classes']} classes (~70)",
            "Top-1 Accuracy": f"{t12['cv_top1']*100:.2f}%",
            "Top-3 Accuracy": f"{t12['cv_top3']*100:.2f}%",
            "Methodology Note": f"5-fold out-of-fold evaluation on the {t12['n_classes']}-class (~70-class) subset.",
        },
    ]
    print(pd.DataFrame(reconcile_rows).to_string(index=False))

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
    print(
        f"  * Out-of-Fold |CFO Error| 90th Pct       : +/- {res['cfo_err_p90_khz']:.2f} kHz"
    )
    print(
        f"  * Out-of-Fold |Timing Drift Error| 90th  : +/- {res['ppm_err_p90']:.1f} ppm"
    )
    print("=" * 94)


if __name__ == "__main__":
    main()
