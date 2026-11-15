"""
Leak-Free Backtesting, Out-of-Time Evaluation, Residual Uncertainty & Ablation Engine
(`src/adsb_rf/evaluation.py`).

Provides:
1. Canonical aircraft profile builder (`build_aircraft_profiles_from_df`)
2. Leak-free Stratified 80/20 Holdout, 5-Fold Stratified CV, and Chronological Out-of-Time evaluation
3. Empirical out-of-fold backtest residual quantiles for uncertainty error bars
4. Feature Group Ablation (CFO vs. Hardware Transients/Clock vs. Full 12-Feature Vector)
5. Ensemble Member Ablation (Mahalanobis vs. RandomForest vs. ExtraTrees vs. Hybrid Ensemble)
6. Reconciliation of in-sample clean inlier fit (100.0%) vs. exported CSV match rate (86.5%)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .assumptions import DEFAULT_ASSUMPTIONS
from .features import (
    FEATURE_SUBSETS,
    RF_FEATURE_NAMES,
    RF_MAHALANOBIS_WEIGHTS,
    RF_PHYSICAL_STD_FLOOR,
    extract_feature_matrix_from_df,
)
from .models import HAS_SKLEARN, fit_fold_ensemble, predict_fold_ensemble

if HAS_SKLEARN:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def build_aircraft_profiles_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-aircraft physical-layer RF fingerprint centroids, robust standard deviations,
    and empirical 5th-95th percentile intervals from `adsb_table.csv`.
    """
    profiles: List[Dict[str, Any]] = []
    floor_cfo_khz = DEFAULT_ASSUMPTIONS.rf_physical_std_floor[1] / 1e3
    floor_ppm = DEFAULT_ASSUMPTIONS.rf_physical_std_floor[2]
    floor_eirp = DEFAULT_ASSUMPTIONS.rf_physical_std_floor[5]

    for icao_hex, grp in df.groupby("icao_hex"):
        clean_grp = (
            grp[~grp["is_anomaly"].astype(bool)]
            if "is_anomaly" in grp.columns and (~grp["is_anomaly"].astype(bool)).any()
            else grp
        )
        cs_series = grp["callsign"].dropna() if "callsign" in grp.columns else pd.Series(dtype=str)
        cs_valid = [str(c) for c in cs_series if str(c).strip() not in ("", "None", "N/A", "nan")]
        callsign = cs_valid[-1] if cs_valid else "N/A"

        cfo_col = clean_grp["freq_offset_hz"].fillna(clean_grp["cfo_raw_hz"]) if "freq_offset_hz" in clean_grp.columns else pd.Series(dtype=float)
        cfo_vals = cfo_col.dropna() / 1e3
        ppm_vals = clean_grp["timing_offset_ppm"].dropna() if "timing_offset_ppm" in clean_grp.columns else pd.Series(dtype=float)
        eirp_vals = clean_grp["eirp_dbm"].dropna() if "eirp_dbm" in clean_grp.columns else pd.Series(dtype=float)
        pnoise_vals = clean_grp["phase_noise_std"].dropna() if "phase_noise_std" in clean_grp.columns else pd.Series(dtype=float)

        med_cfo = float(cfo_vals.median()) if len(cfo_vals) > 0 else 0.0
        mad_cfo = float(np.median(np.abs(cfo_vals - med_cfo)) * 1.4826) if len(cfo_vals) > 1 else floor_cfo_khz
        std_cfo = max(float(cfo_vals.std(ddof=0)) if len(cfo_vals) > 1 else floor_cfo_khz, mad_cfo, floor_cfo_khz)

        med_ppm = float(ppm_vals.median()) if len(ppm_vals) > 0 else 0.0
        mad_ppm = float(np.median(np.abs(ppm_vals - med_ppm)) * 1.4826) if len(ppm_vals) > 1 else floor_ppm
        std_ppm = max(float(ppm_vals.std(ddof=0)) if len(ppm_vals) > 1 else floor_ppm, mad_ppm, floor_ppm)

        med_eirp = float(eirp_vals.median()) if len(eirp_vals) > 0 else 55.0
        mad_eirp = float(np.median(np.abs(eirp_vals - med_eirp)) * 1.4826) if len(eirp_vals) > 1 else floor_eirp
        std_eirp = max(float(eirp_vals.std(ddof=0)) if len(eirp_vals) > 1 else floor_eirp, mad_eirp, floor_eirp)

        profiles.append(
            {
                "icao_hex": str(icao_hex),
                "callsign": callsign,
                "frame_count": int(len(grp)),
                "clean_frame_count": int(len(clean_grp)),
                "mean_cfo_khz": round(med_cfo, 3),
                "std_cfo_khz": round(std_cfo, 3),
                "cfo_iqr_90_khz": round(
                    float(np.percentile(cfo_vals, 95) - np.percentile(cfo_vals, 5)) / 2.0
                    if len(cfo_vals) >= 5
                    else 1.645 * std_cfo,
                    3,
                ),
                "mean_timing_ppm": round(med_ppm, 2),
                "std_timing_ppm": round(std_ppm, 2),
                "ppm_iqr_90": round(
                    float(np.percentile(ppm_vals, 95) - np.percentile(ppm_vals, 5)) / 2.0
                    if len(ppm_vals) >= 5
                    else 1.645 * std_ppm,
                    2,
                ),
                "mean_eirp_dbm": round(med_eirp, 2),
                "std_eirp_dbm": round(std_eirp, 2),
                "mean_phase_noise_rad": round(float(pnoise_vals.median()), 3) if len(pnoise_vals) > 0 else 0.30,
            }
        )

    prof_df = pd.DataFrame(profiles)
    if not prof_df.empty:
        prof_df = prof_df.sort_values("frame_count", ascending=False).reset_index(drop=True)
    return prof_df


def _prepare_clean_eval_subset(
    df: pd.DataFrame,
    min_frames: int = DEFAULT_ASSUMPTIONS.min_frames_enrolled,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Filters `df` to clean inlier frames (`is_anomaly == 0`) belonging to aircraft with at least
    `min_frames` clean frames, returning `(sub_df, X_raw, y_idx, classes)`.
    """
    clean_df = (
        df[~df["is_anomaly"].astype(bool)].copy()
        if "is_anomaly" in df.columns
        else df.copy()
    )
    clean_df = clean_df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    counts = clean_df["icao_hex"].value_counts()
    eligible_icaos = sorted(counts[counts >= min_frames].index.astype(str).tolist())

    sub_df = clean_df[clean_df["icao_hex"].astype(str).isin(eligible_icaos)].copy().reset_index(drop=True)
    X_raw = extract_feature_matrix_from_df(sub_df)
    class_to_idx = {c: i for i, c in enumerate(eligible_icaos)}
    y_idx = np.array([class_to_idx[str(v)] for v in sub_df["icao_hex"]], dtype=np.int64)
    return sub_df, X_raw, y_idx, eligible_icaos


def evaluate_subset_metrics(
    X_raw: np.ndarray,
    y_idx: np.ndarray,
    timestamps: np.ndarray,
    n_classes: int,
    feature_indices: Optional[Sequence[int]] = None,
    ensemble_mode: str = "ensemble",
) -> Dict[str, Any]:
    """
    Evaluates a specific feature subset and ensemble configuration across:
      1. 80/20 Stratified Holdout Split
      2. 5-Fold Stratified Cross-Validation (with Out-of-Fold Top-1, Top-3, and Mahalanobis residuals)
      3. Chronological Out-of-Time Split (first 80% of timestamps per aircraft -> train, final 20% -> test)
    """
    if len(y_idx) < 10 or n_classes < 2:
        return {
            "holdout_top1": 1.0,
            "holdout_top3": 1.0,
            "cv_top1": 1.0,
            "cv_top3": 1.0,
            "oot_top1": 1.0,
            "oot_top3": 1.0,
            "oof_mahal_residuals": np.zeros(len(y_idx)),
            "oof_cfo_abs_err_khz": np.zeros(len(y_idx)),
            "oof_ppm_abs_err": np.zeros(len(y_idx)),
        }

    # --- 1. 80/20 Stratified Holdout ---
    if HAS_SKLEARN:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_80, te_20 = next(sss.split(X_raw, y_idx))
    else:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(y_idx))
        cut = int(0.8 * len(y_idx))
        tr_80, te_20 = perm[:cut], perm[cut:]

    b_80 = fit_fold_ensemble(
        X_raw[tr_80], y_idx[tr_80], n_classes=n_classes, feature_indices=feature_indices, ensemble_mode=ensemble_mode
    )
    probs_20, _ = predict_fold_ensemble(b_80, X_raw[te_20])
    preds_20 = np.argmax(probs_20, axis=1)
    top3_20 = np.argsort(probs_20, axis=1)[:, -min(3, n_classes) :]
    holdout_top1 = float(np.mean(preds_20 == y_idx[te_20]))
    holdout_top3 = float(
        np.mean([int(y_idx[te_20][k]) in top3_20[k] for k in range(len(te_20))])
    )

    member_modes = ("mahalanobis", "rf", "et", "ensemble") if ensemble_mode == "ensemble" else ()
    member_holdout_top1: Dict[str, float] = {}
    for mm in member_modes:
        p_m, _ = predict_fold_ensemble(b_80, X_raw[te_20], override_mode=mm)
        member_holdout_top1[mm] = float(np.mean(np.argmax(p_m, axis=1) == y_idx[te_20]))

    # --- 2. 5-Fold Stratified Cross-Validation ---
    if HAS_SKLEARN:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X_raw, y_idx))
    else:
        rng_cv = np.random.default_rng(42)
        perm = rng_cv.permutation(len(y_idx))
        folds = [(perm[np.arange(len(y_idx)) % 5 != k], perm[np.arange(len(y_idx)) % 5 == k]) for k in range(5)]

    oof_top1_hits = 0
    oof_top3_hits = 0
    oof_total = 0
    oof_mahal = np.zeros(len(y_idx), dtype=np.float64)
    oof_cfo_err = np.zeros(len(y_idx), dtype=np.float64)
    oof_ppm_err = np.zeros(len(y_idx), dtype=np.float64)
    member_cv_top1_hits: Dict[str, int] = {mm: 0 for mm in member_modes}
    member_cv_top3_hits: Dict[str, int] = {mm: 0 for mm in member_modes}

    for tr_i, va_i in folds:
        fb = fit_fold_ensemble(
            X_raw[tr_i], y_idx[tr_i], n_classes=n_classes, feature_indices=feature_indices, ensemble_mode=ensemble_mode
        )
        f_probs, f_dists = predict_fold_ensemble(fb, X_raw[va_i])
        f_preds = np.argmax(f_probs, axis=1)
        f_top3 = np.argsort(f_probs, axis=1)[:, -min(3, n_classes) :]

        oof_top1_hits += int(np.sum(f_preds == y_idx[va_i]))
        for k, global_row in enumerate(va_i):
            true_c = int(y_idx[global_row])
            if true_c in f_top3[k]:
                oof_top3_hits += 1
            oof_mahal[global_row] = float(f_dists[k, true_c])
            # Record physical CFO and Timing residuals relative to training-fold centroid
            if 1 in fb["feat_idx"]:
                pos1 = fb["feat_idx"].index(1)
                val1 = X_raw[global_row, 1]
                if not np.isnan(val1):
                    oof_cfo_err[global_row] = abs(val1 - fb["centroid_mat"][true_c, pos1]) / 1e3
            if 2 in fb["feat_idx"]:
                pos2 = fb["feat_idx"].index(2)
                val2 = X_raw[global_row, 2]
                if not np.isnan(val2):
                    oof_ppm_err[global_row] = abs(val2 - fb["centroid_mat"][true_c, pos2])

        for mm in member_modes:
            pm, _ = predict_fold_ensemble(fb, X_raw[va_i], override_mode=mm)
            member_cv_top1_hits[mm] += int(np.sum(np.argmax(pm, axis=1) == y_idx[va_i]))
            top3_m = np.argsort(pm, axis=1)[:, -min(3, n_classes) :]
            member_cv_top3_hits[mm] += int(
                sum(int(y_idx[va_i][k]) in top3_m[k] for k in range(len(va_i)))
            )

        oof_total += len(va_i)

    cv_top1 = float(oof_top1_hits / max(oof_total, 1))
    cv_top3 = float(oof_top3_hits / max(oof_total, 1))

    # --- 3. Chronological Out-of-Time Split (First 80% time vs. Final 20% time per aircraft) ---
    oot_tr_list: List[int] = []
    oot_te_list: List[int] = []
    order_ts = np.argsort(timestamps, kind="mergesort")
    for c_id in range(n_classes):
        c_rows = order_ts[y_idx[order_ts] == c_id]
        if len(c_rows) < 2:
            oot_tr_list.extend(c_rows.tolist())
            continue
        n_te = max(1, int(round(0.20 * len(c_rows))))
        oot_tr_list.extend(c_rows[:-n_te].tolist())
        oot_te_list.extend(c_rows[-n_te:].tolist())

    oot_tr = np.array(oot_tr_list, dtype=np.int64)
    oot_te = np.array(oot_te_list, dtype=np.int64)
    b_oot = fit_fold_ensemble(
        X_raw[oot_tr], y_idx[oot_tr], n_classes=n_classes, feature_indices=feature_indices, ensemble_mode=ensemble_mode
    )
    probs_oot, _ = predict_fold_ensemble(b_oot, X_raw[oot_te])
    preds_oot = np.argmax(probs_oot, axis=1)
    top3_oot = np.argsort(probs_oot, axis=1)[:, -min(3, n_classes) :]
    oot_top1 = float(np.mean(preds_oot == y_idx[oot_te]))
    oot_top3 = float(
        np.mean([int(y_idx[oot_te][k]) in top3_oot[k] for k in range(len(oot_te))])
    )

    member_metrics: Dict[str, Dict[str, float]] = {}
    for mm in member_modes:
        p_oot_m, _ = predict_fold_ensemble(b_oot, X_raw[oot_te], override_mode=mm)
        member_metrics[mm] = {
            "holdout_top1": member_holdout_top1[mm],
            "cv_top1": float(member_cv_top1_hits[mm] / max(oof_total, 1)),
            "oot_top1": float(np.mean(np.argmax(p_oot_m, axis=1) == y_idx[oot_te])),
            "cv_top3": float(member_cv_top3_hits[mm] / max(oof_total, 1)),
        }

    return {
        "holdout_top1": holdout_top1,
        "holdout_top3": holdout_top3,
        "cv_top1": cv_top1,
        "cv_top3": cv_top3,
        "oot_top1": oot_top1,
        "oot_top3": oot_top3,
        "oof_mahal_residuals": oof_mahal,
        "oof_cfo_abs_err_khz": oof_cfo_err,
        "oof_ppm_abs_err": oof_ppm_err,
        "member_metrics": member_metrics,
    }


def compute_reconciliation_and_headline_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes the complete dataset census, reconciliation between in-sample fit (100.0%),
    exported table match rate (86.5%), and leak-free out-of-sample evaluation across both
    class eligibility tiers (`>= 5` frames = 115 classes; `>= 12` frames = 68 classes / ~70 classes).
    """
    total_frames = len(df)
    total_icaos = int(df["icao_hex"].nunique())

    clean_mask = ~df["is_anomaly"].astype(bool) if "is_anomaly" in df.columns else np.ones(len(df), dtype=bool)
    clean_df = df[clean_mask]
    anom_df = df[~clean_mask]

    clean_frames = len(clean_df)
    clean_icaos = int(clean_df["icao_hex"].nunique())
    anom_frames = len(anom_df)

    all_match_count = int((df["predicted_icao_hex"] == df["icao_hex"]).sum())
    clean_match_count = int((clean_df["predicted_icao_hex"] == clean_df["icao_hex"]).sum())
    anom_match_count = int((anom_df["predicted_icao_hex"] == anom_df["icao_hex"]).sum())

    all_match_rate = float(all_match_count / max(total_frames, 1))
    clean_inlier_fit = float(clean_match_count / max(clean_frames, 1))
    anom_match_rate = float(anom_match_count / max(anom_frames, 1))

    # Tier 1: All Enrolled Aircraft (>= 5 clean frames)
    sub5_df, X5, y5, cls5 = _prepare_clean_eval_subset(df, min_frames=DEFAULT_ASSUMPTIONS.min_frames_enrolled)
    ts5 = sub5_df["timestamp"].to_numpy(dtype=np.float64)
    res5 = evaluate_subset_metrics(X5, y5, ts5, n_classes=len(cls5))

    # Tier 2: Well-Enrolled Aircraft (>= 12 clean frames, ~70 classes)
    sub12_df, X12, y12, cls12 = _prepare_clean_eval_subset(df, min_frames=DEFAULT_ASSUMPTIONS.min_frames_well_enrolled)
    ts12 = sub12_df["timestamp"].to_numpy(dtype=np.float64)
    res12 = evaluate_subset_metrics(X12, y12, ts12, n_classes=len(cls12))

    oof_m = res5["oof_mahal_residuals"]
    oof_cfo = res5["oof_cfo_abs_err_khz"]
    oof_ppm = res5["oof_ppm_abs_err"]

    return {
        "total_frames": total_frames,
        "total_icaos": total_icaos,
        "clean_frames": clean_frames,
        "clean_icaos": clean_icaos,
        "anom_frames": anom_frames,
        "all_match_count": all_match_count,
        "all_match_rate": all_match_rate,
        "clean_match_count": clean_match_count,
        "clean_inlier_fit": clean_inlier_fit,
        "anom_match_count": anom_match_count,
        "anom_match_rate": anom_match_rate,
        "tier5": {
            "min_frames": DEFAULT_ASSUMPTIONS.min_frames_enrolled,
            "n_classes": len(cls5),
            "n_frames": len(sub5_df),
            **res5,
        },
        "tier12": {
            "min_frames": DEFAULT_ASSUMPTIONS.min_frames_well_enrolled,
            "n_classes": len(cls12),
            "n_frames": len(sub12_df),
            **res12,
        },
        "residual_uncertainty": {
            "mahal_median": float(np.median(oof_m)) if len(oof_m) > 0 else 0.75,
            "mahal_p90": float(np.percentile(oof_m, 90)) if len(oof_m) > 0 else 1.65,
            "mahal_p95": float(np.percentile(oof_m, 95)) if len(oof_m) > 0 else 2.15,
            "cfo_err_p90_khz": float(np.percentile(oof_cfo, 90)) if len(oof_cfo) > 0 else 0.85,
            "ppm_err_p90": float(np.percentile(oof_ppm, 90)) if len(oof_ppm) > 0 else 65.0,
        },
    }


def run_ablation_suite(
    df: pd.DataFrame,
    min_frames: int = DEFAULT_ASSUMPTIONS.min_frames_enrolled,
    precomputed_full_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Runs the complete Feature Group Ablation (CFO vs. Hardware features) and Ensemble Member
    Ablation on `df` at `min_frames` threshold.
    """
    sub_df, X_raw, y_idx, classes = _prepare_clean_eval_subset(df, min_frames=min_frames)
    ts = sub_df["timestamp"].to_numpy(dtype=np.float64)
    n_classes = len(classes)

    feature_configs = [
        (
            "1. CFO Features Only (CFO_Raw_Hz + Frequency_shift)",
            FEATURE_SUBSETS["cfo_only"],
            "Carrier frequency offset + Doppler-compensated CFO only (2 features)",
        ),
        (
            "2. Hardware & Clock Features Only (No CFO, No Power)",
            FEATURE_SUBSETS["hardware_no_cfo"],
            "Timing_PPM, Lag_Ns, Phase_Noise, PA_Droop, Crest, Curvature, SNR, Preamble (8 features)",
        ),
        (
            "3. Hardware + Channel Power/EIRP (No CFO)",
            FEATURE_SUBSETS["hardware_plus_power_no_cfo"],
            "8 hardware/clock features + Power_dBm + EIRP_dBm (10 features, zero CFO)",
        ),
        (
            "4. Pure Waveform Features (CFO + Hardware, No Power/EIRP)",
            FEATURE_SUBSETS["waveform_10_no_power"],
            "CFO + 8 hardware/clock features without Rx Power or FSPL EIRP (10 features)",
        ),
        (
            "5. Full 12-Feature Physical-Layer Vector (Production)",
            FEATURE_SUBSETS["all_12_features"],
            "All 12 features: CFO + 8 Hardware/Clock + Power/EIRP",
        ),
    ]

    feat_rows = []
    full_res: Optional[Dict[str, Any]] = precomputed_full_res
    for label, f_idx, desc in feature_configs:
        if len(f_idx) == len(RF_FEATURE_NAMES) and full_res is not None:
            m = full_res
        else:
            m = evaluate_subset_metrics(
                X_raw, y_idx, ts, n_classes=n_classes, feature_indices=f_idx, ensemble_mode="ensemble"
            )
            if len(f_idx) == len(RF_FEATURE_NAMES):
                full_res = m
        feat_rows.append(
            {
                "Feature Set": label,
                "Num Features": len(f_idx),
                "80/20 Holdout Acc (%)": round(m["holdout_top1"] * 100.0, 2),
                "5-Fold CV Acc (%)": round(m["cv_top1"] * 100.0, 2),
                "Out-of-Time Acc (%)": round(m["oot_top1"] * 100.0, 2),
                "5-Fold Top-3 Acc (%)": round(m["cv_top3"] * 100.0, 2),
                "Included Signals": desc,
            }
        )

    ensemble_configs = [
        (
            "1. Bayesian Mahalanobis Posterior Only (20% weight member)",
            "mahalanobis",
            "Parametric diagonal Gaussian likelihood around Median/MAD hardware centroid",
        ),
        (
            "2. RandomForest Classifier Only (45% weight member)",
            "rf",
            "100-tree bootstrap RandomForest (max_depth=18, damped sample weights)",
        ),
        (
            "3. ExtraTrees Classifier Only (35% weight member)",
            "et",
            "100-tree randomized split ExtraTrees (max_depth=20, lower variance)",
        ),
        (
            "4. Hybrid 3-Model Soft-Voting Ensemble (Production)",
            "ensemble",
            "45% RandomForest + 35% ExtraTrees + 20% Bayesian Mahalanobis Posterior",
        ),
    ]

    ens_rows = []
    member_dict = full_res.get("member_metrics", {}) if full_res is not None else {}
    for label, mode_str, desc in ensemble_configs:
        if mode_str in member_dict:
            m = member_dict[mode_str]
        else:
            m = evaluate_subset_metrics(
                X_raw,
                y_idx,
                ts,
                n_classes=n_classes,
                feature_indices=FEATURE_SUBSETS["all_12_features"],
                ensemble_mode=mode_str,
            )
        ens_rows.append(
            {
                "Ensemble Member / Model": label,
                "80/20 Holdout Acc (%)": round(m["holdout_top1"] * 100.0, 2),
                "5-Fold CV Acc (%)": round(m["cv_top1"] * 100.0, 2),
                "Out-of-Time Acc (%)": round(m["oot_top1"] * 100.0, 2),
                "5-Fold Top-3 Acc (%)": round(m["cv_top3"] * 100.0, 2),
                "Architecture Role": desc,
            }
        )

    return {
        "feature_ablation_df": pd.DataFrame(feat_rows),
        "ensemble_ablation_df": pd.DataFrame(ens_rows),
    }


def score_simulated_burst_with_residuals(
    prof_df: pd.DataFrame,
    target_icao: str,
    sim_cfo_khz: float,
    sim_ppm: float,
    sim_eirp: float,
    outlier_z: float,
    residual_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a simulated ADS-B burst against the enrolled aircraft profiles in `prof_df`
    using the `Assumptions` Mahalanobis weights and empirical out-of-fold backtest residual
    quantiles (`mahal_p90`, `mahal_p95`, `cfo_err_p90_khz`, `ppm_err_p90`).
    """
    if prof_df.empty or target_icao not in prof_df["icao_hex"].values:
        return {
            "d_claimed": 0.0,
            "d_ci_low": 0.0,
            "d_ci_high": 0.0,
            "claimed_prob": 1.0,
            "pred_icao": target_icao,
            "pred_cs": "N/A",
            "is_spoof": False,
            "mahal_p95_backtest": 2.15,
        }

    # Normalize the 3 interactive slider dimensions using the canonical Mahalanobis weights
    # (CFO_Raw_Hz + Frequency_shift = 0.45, Timing_Offset_PPM = 0.28, Power + EIRP = 0.17)
    w_cfo_raw = DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights[0] + DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights[1]
    w_ppm_raw = DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights[2]
    w_eirp_raw = DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights[4] + DEFAULT_ASSUMPTIONS.rf_mahalanobis_weights[5]
    w_tot = w_cfo_raw + w_ppm_raw + w_eirp_raw
    w_cfo, w_ppm, w_eirp = w_cfo_raw / w_tot, w_ppm_raw / w_tot, w_eirp_raw / w_tot

    std_cfo = np.maximum(prof_df["std_cfo_khz"].to_numpy(dtype=np.float64), DEFAULT_ASSUMPTIONS.rf_physical_std_floor[1] / 1e3)
    std_ppm = np.maximum(prof_df["std_timing_ppm"].to_numpy(dtype=np.float64), DEFAULT_ASSUMPTIONS.rf_physical_std_floor[2])
    std_eirp = np.maximum(prof_df["std_eirp_dbm"].to_numpy(dtype=np.float64), DEFAULT_ASSUMPTIONS.rf_physical_std_floor[5])

    z_cfo = (sim_cfo_khz - prof_df["mean_cfo_khz"].to_numpy(dtype=np.float64)) / std_cfo
    z_ppm = (sim_ppm - prof_df["mean_timing_ppm"].to_numpy(dtype=np.float64)) / std_ppm
    z_eirp = (sim_eirp - prof_df["mean_eirp_dbm"].to_numpy(dtype=np.float64)) / std_eirp

    d_all = np.sqrt(w_cfo * (z_cfo ** 2) + w_ppm * (z_ppm ** 2) + w_eirp * (z_eirp ** 2))
    target_idx = int(prof_df.index[prof_df["icao_hex"] == target_icao][0])
    d_claimed = float(d_all[target_idx])
    best_idx = int(np.argmin(d_all))
    pred_icao = str(prof_df.loc[best_idx, "icao_hex"])
    pred_cs = str(prof_df.loc[best_idx, "callsign"])

    log_lik = -0.5 * (d_all ** 2)
    lik_shift = np.exp(log_lik - np.max(log_lik))
    post_probs = lik_shift / (np.sum(lik_shift) + 1e-12)
    claimed_prob = float(0.65 * post_probs[target_idx] + 0.35 * np.exp(-0.5 * (d_claimed ** 2)))

    is_spoof = bool(
        d_claimed > outlier_z
        or (
            pred_icao != target_icao
            and d_claimed > DEFAULT_ASSUMPTIONS.secondary_mismatch_z
            and claimed_prob < DEFAULT_ASSUMPTIONS.rf_mismatch_prob_threshold
        )
    )

    res_p90 = residual_stats["mahal_p90"] if residual_stats else 1.65
    res_p95 = residual_stats["mahal_p95"] if residual_stats else 2.15
    d_ci_low = max(0.0, d_claimed - 0.5 * res_p90)
    d_ci_high = d_claimed + 0.5 * res_p90

    return {
        "d_claimed": d_claimed,
        "d_ci_low": d_ci_low,
        "d_ci_high": d_ci_high,
        "claimed_prob": claimed_prob,
        "pred_icao": pred_icao,
        "pred_cs": pred_cs,
        "is_spoof": is_spoof,
        "mahal_p95_backtest": res_p95,
    }
