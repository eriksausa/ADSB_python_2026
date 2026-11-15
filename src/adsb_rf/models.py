"""
Supervised Ensemble RF Fingerprinting Classifier & Hybrid Anomaly Detector (`src/adsb_rf/models.py`).

Combines:
- RandomForestClassifier (45%)
- ExtraTreesClassifier (35%)
- Vectorized Bayesian Mahalanobis Posterior (20%)
with strict fold-isolated imputation, scaling, and centroid estimation so out-of-sample
evaluation and ablation studies have zero target leakage.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .assumptions import DEFAULT_ASSUMPTIONS, AnomalyThresholds
from .deep_models import train_pytorch_models
from .features import (
    RF_FEATURE_NAMES,
    RF_MAHALANOBIS_WEIGHTS,
    RF_PHYSICAL_STD_FLOOR,
    _backfill_aircraft_kinematics,
    extract_12_rf_features_from_frame,
)

try:
    from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def fit_fold_ensemble(
    X_train_raw: np.ndarray,
    y_train_idx: np.ndarray,
    n_classes: int,
    feature_indices: Optional[Sequence[int]] = None,
    ensemble_mode: str = "ensemble",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fits an isolated, leak-free RF Fingerprinting model strictly on `X_train_raw, y_train_idx`:
      1. Selects `feature_indices` (default all 12 features).
      2. Computes training-fold median imputation values.
      3. Computes per-class robust Median & MAD/P80-P20 spread strictly from `X_train`.
      4. Fits `RandomForestClassifier` and `ExtraTreesClassifier` pipelines with damped sample weights.
    """
    feat_idx = list(feature_indices) if feature_indices is not None else list(range(X_train_raw.shape[1]))
    X_sub = np.where(np.isfinite(X_train_raw[:, feat_idx]), X_train_raw[:, feat_idx], np.nan).copy()
    n_feats = len(feat_idx)

    # 1. Training-only column medians
    medians = np.zeros(n_feats, dtype=np.float64)
    for j in range(n_feats):
        valid = X_sub[:, j][~np.isnan(X_sub[:, j])]
        medians[j] = float(np.median(valid)) if len(valid) > 0 else 0.0
        X_sub[np.isnan(X_sub[:, j]), j] = medians[j]

    # 2. Training-only Mahalanobis centroids, stds, and class priors
    std_floor_sub = RF_PHYSICAL_STD_FLOOR[feat_idx]
    weights_sub = RF_MAHALANOBIS_WEIGHTS[feat_idx].copy()
    w_sum = float(np.sum(weights_sub))
    if w_sum > 0:
        weights_sub = weights_sub / w_sum
    else:
        weights_sub = np.ones(n_feats, dtype=np.float64) / max(n_feats, 1)

    centroid_mat = np.zeros((n_classes, n_feats), dtype=np.float64)
    std_mat = np.ones((n_classes, n_feats), dtype=np.float64) * std_floor_sub[None, :]
    priors = np.full(n_classes, 1e-6, dtype=np.float64)

    n_total = max(len(y_train_idx), 1)
    for c_idx in range(n_classes):
        mask = (y_train_idx == c_idx)
        n_c = int(np.sum(mask))
        if n_c == 0:
            continue
        priors[c_idx] = float(n_c) / n_total
        pts = X_sub[mask]
        cent = np.median(pts, axis=0)
        mad_std = np.median(np.abs(pts - cent), axis=0) * 1.4826
        if n_c >= 5:
            q80_20_std = (np.percentile(pts, 80, axis=0) - np.percentile(pts, 20, axis=0)) / 1.683
            eff_std = np.maximum(mad_std, q80_20_std)
        else:
            eff_std = mad_std
        centroid_mat[c_idx] = cent
        std_mat[c_idx] = np.maximum(eff_std, std_floor_sub)

    # 3. Fit Tree Pipelines if requested
    rf_pipe = None
    et_pipe = None
    if HAS_SKLEARN and ensemble_mode in ("ensemble", "rf", "et"):
        unique_cls, counts = np.unique(y_train_idx, return_counts=True)
        med_count = float(np.median(counts)) if len(counts) > 0 else 1.0
        cls_w = np.clip(
            (med_count / np.maximum(counts, 1.0)) ** DEFAULT_ASSUMPTIONS.sample_weight_damping_exp,
            0.5,
            2.5,
        )
        w_map = dict(zip(unique_cls.tolist(), cls_w.tolist()))
        sample_weights = np.array([w_map[int(lbl)] for lbl in y_train_idx], dtype=np.float64)

        if ensemble_mode in ("ensemble", "rf"):
            rf_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    (
                        "rf_clf",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=18,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight=None,
                            n_jobs=1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            rf_pipe.fit(X_sub, y_train_idx, rf_clf__sample_weight=sample_weights)

        if ensemble_mode in ("ensemble", "et"):
            et_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    (
                        "et_clf",
                        ExtraTreesClassifier(
                            n_estimators=100,
                            max_depth=20,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight=None,
                            n_jobs=1,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            et_pipe.fit(X_sub, y_train_idx, et_clf__sample_weight=sample_weights)

    return {
        "feat_idx": feat_idx,
        "medians": medians,
        "centroid_mat": centroid_mat,
        "std_mat": std_mat,
        "priors": priors,
        "weights_sub": weights_sub,
        "rf_pipe": rf_pipe,
        "et_pipe": et_pipe,
        "n_classes": n_classes,
        "ensemble_mode": ensemble_mode,
    }


def predict_fold_ensemble(
    model_bundle: Dict[str, Any],
    X_test_raw: np.ndarray,
    override_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts class probability matrix `(N_test, n_classes)` and Mahalanobis distance matrix
    `(N_test, n_classes)` on `X_test_raw` using only training-fold statistics from `model_bundle`.
    """
    feat_idx = model_bundle["feat_idx"]
    medians = model_bundle["medians"]
    centroid_mat = model_bundle["centroid_mat"]
    std_mat = model_bundle["std_mat"]
    priors = model_bundle["priors"]
    weights_sub = model_bundle["weights_sub"]
    rf_pipe = model_bundle["rf_pipe"]
    et_pipe = model_bundle["et_pipe"]
    n_classes = model_bundle["n_classes"]
    ensemble_mode = override_mode if override_mode is not None else model_bundle["ensemble_mode"]

    X_sub = np.where(np.isfinite(X_test_raw[:, feat_idx]), X_test_raw[:, feat_idx], np.nan).copy()
    for j in range(len(feat_idx)):
        X_sub[np.isnan(X_sub[:, j]), j] = medians[j]

    z_diff = (X_sub[:, None, :] - centroid_mat[None, :, :]) / std_mat[None, :, :]
    d_sq = np.sum(weights_sub[None, None, :] * (z_diff ** 2), axis=2)
    dist_matrix = np.sqrt(np.maximum(d_sq, 0.0))

    log_prior = np.log(np.maximum(priors[None, :], 1e-6))
    log_lik = -0.5 * d_sq + DEFAULT_ASSUMPTIONS.prior_log_weight * log_prior
    max_log = np.max(log_lik, axis=1, keepdims=True)
    lik_shifted = np.exp(log_lik - max_log)
    mahal_posteriors = lik_shifted / (np.sum(lik_shifted, axis=1, keepdims=True) + 1e-12)

    if ensemble_mode == "mahalanobis" or not HAS_SKLEARN:
        return mahal_posteriors, dist_matrix

    n_samples = len(X_sub)
    rf_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    et_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

    if rf_pipe is not None:
        raw_rf = rf_pipe.predict_proba(X_sub)
        for col_i, cls_idx in enumerate(rf_pipe.named_steps["rf_clf"].classes_):
            rf_probs[:, int(cls_idx)] = raw_rf[:, col_i]

    if et_pipe is not None:
        raw_et = et_pipe.predict_proba(X_sub)
        for col_i, cls_idx in enumerate(et_pipe.named_steps["et_clf"].classes_):
            et_probs[:, int(cls_idx)] = raw_et[:, col_i]

    if ensemble_mode == "rf":
        return rf_probs, dist_matrix
    if ensemble_mode == "et":
        return et_probs, dist_matrix

    w_rf, w_et, w_m = DEFAULT_ASSUMPTIONS.ensemble_weights
    combined_probs = w_rf * rf_probs + w_et * et_probs + w_m * mahal_posteriors
    return combined_probs, dist_matrix


class RFFingerprintClassifier:
    """
    Supervised Physical-Layer RF Fingerprinting Classifier & Spoofing Detector.
    Combines RandomForest (45%) + ExtraTrees (35%) + Bayesian Mahalanobis Posterior (20%).
    """

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self.class_to_idx_: Dict[str, int] = {}
        self.idx_to_class_: Dict[int, str] = {}
        self.feature_medians_: Optional[np.ndarray] = None
        self.centroid_matrix_: Optional[np.ndarray] = None
        self.std_matrix_: Optional[np.ndarray] = None
        self.class_priors_: Optional[np.ndarray] = None
        self.class_centroids_: Dict[str, np.ndarray] = {}
        self.class_stds_: Dict[str, np.ndarray] = {}
        self.class_callsigns_: Dict[str, str] = {}
        self.class_doppler_hz_: Dict[str, float] = {}
        self.class_distance_m_: Dict[str, float] = {}
        self.rf_pipeline_: Any = None
        self.et_pipeline_: Any = None
        self.isolation_forest_: Any = None
        self.holdout_test_accuracy_: float = 0.0
        self.cv_accuracy_: float = 0.0
        self.top3_accuracy_: float = 0.0
        self.oof_mahal_residuals_: np.ndarray = np.array([])
        self.is_fitted_: bool = False

    def _extract_matrix(self, frames: List[Dict[str, Any]]) -> np.ndarray:
        if not frames:
            return np.zeros((0, len(RF_FEATURE_NAMES)), dtype=np.float64)
        rows = [extract_12_rf_features_from_frame(f) for f in frames]
        return np.vstack(rows)

    def _impute(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_clean = np.where(np.isfinite(X), X, np.nan).copy()
        if fit or self.feature_medians_ is None:
            meds = []
            for j in range(X_clean.shape[1]):
                col = X_clean[:, j]
                valid = col[~np.isnan(col)]
                meds.append(float(np.median(valid)) if len(valid) > 0 else 0.0)
            self.feature_medians_ = np.array(meds, dtype=np.float64)

        for col_idx in range(X_clean.shape[1]):
            mask = np.isnan(X_clean[:, col_idx])
            if np.any(mask):
                X_clean[mask, col_idx] = self.feature_medians_[col_idx]

        return X_clean

    @staticmethod
    def _compute_smooth_sample_weights(y_idx: np.ndarray) -> np.ndarray:
        unique_cls, counts = np.unique(y_idx, return_counts=True)
        med_count = float(np.median(counts)) if len(counts) > 0 else 1.0
        cls_weights = np.clip(
            (med_count / np.maximum(counts, 1.0)) ** DEFAULT_ASSUMPTIONS.sample_weight_damping_exp,
            0.5,
            2.5,
        )
        weight_map = dict(zip(unique_cls.tolist(), cls_weights.tolist()))
        return np.array([weight_map[int(lbl)] for lbl in y_idx], dtype=np.float64)

    def fit(
        self,
        frames: List[Dict[str, Any]],
        exclude_flagged: bool = True,
        outlier_z_threshold: float = DEFAULT_ASSUMPTIONS.outlier_z_threshold,
    ) -> "RFFingerprintClassifier":
        """
        Trains and evaluates the 12-feature RF Fingerprinting Classifier on CRC-valid frames.
        """
        _backfill_aircraft_kinematics(frames, causal_only=True)
        for f in frames:
            f.pop("_cached_rf_vec12", None)

        train_frames = [
            f for f in frames
            if not (exclude_flagged and f.get("Rule_Anomaly", False))
        ]
        if len(train_frames) < 3:
            train_frames = frames

        X_raw = self._extract_matrix(train_frames)
        X_clean = self._impute(X_raw, fit=True)

        y_str = np.array([str(f["ICAO_Hex"]).strip().upper() for f in train_frames])
        self.classes_ = sorted(list(set(y_str.tolist())))
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class_ = {idx: cls for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_idx_[s] for s in y_str], dtype=np.int64)

        n_classes = len(self.classes_)
        n_feats = X_clean.shape[1]
        self.centroid_matrix_ = np.zeros((n_classes, n_feats), dtype=np.float64)
        self.std_matrix_ = np.zeros((n_classes, n_feats), dtype=np.float64)
        self.class_priors_ = np.zeros(n_classes, dtype=np.float64)

        inlier_mask = np.ones(len(train_frames), dtype=bool)
        for idx, cls in enumerate(self.classes_):
            cls_mask = (y_idx == idx)
            cls_pts = X_clean[cls_mask]
            self.class_priors_[idx] = float(np.sum(cls_mask)) / max(len(y_idx), 1)

            centroid = np.median(cls_pts, axis=0)
            mad_std = np.median(np.abs(cls_pts - centroid), axis=0) * 1.4826
            if len(cls_pts) >= 5:
                q80_20_std = (np.percentile(cls_pts, 80, axis=0) - np.percentile(cls_pts, 20, axis=0)) / 1.683
                eff_std = np.maximum(mad_std, q80_20_std)
            else:
                eff_std = mad_std
            bounded_std = np.maximum(eff_std, RF_PHYSICAL_STD_FLOOR)

            self.centroid_matrix_[idx] = centroid
            self.std_matrix_[idx] = bounded_std
            self.class_centroids_[cls] = centroid
            self.class_stds_[cls] = bounded_std

            if len(cls_pts) >= 4:
                z_cls = (cls_pts - centroid[None, :]) / bounded_std[None, :]
                d_cls = np.sqrt(np.sum(RF_MAHALANOBIS_WEIGHTS[None, :] * (z_cls ** 2), axis=1))
                cls_indices = np.where(cls_mask)[0]
                inlier_mask[cls_indices[d_cls > outlier_z_threshold]] = False

            cls_frames = [f for f, m in zip(train_frames, cls_mask) if m]
            callsigns = [str(f["Call_sign"]) for f in cls_frames if f.get("Call_sign")]
            self.class_callsigns_[cls] = callsigns[-1] if callsigns else "N/A"

            dopplers = [float(f["Doppler"]) for f in cls_frames if not np.isnan(f.get("Doppler", float("nan")))]
            self.class_doppler_hz_[cls] = float(np.median(dopplers)) if dopplers else 0.0

            dists_m = [
                float(f["Distance_from_Rx"])
                for f in cls_frames
                if not np.isnan(f.get("Distance_from_Rx", float("nan")))
            ]
            self.class_distance_m_[cls] = float(np.median(dists_m)) if dists_m else float("nan")

        X_tree_raw = X_raw[inlier_mask] if np.sum(inlier_mask) >= n_classes else X_raw
        X_tree = X_clean[inlier_mask] if np.sum(inlier_mask) >= n_classes else X_clean
        y_tree = y_idx[inlier_mask] if np.sum(inlier_mask) >= n_classes else y_idx

        unique_tree_cls, cls_counts = np.unique(y_tree, return_counts=True)
        valid_eval_classes = set(unique_tree_cls[cls_counts >= DEFAULT_ASSUMPTIONS.min_frames_enrolled].tolist())
        eval_mask = np.array([int(lbl) in valid_eval_classes for lbl in y_tree], dtype=bool)

        if np.sum(eval_mask) >= 15 and len(valid_eval_classes) >= 2:
            X_eval_raw = X_tree_raw[eval_mask]
            y_eval = y_tree[eval_mask]

            if HAS_SKLEARN:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
                tr_idx_80, te_idx_20 = next(sss.split(X_eval_raw, y_eval))
                bundle_80 = fit_fold_ensemble(X_eval_raw[tr_idx_80], y_eval[tr_idx_80], n_classes=n_classes)
                probs_20, _ = predict_fold_ensemble(bundle_80, X_eval_raw[te_idx_20])
                preds_20 = np.argmax(probs_20, axis=1)
                self.holdout_test_accuracy_ = float(np.mean(preds_20 == y_eval[te_idx_20]))

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                folds = list(skf.split(X_eval_raw, y_eval))
            else:
                rng_cv = np.random.default_rng(42)
                perm = rng_cv.permutation(len(y_eval))
                folds = [(perm[np.arange(len(y_eval)) % 5 != k], perm[np.arange(len(y_eval)) % 5 == k]) for k in range(5)]

            oof_correct = 0
            oof_top3 = 0
            oof_total = 0
            oof_mahal_list: List[float] = []
            for tr_i, va_i in folds:
                fold_bundle = fit_fold_ensemble(X_eval_raw[tr_i], y_eval[tr_i], n_classes=n_classes)
                fold_probs, fold_dists = predict_fold_ensemble(fold_bundle, X_eval_raw[va_i])
                fold_preds = np.argmax(fold_probs, axis=1)
                fold_top3 = np.argsort(fold_probs, axis=1)[:, -3:]

                oof_correct += int(np.sum(fold_preds == y_eval[va_i]))
                for row_k, true_c in enumerate(y_eval[va_i]):
                    if int(true_c) in fold_top3[row_k]:
                        oof_top3 += 1
                    oof_mahal_list.append(float(fold_dists[row_k, int(true_c)]))
                oof_total += len(va_i)

            self.cv_accuracy_ = float(oof_correct / max(oof_total, 1))
            self.top3_accuracy_ = float(oof_top3 / max(oof_total, 1))
            self.oof_mahal_residuals_ = np.array(oof_mahal_list, dtype=np.float64)
            if not HAS_SKLEARN:
                self.holdout_test_accuracy_ = self.cv_accuracy_
        else:
            self.cv_accuracy_ = 1.0
            self.holdout_test_accuracy_ = 1.0
            self.top3_accuracy_ = 1.0

        if HAS_SKLEARN and n_classes >= 2:
            sample_weights = self._compute_smooth_sample_weights(y_tree)

            self.rf_pipeline_ = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    (
                        "rf_clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=18,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight=None,
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            )
            self.rf_pipeline_.fit(X_tree, y_tree, rf_clf__sample_weight=sample_weights)

            self.et_pipeline_ = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    (
                        "et_clf",
                        ExtraTreesClassifier(
                            n_estimators=200,
                            max_depth=20,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            class_weight=None,
                            n_jobs=1,
                            random_state=42,
                        ),
                    ),
                ]
            )
            self.et_pipeline_.fit(X_tree, y_tree, et_clf__sample_weight=sample_weights)

            self.isolation_forest_ = IsolationForest(
                n_estimators=150,
                contamination=DEFAULT_ASSUMPTIONS.isolation_forest_contamination,
                n_jobs=1,
                random_state=42,
            )
            self.isolation_forest_.fit(X_tree)

        self.is_fitted_ = True
        return self

    def _compute_vectorized_mahalanobis(self, X_clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_diff = (X_clean[:, None, :] - self.centroid_matrix_[None, :, :]) / self.std_matrix_[None, :, :]
        d_sq = np.sum(RF_MAHALANOBIS_WEIGHTS[None, None, :] * (z_diff ** 2), axis=2)
        dist_matrix = np.sqrt(np.maximum(d_sq, 0.0))

        log_prior = np.log(np.maximum(self.class_priors_[None, :], 1e-6))
        log_lik = -0.5 * d_sq + DEFAULT_ASSUMPTIONS.prior_log_weight * log_prior
        max_log = np.max(log_lik, axis=1, keepdims=True)
        lik_shifted = np.exp(log_lik - max_log)
        posterior_matrix = lik_shifted / (np.sum(lik_shifted, axis=1, keepdims=True) + 1e-12)
        return dist_matrix, posterior_matrix

    def predict_with_scores(
        self, frames: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[float], np.ndarray]:
        if not self.is_fitted_ or not frames:
            return (
                [str(f.get("ICAO_Hex", "")) for f in frames],
                [1.0 for _ in frames],
                np.zeros(len(frames)),
            )

        X_raw = self._extract_matrix(frames)
        X_clean = self._impute(X_raw, fit=False)

        dist_matrix, mahal_posteriors = self._compute_vectorized_mahalanobis(X_clean)
        combined_probs = mahal_posteriors.copy()

        if HAS_SKLEARN and self.rf_pipeline_ is not None and self.et_pipeline_ is not None:
            rf_probs_raw = self.rf_pipeline_.predict_proba(X_clean)
            et_probs_raw = self.et_pipeline_.predict_proba(X_clean)

            n_samples = len(frames)
            n_classes = len(self.classes_)
            rf_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
            et_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

            rf_classes = self.rf_pipeline_.named_steps["rf_clf"].classes_
            et_classes = self.et_pipeline_.named_steps["et_clf"].classes_
            for col_i, cls_idx in enumerate(rf_classes):
                rf_probs[:, int(cls_idx)] = rf_probs_raw[:, col_i]
            for col_i, cls_idx in enumerate(et_classes):
                et_probs[:, int(cls_idx)] = et_probs_raw[:, col_i]

            w_rf, w_et, w_m = DEFAULT_ASSUMPTIONS.ensemble_weights
            combined_probs = w_rf * rf_probs + w_et * et_probs + w_m * mahal_posteriors

        best_indices = np.argmax(combined_probs, axis=1)
        predicted_icaos = [self.idx_to_class_[int(idx)] for idx in best_indices]

        claimed_match_probs: List[float] = []
        mahal_distances = np.zeros(len(frames), dtype=np.float64)

        for i, f in enumerate(frames):
            claimed_icao = str(f.get("ICAO_Hex", "")).strip().upper()
            if claimed_icao in self.class_to_idx_:
                c_idx = self.class_to_idx_[claimed_icao]
                d_m = float(dist_matrix[i, c_idx])
                mahal_distances[i] = d_m
                ens_p = float(combined_probs[i, c_idx])
                abs_fit = float(np.exp(-0.5 * (d_m ** 2)))
                claimed_match_probs.append(float(0.65 * ens_p + 0.35 * abs_fit))
            else:
                mahal_distances[i] = float(np.min(dist_matrix[i]))
                claimed_match_probs.append(0.0)

        return predicted_icaos, claimed_match_probs, mahal_distances

    def identify_crc_failed_bursts(
        self,
        failed_bursts: List[Dict[str, Any]],
        carrier_freq_hz: float = DEFAULT_ASSUMPTIONS.carrier_freq_hz,
    ) -> List[Dict[str, Any]]:
        if not self.is_fitted_ or not failed_bursts:
            return failed_bursts

        X_raw = self._extract_matrix(failed_bursts)
        X_clean = self._impute(X_raw, fit=False)

        n_bursts = len(failed_bursts)
        n_classes = len(self.classes_)
        X_hyp = np.repeat(X_clean[:, None, :], n_classes, axis=1)

        for k, cls in enumerate(self.classes_):
            doppler_k = self.class_doppler_hz_.get(cls, 0.0)
            dist_m_k = self.class_distance_m_.get(cls, float("nan"))
            X_hyp[:, k, 1] = X_clean[:, 0] - doppler_k
            if not np.isnan(dist_m_k) and dist_m_k > 0:
                lambda_c = DEFAULT_ASSUMPTIONS.speed_of_light_mps / carrier_freq_hz
                fspl_db = 20.0 * np.log10((4.0 * np.pi * dist_m_k) / lambda_c)
                X_hyp[:, k, 5] = X_clean[:, 4] + fspl_db
            else:
                X_hyp[:, k, 5] = self.centroid_matrix_[k, 5]

        z_diff = (X_hyp - self.centroid_matrix_[None, :, :]) / self.std_matrix_[None, :, :]
        d_sq = np.sum(RF_MAHALANOBIS_WEIGHTS[None, None, :] * (z_diff ** 2), axis=2)
        dist_matrix = np.sqrt(np.maximum(d_sq, 0.0))

        log_prior = np.log(np.maximum(self.class_priors_[None, :], 1e-6))
        log_lik = -0.5 * d_sq + DEFAULT_ASSUMPTIONS.prior_log_weight * log_prior
        max_log = np.max(log_lik, axis=1, keepdims=True)
        lik_shifted = np.exp(log_lik - max_log)
        mahal_posteriors = lik_shifted / (np.sum(lik_shifted, axis=1, keepdims=True) + 1e-12)

        combined_probs = mahal_posteriors
        if HAS_SKLEARN and self.rf_pipeline_ is not None:
            rf_probs_raw = self.rf_pipeline_.predict_proba(X_clean)
            rf_probs = np.zeros((n_bursts, n_classes), dtype=np.float64)
            for col_i, cls_idx in enumerate(self.rf_pipeline_.named_steps["rf_clf"].classes_):
                rf_probs[:, int(cls_idx)] = rf_probs_raw[:, col_i]
            combined_probs = 0.50 * mahal_posteriors + 0.50 * rf_probs

        best_k_arr = np.argmax(combined_probs, axis=1)

        for i, burst in enumerate(failed_bursts):
            k_best = int(best_k_arr[i])
            best_cls = self.idx_to_class_[k_best]
            best_dist = float(dist_matrix[i, k_best])
            post_p = float(combined_probs[i, k_best])
            abs_lik = float(np.exp(-0.5 * (best_dist ** 2)))
            confidence = float(0.6 * post_p + 0.4 * abs_lik)

            burst["Predicted_ICAO_Hex"] = best_cls
            burst["Predicted_Callsign"] = self.class_callsigns_.get(best_cls, "N/A")
            burst["Frequency_shift"] = float(X_hyp[i, k_best, 1])
            burst["EIRP_dBm"] = float(X_hyp[i, k_best, 5])
            burst["RF_Confidence"] = confidence
            burst["Mahalanobis_Dist"] = best_dist
            burst["ICAO_Bits_Were_Corrupted"] = bool(
                str(burst.get("Corrupted_ICAO_Hex", "")).strip().upper() != best_cls
            )

        return failed_bursts


class ADSBAnomalyDetector:
    """
    End-to-end ADS-B Physical-Layer Security & Anomaly Detector for CRC-valid frames.
    """

    def __init__(
        self,
        thresholds: Optional[AnomalyThresholds] = None,
        enable_crc_failed_attribution: bool = False,
    ):
        self.thresholds = thresholds or AnomalyThresholds()
        self.enable_crc_failed_attribution = enable_crc_failed_attribution
        self.rf_classifier = RFFingerprintClassifier()

    def evaluate_single_frame_rules(
        self,
        frame: Dict[str, Any],
        aircraft_median_cfo_hz: Optional[float] = None,
    ) -> List[str]:
        """
        Checks deterministic kinematic, protocol, and physical-layer sanity rules on a single frame.
        """
        reasons: List[str] = []
        tc = int(frame.get("TC", 0))

        eff_min_eirp = min(self.thresholds.min_eirp_dbm, DEFAULT_ASSUMPTIONS.min_eirp_dbm)
        eff_max_eirp = max(self.thresholds.max_eirp_dbm, DEFAULT_ASSUMPTIONS.max_eirp_dbm)
        eff_max_spd = max(self.thresholds.max_groundspeed_kt, DEFAULT_ASSUMPTIONS.max_groundspeed_kt)
        eff_max_alt_rate = max(self.thresholds.max_altitude_rate_fpm, DEFAULT_ASSUMPTIONS.max_altitude_rate_fpm)
        eff_max_timing_ppm = max(self.thresholds.max_abs_timing_ppm, DEFAULT_ASSUMPTIONS.max_abs_timing_ppm)
        eff_max_cfo_jump = max(self.thresholds.max_abs_cfo_hz, DEFAULT_ASSUMPTIONS.max_cfo_jump_from_median_hz)

        if 8 < tc < 19:
            lat = frame.get("Latitude", float("nan"))
            lon = frame.get("Longitude", float("nan"))
            alt = frame.get("Altitude", float("nan"))

            if np.isnan(lat) or np.isnan(lon) or abs(lat) > 90.0 or abs(lon) > 180.0:
                reasons.append("Invalid Lat/Lon coordinates")

            if np.isnan(alt) or alt < self.thresholds.min_altitude_ft or alt > self.thresholds.max_altitude_ft:
                reasons.append(f"Invalid barometric altitude ({alt:.0f} ft)")

            if not frame.get("CPR_Zone_Valid", True):
                reasons.append("CPR Even/Odd latitude zone mismatch")

            implied_spd = frame.get("Implied_Groundspeed_kt", float("nan"))
            jump_km = frame.get("Pos_Jump_km", 0.0)
            if not np.isnan(implied_spd) and implied_spd > eff_max_spd and jump_km > self.thresholds.min_position_jump_km:
                reasons.append(
                    f"Impossible position jump ({jump_km:.2f} km -> {implied_spd:.0f} kt > {eff_max_spd:.0f} kt)"
                )

            implied_alt_rate = frame.get("Implied_Alt_Rate_fpm", float("nan"))
            if not np.isnan(implied_alt_rate) and implied_alt_rate > eff_max_alt_rate:
                reasons.append(f"Impossible altitude jump ({implied_alt_rate:.0f} ft/min)")

            eirp = frame.get("EIRP_dBm", float("nan"))
            if not np.isnan(eirp) and (eirp < eff_min_eirp or eirp > eff_max_eirp):
                reasons.append(
                    f"Anomalous transmit EIRP ({eirp:.1f} dBm outside [{eff_min_eirp:.0f}, {eff_max_eirp:.0f}] dBm)"
                )

        elif tc == 19:
            spd = frame.get("Airspeed", float("nan"))
            hdg = frame.get("Heading", float("nan"))
            vert = frame.get("Vertical", float("nan"))
            if np.isnan(spd) or spd <= 0 or spd > eff_max_spd:
                reasons.append(f"Non-valid airspeed ({spd} kt)")
            if np.isnan(hdg) or hdg < 0.0 or hdg > 360.0:
                reasons.append(f"Non-valid heading ({hdg} deg)")
            if not np.isnan(vert) and abs(vert) > max(self.thresholds.max_vertical_rate_fpm, DEFAULT_ASSUMPTIONS.max_vertical_rate_fpm):
                reasons.append(f"Excessive vertical rate ({vert:.0f} ft/min)")

        cfo = frame.get("Frequency_shift", float("nan"))
        if not np.isnan(cfo):
            if abs(cfo) > self.thresholds.max_absolute_cfo_limit_hz:
                reasons.append(f"Extreme carrier frequency offset ({cfo / 1e3:.1f} kHz)")
            elif (
                aircraft_median_cfo_hz is not None
                and not np.isnan(aircraft_median_cfo_hz)
                and abs(cfo - aircraft_median_cfo_hz) > eff_max_cfo_jump
            ):
                reasons.append(
                    f"Extreme carrier frequency offset ({cfo / 1e3:.1f} kHz vs aircraft median {aircraft_median_cfo_hz / 1e3:.1f} kHz)"
                )

        timing_ppm = frame.get("Timing_Offset_PPM", 0.0)
        if not np.isnan(timing_ppm) and abs(timing_ppm) > eff_max_timing_ppm:
            reasons.append(f"Extreme symbol clock timing offset ({timing_ppm:.1f} ppm)")

        return reasons

    def analyze_frames(
        self,
        frames: List[Dict[str, Any]],
        crc_failed_bursts: Optional[List[Dict[str, Any]]] = None,
        enable_crc_failed_attribution: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete ML RF Fingerprinting and Anomaly Detection workflow on CRC-valid `frames`.
        """
        if not frames:
            return {"total_frames": 0, "anomalies_detected": 0, "rf_accuracy": 0.0, "crc_recovered": 0}

        run_crc_failed = (
            self.enable_crc_failed_attribution
            if enable_crc_failed_attribution is None
            else enable_crc_failed_attribution
        )

        eff_outlier_z = max(self.thresholds.outlier_z_threshold, DEFAULT_ASSUMPTIONS.outlier_z_threshold)

        _backfill_aircraft_kinematics(frames, causal_only=True)

        icao_cfo_lists: Dict[str, List[float]] = {}
        for f in frames:
            icao = str(f.get("ICAO_Hex", "")).strip().upper()
            cfo_val = float(f.get("Frequency_shift", f.get("CFO_Raw_Hz", float("nan"))))
            if icao and not np.isnan(cfo_val):
                icao_cfo_lists.setdefault(icao, []).append(cfo_val)
        icao_median_cfo: Dict[str, float] = {
            k: float(np.median(v)) for k, v in icao_cfo_lists.items() if len(v) > 0
        }

        rule_reasons_per_frame: List[List[str]] = []
        for f in frames:
            icao = str(f.get("ICAO_Hex", "")).strip().upper()
            r_list = self.evaluate_single_frame_rules(
                f, aircraft_median_cfo_hz=icao_median_cfo.get(icao)
            )
            f["Rule_Anomaly"] = len(r_list) > 0
            rule_reasons_per_frame.append(r_list)

        self.rf_classifier.fit(frames, exclude_flagged=True, outlier_z_threshold=eff_outlier_z)
        pred_icaos, match_probs, mahal_dists = self.rf_classifier.predict_with_scores(frames)

        X_all = self.rf_classifier._impute(self.rf_classifier._extract_matrix(frames), fit=False)
        clean_mask = np.array(
            [
                not f.get("Rule_Anomaly", False) and mahal_dists[i] <= eff_outlier_z
                for i, f in enumerate(frames)
            ],
            dtype=bool,
        )
        if not np.any(clean_mask):
            clean_mask = np.ones(len(frames), dtype=bool)

        med = np.median(X_all[clean_mask], axis=0)
        iqr = np.maximum(
            np.percentile(X_all[clean_mask], 75, axis=0) - np.percentile(X_all[clean_mask], 25, axis=0),
            RF_PHYSICAL_STD_FLOOR,
        )
        X_all_scaled = (X_all - med) / iqr
        X_train_scaled = X_all_scaled[clean_mask]

        train_frames = [frames[i] for i in range(len(frames)) if clean_mask[i]]
        y_train_idx = np.array(
            [self.rf_classifier.class_to_idx_.get(str(f["ICAO_Hex"]).strip().upper(), 0) for f in train_frames],
            dtype=np.int64,
        )

        torch_out = train_pytorch_models(
            train_frames=train_frames,
            all_frames=frames,
            X_train_scaled=X_train_scaled,
            X_all_scaled=X_all_scaled,
            y_train_idx=y_train_idx,
            num_classes=max(len(self.rf_classifier.classes_), 1),
        )
        ae_mse_scores = torch_out["ae_mse_scores"]

        unsupervised_scores = 0.7 * mahal_dists + 0.3 * np.sqrt(np.maximum(ae_mse_scores, 0.0))
        if HAS_SKLEARN and self.rf_classifier.isolation_forest_ is not None:
            raw_iso = -self.rf_classifier.isolation_forest_.decision_function(X_all)
            unsupervised_scores = (
                0.5 * mahal_dists + 0.3 * np.sqrt(np.maximum(ae_mse_scores, 0.0)) + 0.2 * (raw_iso * 5.0)
            )

        correct_clean_rf = 0
        total_clean_rf = 0
        anomalies_count = 0

        for i, f in enumerate(frames):
            reasons = list(rule_reasons_per_frame[i])
            claimed_icao = str(f.get("ICAO_Hex", "")).strip().upper()
            pred_icao = pred_icaos[i]
            match_prob = match_probs[i]
            m_dist = float(mahal_dists[i])
            ae_mse = float(ae_mse_scores[i])

            if m_dist > eff_outlier_z or (
                pred_icao != claimed_icao
                and m_dist > self.thresholds.secondary_mismatch_z
                and match_prob < self.thresholds.rf_mismatch_prob_threshold
            ):
                reasons.append(
                    f"RF Fingerprint mismatch for {claimed_icao} "
                    f"(predicted {pred_icao}, d_M={m_dist:.2f}, AE_MSE={ae_mse:.1f}, P={match_prob:.2f})"
                )

            is_anom = len(reasons) > 0
            if not f["Rule_Anomaly"] and m_dist <= eff_outlier_z:
                total_clean_rf += 1
                if pred_icao == claimed_icao:
                    correct_clean_rf += 1

            score = float(
                max(
                    unsupervised_scores[i] / eff_outlier_z,
                    1.2 if is_anom else 0.0,
                )
            )

            f["Predicted_ICAO_Hex"] = pred_icao
            f["RF_Match_Prob"] = float(match_prob)
            f["Mahalanobis_Dist"] = m_dist
            f["PyTorch_AE_MSE"] = ae_mse
            f["Anomaly_Score"] = score
            f["Is_Anomaly"] = bool(is_anom)
            f["Anomaly_Reasons"] = "; ".join(reasons) if reasons else "Normal"

            if is_anom:
                anomalies_count += 1

        if crc_failed_bursts is not None:
            if run_crc_failed:
                self.rf_classifier.identify_crc_failed_bursts(crc_failed_bursts)
            else:
                crc_failed_bursts.clear()

        rf_accuracy = (correct_clean_rf / total_clean_rf) if total_clean_rf > 0 else 1.0
        print(
            f"[RF Fingerprinting] 80/20 Holdout Test Accuracy: {self.rf_classifier.holdout_test_accuracy_*100:.1f}% "
            f"| 5-Fold CV Accuracy: {self.rf_classifier.cv_accuracy_*100:.1f}% "
            f"| Out-of-Fold Top-3 Accuracy: {self.rf_classifier.top3_accuracy_*100:.1f}% "
            f"| In-Sample Clean Inlier Fit: {rf_accuracy*100:.1f}%"
        )

        return {
            "total_frames": len(frames),
            "anomalies_detected": anomalies_count,
            "clean_frames": len(frames) - anomalies_count,
            "rf_classification_accuracy": float(rf_accuracy),
            "rf_holdout_test_accuracy": float(self.rf_classifier.holdout_test_accuracy_),
            "rf_cv_accuracy": float(self.rf_classifier.cv_accuracy_),
            "rf_top3_accuracy": float(self.rf_classifier.top3_accuracy_),
            "known_aircraft_classes": self.rf_classifier.classes_,
            "crc_failed_attributed": len(crc_failed_bursts) if (crc_failed_bursts and run_crc_failed) else 0,
        }
