"""
Modular Anomaly Detectors for Industrial IoT Machine Telemetry.
Provides a clean, uniform interface for:
- Method A: Fixed Domain & Statistical Thresholds
- Method B: Training-set Standardized Z-Score Anomaly Detection
- Method C: Multivariate Isolation Forest Anomaly Detection
- Method D: Ensemble / Comparative Combined Evaluator
"""

from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib

from src.config import (
    NUMERIC_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURE_COLS,
    DOMAIN_THRESHOLDS,
    ZSCORE_THRESHOLD,
    IFOREST_PARAMS,
    RANDOM_SEED
)


class BaseAnomalyDetector(ABC):
    """Abstract Base Class for all IoT Anomaly Detectors."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseAnomalyDetector":
        """Fit detector on training data."""
        pass

    @abstractmethod
    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single telemetry record dictionary.
        Returns dict containing: is_anomaly (0 or 1), score (float), explanation (str).
        """
        pass

    @abstractmethod
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a DataFrame of telemetry records.
        Returns DataFrame with prediction columns appended.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize detector state to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> "BaseAnomalyDetector":
        """Load detector state from disk."""
        pass


class ThresholdDetector(BaseAnomalyDetector):
    """
    Method A: Physics-grounded Domain and Statistical Thresholds.
    Implements failure-mode threshold rules based on Matzka (2020) milling machine physics:
    - Heat Dissipation Failure (HDF) condition: temp_diff < 8.6 K & speed < 1380 rpm
    - Power Failure (PWF) condition: power < 3500 W or > 9000 W
    - Tool Wear Failure (TWF) condition: tool_wear >= 200 min
    - Overstrain Failure (OSF) condition: (tool_wear * torque) > variant limit
    - Extreme speed or torque outliers
    """

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or DOMAIN_THRESHOLDS.copy()
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ThresholdDetector":
        # Thresholds are domain physics-based, but we can verify / calibrate against train percentiles
        self.fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        air_temp = float(record.get("air_temperature", 300.0))
        proc_temp = float(record.get("process_temperature", 310.0))
        speed = float(record.get("rotational_speed", 1500.0))
        torque = float(record.get("torque", 40.0))
        tool_wear = float(record.get("tool_wear", 0.0))
        product_type = str(record.get("product_type", "L"))

        # Derived physics values
        temp_diff = proc_temp - air_temp
        power = torque * speed * (2.0 * math.pi / 60.0)
        overstrain = tool_wear * torque

        triggers = []
        severity = 0.0

        # Rule 1: Heat dissipation
        if temp_diff < self.thresholds["hdf_temp_diff_min"] and speed < self.thresholds["hdf_speed_max"]:
            triggers.append(f"HDF_Rule(temp_diff={temp_diff:.1f}K<8.6K,speed={speed:.0f}<1380)")
            severity = max(severity, 0.85)

        # Rule 2: Power bounds
        if power < self.thresholds["power_min_watts"]:
            triggers.append(f"PWF_LowPower({power:.0f}W<3500W)")
            severity = max(severity, 0.90)
        elif power > self.thresholds["power_max_watts"]:
            triggers.append(f"PWF_HighPower({power:.0f}W>9000W)")
            severity = max(severity, 0.95)

        # Rule 3: Tool wear limit
        if tool_wear >= self.thresholds["tool_wear_max_min"]:
            triggers.append(f"TWF_ToolWear({tool_wear:.0f}min>=200min)")
            severity = max(severity, 0.80)

        # Rule 4: Overstrain product
        os_limit = self.thresholds["overstrain_limits"].get(product_type, 11000.0)
        if overstrain > os_limit:
            triggers.append(f"OSF_Overstrain({overstrain:.0f}>{os_limit:.0f})")
            severity = max(severity, 0.90)

        # Rule 5: Univariate extremes
        if torque > self.thresholds["torque_high"]:
            triggers.append(f"HighTorque({torque:.1f}Nm>{self.thresholds['torque_high']})")
            severity = max(severity, 0.75)
        elif torque < self.thresholds["torque_low"]:
            triggers.append(f"LowTorque({torque:.1f}Nm<{self.thresholds['torque_low']})")
            severity = max(severity, 0.75)

        if speed > self.thresholds["rotational_speed_high"]:
            triggers.append(f"HighSpeed({speed:.0f}rpm>{self.thresholds['rotational_speed_high']})")
            severity = max(severity, 0.70)
        elif speed < self.thresholds["rotational_speed_low"]:
            triggers.append(f"LowSpeed({speed:.0f}rpm<{self.thresholds['rotational_speed_low']})")
            severity = max(severity, 0.70)

        is_anomaly = 1 if len(triggers) > 0 else 0
        explanation = "; ".join(triggers) if triggers else "Normal operation"

        return {
            "pred_threshold": is_anomaly,
            "score_threshold": float(severity),
            "reason_threshold": explanation
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        scores = []
        reasons = []
        for _, row in df.iterrows():
            res = self.predict_record(row.to_dict())
            preds.append(res["pred_threshold"])
            scores.append(res["score_threshold"])
            reasons.append(res["reason_threshold"])

        out_df = df.copy()
        out_df["pred_threshold"] = preds
        out_df["score_threshold"] = scores
        out_df["reason_threshold"] = reasons
        return out_df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)

    def load(self, path: Path) -> "ThresholdDetector":
        with open(path, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)
        self.fitted = True
        return self


class ZScoreDetector(BaseAnomalyDetector):
    """
    Method B: Standardized Z-Score Anomaly Detector.
    Calculates feature distances from the training-set normal baseline:
    z_i = (x_i - mean_i) / std_i.
    Triggers anomaly if the root-mean-squared Z-score across features or max(|z_i|)
    exceeds the configured threshold (e.g. 3.0 standard deviations).
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        threshold: float = ZSCORE_THRESHOLD
    ):
        self.features = features or ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "temp_diff", "power_watts"]
        self.threshold = threshold
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ZScoreDetector":
        train_df = train_df.copy()
        if "temp_diff" not in train_df.columns:
            train_df["temp_diff"] = train_df["process_temperature"] - train_df["air_temperature"]
        if "power_watts" not in train_df.columns:
            train_df["power_watts"] = train_df["torque"] * train_df["rotational_speed"] * (2.0 * math.pi / 60.0)

        for col in self.features:
            if col in train_df.columns:
                m = float(train_df[col].mean())
                s = float(train_df[col].std())
                self.means[col] = m
                self.stds[col] = s if s > 1e-6 else 1.0

        self.fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("ZScoreDetector must be fitted before predict_record()")

        rec = record.copy()
        if "temp_diff" not in rec and "process_temperature" in rec and "air_temperature" in rec:
            rec["temp_diff"] = float(rec["process_temperature"]) - float(rec["air_temperature"])
        if "power_watts" not in rec and "torque" in rec and "rotational_speed" in rec:
            rec["power_watts"] = float(rec["torque"]) * float(rec["rotational_speed"]) * (2.0 * math.pi / 60.0)

        z_scores = {}
        squared_z_sum = 0.0
        n_features = 0

        for col in self.features:
            val = float(rec.get(col, self.means.get(col, 0.0)))
            z = (val - self.means[col]) / self.stds[col]
            z_scores[col] = z
            squared_z_sum += z * z
            n_features += 1

        rms_z = math.sqrt(squared_z_sum / max(n_features, 1))
        max_abs_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0

        # Anomaly when max |z| exceeds threshold or RMS Z exceeds 2.5
        is_anomaly = 1 if (max_abs_z >= self.threshold or rms_z >= 2.5) else 0

        return {
            "pred_zscore": is_anomaly,
            "score_zscore": round(float(rms_z), 4),
            "max_abs_z": round(float(max_abs_z), 4)
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        preds = []
        scores = []
        max_zs = []
        for _, row in df.iterrows():
            res = self.predict_record(row.to_dict())
            preds.append(res["pred_zscore"])
            scores.append(res["score_zscore"])
            max_zs.append(res["max_abs_z"])

        out_df = df.copy()
        out_df["pred_zscore"] = preds
        out_df["score_zscore"] = scores
        out_df["max_abs_z"] = max_zs
        return out_df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "features": self.features,
            "threshold": self.threshold,
            "means": self.means,
            "stds": self.stds
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> "ZScoreDetector":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.features = data["features"]
        self.threshold = data["threshold"]
        self.means = data["means"]
        self.stds = data["stds"]
        self.fitted = True
        return self


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Method C: Multivariate Isolation Forest Anomaly Detection.
    Trained offline on normal operating samples (machine_failure == 0).
    Uses StandardScaler on input features to prevent scale dominance.
    Score output: normalized anomaly score where values > threshold indicate anomalies.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self.features = features or ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "temp_diff", "power_watts", "overstrain_val"]
        self.params = model_params or IFOREST_PARAMS.copy()
        self.scaler = StandardScaler()
        self.model = IsolationForest(**self.params)
        self.decision_threshold = 0.0  # calibrated on train set
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "IsolationForestDetector":
        train_df = train_df.copy()
        # Compute derived features if missing
        if "temp_diff" not in train_df.columns:
            train_df["temp_diff"] = train_df["process_temperature"] - train_df["air_temperature"]
        if "power_watts" not in train_df.columns:
            train_df["power_watts"] = train_df["torque"] * train_df["rotational_speed"] * (2.0 * math.pi / 60.0)
        if "overstrain_val" not in train_df.columns:
            train_df["overstrain_val"] = train_df["tool_wear"] * train_df["torque"]

        X = train_df[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Calibrate decision threshold: decision_function returns negative values for anomalies
        # We define score = -decision_function(X) so higher score = more anomalous
        scores = -self.model.decision_function(X_scaled)
        contamination = float(self.params.get("contamination", 0.035))
        self.decision_threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))

        self.fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before predict_record()")

        rec = record.copy()
        if "temp_diff" not in rec and "process_temperature" in rec and "air_temperature" in rec:
            rec["temp_diff"] = float(rec["process_temperature"]) - float(rec["air_temperature"])
        if "power_watts" not in rec and "torque" in rec and "rotational_speed" in rec:
            rec["power_watts"] = float(rec["torque"]) * float(rec["rotational_speed"]) * (2.0 * math.pi / 60.0)
        if "overstrain_val" not in rec and "tool_wear" in rec and "torque" in rec:
            rec["overstrain_val"] = float(rec["tool_wear"]) * float(rec["torque"])

        vals = np.array([[float(rec.get(col, 0.0)) for col in self.features]])
        vals_scaled = self.scaler.transform(vals)

        # Raw decision score: invert so positive higher values are anomalous
        score = float(-self.model.decision_function(vals_scaled)[0])
        is_anomaly = 1 if score >= self.decision_threshold else 0

        return {
            "pred_iforest": is_anomaly,
            "score_iforest": round(score, 4),
            "iforest_threshold": round(self.decision_threshold, 4)
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "temp_diff" not in df.columns:
            df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
        if "power_watts" not in df.columns:
            df["power_watts"] = df["torque"] * df["rotational_speed"] * (2.0 * math.pi / 60.0)
        if "overstrain_val" not in df.columns:
            df["overstrain_val"] = df["tool_wear"] * df["torque"]

        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        preds = (scores >= self.decision_threshold).astype(int)

        out_df = df.copy()
        out_df["pred_iforest"] = preds
        out_df["score_iforest"] = np.round(scores, 4)
        out_df["iforest_threshold"] = round(self.decision_threshold, 4)
        return out_df

    def save(self, model_path: Path, scaler_path: Path, meta_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        meta = {
            "features": self.features,
            "decision_threshold": self.decision_threshold,
            "params": self.params
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, model_path: Path, scaler_path: Path, meta_path: Path) -> "IsolationForestDetector":
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.features = meta["features"]
        self.decision_threshold = meta["decision_threshold"]
        self.params = meta["params"]
        self.fitted = True
        return self


class UnifiedPipelineDetector:
    """
    Orchestrates the Threshold, Z-score, and Isolation Forest detectors into a unified
    streaming/batch evaluation engine.

    The final decision ("pred_ensemble") is produced by a stacked meta-classifier
    (cost-sensitive Logistic Regression) over the three detectors' continuous scores,
    not a hard majority vote. Hard voting discards each detector's confidence and
    treats a borderline signal the same as a decisive one; in particular it under-weights
    the physics ThresholdDetector, which alone reaches ~97% recall on AI4I failures because
    its rules are derived from the same physical mechanisms (HDF/PWF/OSF/TWF) used to
    generate the dataset's failure label. `fit_meta()` learns the combination weights and
    tunes the decision threshold for F-beta (beta=2 favors recall) on a validation split
    that must be disjoint from the holdout test set. "pred_majority_vote" (>=2 of 3) and
    "pred_any" (OR of all three, a high-recall safety net) are retained as columns for
    comparison against the learned ensemble.
    """

    SCORE_COLS = ["score_threshold", "score_zscore", "score_iforest"]

    def __init__(
        self,
        threshold_detector: ThresholdDetector,
        zscore_detector: ZScoreDetector,
        iforest_detector: IsolationForestDetector
    ):
        self.threshold_det = threshold_detector
        self.zscore_det = zscore_detector
        self.iforest_det = iforest_detector
        self.meta_scaler = None
        self.meta_model = None
        self.meta_threshold: float = 0.5
        self.meta_fitted = False

    def fit_meta(
        self,
        val_df: pd.DataFrame,
        label_col: str = "machine_failure",
        failure_cost_ratio: float = 5.0,
        f_beta: float = 2.0
    ) -> "UnifiedPipelineDetector":
        """
        Fits the stacking meta-model on a labeled validation split (base detectors must
        already be fitted). Convenience wrapper around fit_meta_from_scores() for callers
        that just have a raw (unscored) validation DataFrame rather than pre-computed
        out-of-fold scores; see fit_meta_from_scores() for the recommended cross-validated
        alternative, which exposes the meta-model to far more failure examples.
        """
        scored = self._score_base_detectors(val_df)
        return self.fit_meta_from_scores(
            scored, label_col=label_col, failure_cost_ratio=failure_cost_ratio, f_beta=f_beta
        )

    def fit_meta_from_scores(
        self,
        scored_df: pd.DataFrame,
        label_col: str = "machine_failure",
        failure_cost_ratio: float = 5.0,
        f_beta: float = 2.0
    ) -> "UnifiedPipelineDetector":
        """
        Fits the stacking meta-model directly from a DataFrame already containing
        SCORE_COLS + label_col. This is the core fitting routine; it is split out from
        fit_meta() so callers can pass leak-free out-of-fold (OOF) scores gathered via
        K-fold cross-validation across the whole training split (see
        train_offline.build_oof_meta_features), rather than scores from a single
        held-out slice. More OOF rows means more failure examples inform both the
        Logistic Regression fit and the F-beta threshold sweep, which otherwise risk being
        tuned off a small, high-variance sample.

        class_weight biases the fit toward recall; the decision threshold is swept over
        [0.01, 0.99] to maximize F-beta rather than assuming P>=0.5.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import fbeta_score

        X = scored_df[self.SCORE_COLS].values
        y = scored_df[label_col].astype(int).values

        self.meta_scaler = StandardScaler().fit(X)
        X_scaled = self.meta_scaler.transform(X)

        self.meta_model = LogisticRegression(
            class_weight={0: 1.0, 1: failure_cost_ratio},
            max_iter=2000
        ).fit(X_scaled, y)

        probs = self.meta_model.predict_proba(X_scaled)[:, 1]

        best_threshold, best_fbeta = 0.5, -1.0
        for t in np.arange(0.01, 1.00, 0.01):
            preds = (probs >= t).astype(int)
            score = fbeta_score(y, preds, beta=f_beta, zero_division=0)
            if score > best_fbeta:
                best_fbeta, best_threshold = score, float(t)

        self.meta_threshold = best_threshold
        self.meta_fitted = True
        print(
            f"  Meta-ensemble calibrated: threshold={self.meta_threshold:.2f}, "
            f"F{f_beta:.0f}={best_fbeta:.4f} on {len(y)} records "
            f"({int(y.sum())} failures)"
        )
        return self

    def _score_base_detectors(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.threshold_det.predict_batch(df)
        out = self.zscore_det.predict_batch(out)
        out = self.iforest_det.predict_batch(out)
        return out

    def _apply_meta(self, scored: pd.DataFrame) -> pd.DataFrame:
        votes = scored["pred_threshold"] + scored["pred_zscore"] + scored["pred_iforest"]
        scored["pred_majority_vote"] = (votes >= 2).astype(int)
        scored["pred_any"] = (votes >= 1).astype(int)

        if self.meta_fitted:
            X = scored[self.SCORE_COLS].values
            X_scaled = self.meta_scaler.transform(X)
            probs = self.meta_model.predict_proba(X_scaled)[:, 1]
            scored["score_ensemble"] = probs
            scored["pred_ensemble"] = (probs >= self.meta_threshold).astype(int)
        else:
            # No meta-model fitted/loaded yet: fall back to majority vote.
            scored["score_ensemble"] = votes / 3.0
            scored["pred_ensemble"] = scored["pred_majority_vote"]
        return scored

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates all detectors on a single event."""
        res_t = self.threshold_det.predict_record(record)
        res_z = self.zscore_det.predict_record(record)
        res_i = self.iforest_det.predict_record(record)

        combined = record.copy()
        combined.update(res_t)
        combined.update(res_z)
        combined.update(res_i)

        votes = res_t["pred_threshold"] + res_z["pred_zscore"] + res_i["pred_iforest"]
        combined["pred_majority_vote"] = 1 if votes >= 2 else 0
        combined["pred_any"] = 1 if votes >= 1 else 0

        if self.meta_fitted:
            x = np.array([[res_t["score_threshold"], res_z["score_zscore"], res_i["score_iforest"]]])
            x_scaled = self.meta_scaler.transform(x)
            prob = float(self.meta_model.predict_proba(x_scaled)[0, 1])
            combined["score_ensemble"] = prob
            combined["pred_ensemble"] = 1 if prob >= self.meta_threshold else 0
        else:
            combined["score_ensemble"] = votes / 3.0
            combined["pred_ensemble"] = combined["pred_majority_vote"]

        return combined

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates all detectors on a DataFrame."""
        scored = self._score_base_detectors(df)
        return self._apply_meta(scored)

    def save_meta(self, path: Path) -> None:
        """Persists the fitted meta-model so streaming/offline consumers can reload it."""
        if not self.meta_fitted:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self.meta_scaler, "model": self.meta_model, "threshold": self.meta_threshold},
            path
        )

    def load_meta(self, path: Path) -> "UnifiedPipelineDetector":
        """Loads a previously fitted meta-model; leaves majority-vote fallback if absent."""
        path = Path(path)
        if path.exists():
            bundle = joblib.load(path)
            self.meta_scaler = bundle["scaler"]
            self.meta_model = bundle["model"]
            self.meta_threshold = bundle["threshold"]
            self.meta_fitted = True
        return self
