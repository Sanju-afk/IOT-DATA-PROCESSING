"""
Offline training script for Industrial IoT Anomaly Detectors.
Fits Domain Thresholds, Z-Score reference statistics, and Isolation Forest
strictly on the training split, ensuring zero data leakage. Evaluates on holdout test set.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import (
    MODELS_DIR,
    OUTPUT_DIR,
    IFOREST_MODEL_PATH,
    SCALER_MODEL_PATH,
    DETECTOR_METADATA_PATH,
    ENSEMBLE_META_PATH,
    ENSEMBLE_CV_FOLDS,
    ENSEMBLE_FAILURE_COST_RATIO,
    ENSEMBLE_F_BETA,
    RANDOM_SEED
)
from src.data_loader import (
    download_ai4i_dataset,
    load_and_preprocess_dataset,
    get_train_test_split
)
from src.models.detectors import (
    ThresholdDetector,
    ZScoreDetector,
    IsolationForestDetector,
    UnifiedPipelineDetector
)
from src.evaluation import compare_detectors, format_markdown_metrics_table


def build_oof_meta_features(
    train_df: pd.DataFrame,
    n_folds: int = ENSEMBLE_CV_FOLDS,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates leak-free out-of-fold (OOF) detector scores for every row of train_df, for
    fitting the ensemble meta-model. For each fold, fresh Threshold/Z-Score/IsolationForest
    detectors are fit on the other folds' data (Z-Score/IsolationForest on their normal-only
    rows, matching the production fitting procedure) and used to score the held-out fold, so
    every row's scores come from detectors that never saw that row. This mirrors how
    sklearn's StackingClassifier(cv=...) generates meta-features, and exposes the meta-model
    to every training-split failure exactly once, instead of only the failures that happened
    to land in a single held-out validation slice.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof_parts = []

    for fold_idx, (fit_idx, holdout_idx) in enumerate(skf.split(train_df, train_df["machine_failure"]), start=1):
        fold_fit_df = train_df.iloc[fit_idx]
        fold_holdout_df = train_df.iloc[holdout_idx]
        fold_fit_normal = fold_fit_df[fold_fit_df["machine_failure"] == 0]

        fold_threshold_det = ThresholdDetector().fit(fold_fit_df)
        fold_zscore_det = ZScoreDetector().fit(fold_fit_normal)
        fold_iforest_det = IsolationForestDetector().fit(fold_fit_normal)

        scored_fold = fold_threshold_det.predict_batch(fold_holdout_df)
        scored_fold = fold_zscore_det.predict_batch(scored_fold)
        scored_fold = fold_iforest_det.predict_batch(scored_fold)

        n_fail = int(scored_fold["machine_failure"].sum())
        print(f"  Fold {fold_idx}/{n_folds}: scored {len(scored_fold)} OOF rows ({n_fail} failures)")
        oof_parts.append(scored_fold[UnifiedPipelineDetector.SCORE_COLS + ["machine_failure"]])

    oof_df = pd.concat(oof_parts, ignore_index=True)
    print(f"  Total OOF meta-features: {len(oof_df)} rows ({int(oof_df['machine_failure'].sum())} failures)")
    return oof_df


def train_and_evaluate_offline():
    print("=" * 60)
    print("STEP 1: Data Acquisition & Preprocessing")
    print("=" * 60)
    csv_path = download_ai4i_dataset()
    df = load_and_preprocess_dataset(csv_path)
    train_df, test_df = get_train_test_split(df, test_size=0.20, random_seed=RANDOM_SEED)

    print(f"Total dataset records: {len(df)}")
    print(f"Training split:        {len(train_df)} ({train_df['machine_failure'].sum()} failures)")
    print(f"Holdout test split:    {len(test_df)} ({test_df['machine_failure'].sum()} failures)")

    # Strictly separate normal operational samples for unsupervised training
    train_normal = train_df[train_df["machine_failure"] == 0].copy()
    print(f"Normal training samples used for model fitting: {len(train_normal)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 2: Fitting Anomaly Detectors on Training Split")
    print("=" * 60)

    # Method A: Domain Threshold Detector
    print("[1/3] Calibrating Threshold Detector...")
    threshold_detector = ThresholdDetector()
    threshold_detector.fit(train_df)
    thresholds_path = MODELS_DIR / "thresholds.json"
    threshold_detector.save(thresholds_path)
    print(f"  Threshold configuration saved to {thresholds_path}")

    # Method B: Standardized Z-Score Detector
    print("[2/3] Fitting Z-Score Baseline on Normal Training Samples...")
    zscore_detector = ZScoreDetector()
    zscore_detector.fit(train_normal)
    zscore_path = MODELS_DIR / "zscore_params.json"
    zscore_detector.save(zscore_path)
    print(f"  Z-Score parameters saved to {zscore_path}")

    # Method C: Multivariate Isolation Forest
    print("[3/3] Training Isolation Forest on Normal Training Samples...")
    iforest_detector = IsolationForestDetector()
    iforest_detector.fit(train_normal)
    iforest_detector.save(
        model_path=IFOREST_MODEL_PATH,
        scaler_path=SCALER_MODEL_PATH,
        meta_path=DETECTOR_METADATA_PATH
    )
    print(f"  Isolation Forest model saved to {IFOREST_MODEL_PATH}")
    print(f"  StandardScaler saved to {SCALER_MODEL_PATH}")
    print(f"  Decision threshold calibrated at: {iforest_detector.decision_threshold:.4f}")

    print("\n" + "=" * 60)
    print(f"STEP 3: Generating {ENSEMBLE_CV_FOLDS}-Fold Out-of-Fold Meta-Features")
    print("=" * 60)
    oof_df = build_oof_meta_features(train_df, n_folds=ENSEMBLE_CV_FOLDS, random_seed=RANDOM_SEED)

    print("\n" + "=" * 60)
    print("STEP 4: Fitting Stacked Ensemble Meta-Model on OOF Features")
    print("=" * 60)
    # The deployed base detectors above were fit on the full training split (more data,
    # better calibrated); the meta-model is fit separately on the cross-validated OOF
    # scores so its threshold reflects genuinely unseen predictions rather than in-sample ones.
    unified = UnifiedPipelineDetector(threshold_detector, zscore_detector, iforest_detector)
    unified.fit_meta_from_scores(
        oof_df,
        label_col="machine_failure",
        failure_cost_ratio=ENSEMBLE_FAILURE_COST_RATIO,
        f_beta=ENSEMBLE_F_BETA
    )
    unified.save_meta(ENSEMBLE_META_PATH)
    print(f"  Ensemble meta-model saved to {ENSEMBLE_META_PATH}")

    print("\n" + "=" * 60)
    print("STEP 5: Holdout Evaluation on Test Split (2000 Records)")
    print("=" * 60)
    evaluated_test_df = unified.predict_batch(test_df)

    # Evaluate metrics: compare the learned ensemble against the base detectors and the
    # two cheap hand-coded combination strategies (strict majority vote, OR-based safety net).
    metrics_df = compare_detectors(
        evaluated_test_df,
        label_col="machine_failure",
        detectors=["threshold", "zscore", "iforest", "majority_vote", "any", "ensemble"]
    )

    print("\n" + format_markdown_metrics_table(metrics_df))

    # Save evaluation outputs
    metrics_path = OUTPUT_DIR / "offline_test_metrics.json"
    predictions_path = OUTPUT_DIR / "offline_test_predictions.csv"

    metrics_df.to_json(metrics_path, orient="records", indent=2)
    evaluated_test_df.to_csv(predictions_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved holdout test predictions to {predictions_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate_offline()
