"""
Offline training script for Industrial IoT Anomaly Detectors.
Fits Domain Thresholds, Z-Score reference statistics, and Isolation Forest
strictly on the training split, ensuring zero data leakage. Evaluates on holdout test set.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    MODELS_DIR,
    OUTPUT_DIR,
    IFOREST_MODEL_PATH,
    SCALER_MODEL_PATH,
    DETECTOR_METADATA_PATH,
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
    print("STEP 3: Holdout Evaluation on Test Split (2000 Records)")
    print("=" * 60)
    unified = UnifiedPipelineDetector(threshold_detector, zscore_detector, iforest_detector)
    evaluated_test_df = unified.predict_batch(test_df)

    # Evaluate metrics
    metrics_df = compare_detectors(
        evaluated_test_df,
        label_col="machine_failure",
        detectors=["threshold", "zscore", "iforest", "ensemble"]
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
