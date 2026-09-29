"""
Unit tests for evaluation metrics (Precision, Recall, F1, FPR, ROC-AUC, PR-AUC).
"""

import numpy as np
import pytest

from src.evaluation import compute_anomaly_metrics


def test_compute_anomaly_metrics_perfect():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9])

    metrics = compute_anomaly_metrics(y_true, y_pred, y_scores=scores, detector_name="Perfect")
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["roc_auc"] == 1.0


def test_compute_anomaly_metrics_zero_division():
    # Model predicts all 0s (no positives detected)
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])

    metrics = compute_anomaly_metrics(y_true, y_pred, detector_name="AllZero")
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["accuracy"] == 0.75  # 75% accuracy despite missing all anomalies
