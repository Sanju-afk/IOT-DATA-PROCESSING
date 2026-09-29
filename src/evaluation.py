"""
Evaluation metrics module for Industrial IoT Anomaly Detection.
Calculates Confusion Matrix, Precision, Recall, F1 Score, False Positive Rate (FPR),
Specificity, ROC-AUC, and PR-AUC. Explains the impact of extreme class imbalance.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    detector_name: str = "Detector"
) -> Dict[str, Any]:
    """
    Computes comprehensive anomaly detection metrics for binary classification.
    y_true: 0 = Normal, 1 = Machine Failure / Anomaly
    y_pred: 0 = Predicted Normal, 1 = Predicted Anomaly
    y_scores: continuous anomaly score (optional, needed for AUC metrics)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    total_samples = len(y_true)
    positive_samples = int(np.sum(y_true))
    negative_samples = total_samples - positive_samples
    prevalence = positive_samples / max(total_samples, 1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    fpr = float(fp / negative_samples) if negative_samples > 0 else 0.0
    specificity = float(tn / negative_samples) if negative_samples > 0 else 1.0

    roc_auc = None
    pr_auc = None
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
            pr_auc = float(average_precision_score(y_true, y_scores))
        except Exception:
            pass

    return {
        "detector": detector_name,
        "total_samples": int(total_samples),
        "true_positives (TP)": int(tp),
        "false_positives (FP)": int(fp),
        "true_negatives (TN)": int(tn),
        "false_negatives (FN)": int(fn),
        "class_prevalence": round(float(prevalence), 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "specificity": round(specificity, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A"
    }


def compare_detectors(
    test_df: pd.DataFrame,
    label_col: str = "machine_failure",
    detectors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Evaluates multiple detectors against ground truth failure labels on holdout test set.
    """
    if detectors is None:
        detectors = ["threshold", "zscore", "iforest", "ensemble"]

    y_true = test_df[label_col].values
    results = []

    for det in detectors:
        pred_col = f"pred_{det}"
        score_col = f"score_{det}"

        if pred_col not in test_df.columns:
            continue

        y_pred = test_df[pred_col].values
        y_scores = test_df[score_col].values if score_col in test_df.columns else None

        metrics = compute_anomaly_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            detector_name=det.capitalize()
        )
        results.append(metrics)

    res_df = pd.DataFrame(results)
    return res_df


def format_markdown_metrics_table(res_df: pd.DataFrame) -> str:
    """Formats metrics DataFrame into clean GitHub-flavored markdown."""
    cols_to_show = [
        "detector",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
        "true_positives (TP)",
        "false_positives (FP)",
        "false_negatives (FN)"
    ]
    sub_df = res_df[[c for c in cols_to_show if c in res_df.columns]]
    return sub_df.to_markdown(index=False)
