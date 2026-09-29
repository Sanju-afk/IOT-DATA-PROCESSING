"""
Unit tests for Anomaly Detectors (Threshold, ZScore, Isolation Forest, and Unified Pipeline).
"""

import pytest
import pandas as pd
import numpy as np

from src.models.detectors import (
    ThresholdDetector,
    ZScoreDetector,
    IsolationForestDetector,
    UnifiedPipelineDetector
)


def test_threshold_detector_normal():
    detector = ThresholdDetector()
    normal_record = {
        "air_temperature": 300.0,
        "process_temperature": 310.5,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 50,
        "product_type": "L"
    }
    res = detector.predict_record(normal_record)
    assert res["pred_threshold"] == 0
    assert res["reason_threshold"] == "Normal operation"


def test_threshold_detector_failure_rules():
    detector = ThresholdDetector()

    # Heat Dissipation Failure (HDF): temp_diff < 8.6 & speed < 1380
    hdf_record = {
        "air_temperature": 303.0,
        "process_temperature": 309.0,  # diff = 6.0 < 8.6
        "rotational_speed": 1300,       # speed < 1380
        "torque": 40.0,
        "tool_wear": 20,
        "product_type": "L"
    }
    res_hdf = detector.predict_record(hdf_record)
    assert res_hdf["pred_threshold"] == 1
    assert "HDF" in res_hdf["reason_threshold"]

    # Tool Wear Failure (TWF): tool_wear >= 200 min
    twf_record = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 220,
        "product_type": "L"
    }
    res_twf = detector.predict_record(twf_record)
    assert res_twf["pred_threshold"] == 1
    assert "TWF" in res_twf["reason_threshold"]

    # Power Failure (PWF): Power > 9000 W
    pwf_record = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 2200,
        "torque": 75.0,  # power ~ 17,278 W > 9000 W
        "tool_wear": 50,
        "product_type": "L"
    }
    res_pwf = detector.predict_record(pwf_record)
    assert res_pwf["pred_threshold"] == 1
    assert "PWF" in res_pwf["reason_threshold"]


def test_zscore_detector():
    # Synthetic normal train set
    np.random.seed(42)
    n = 200
    train_df = pd.DataFrame({
        "air_temperature": np.random.normal(300.0, 2.0, n),
        "process_temperature": np.random.normal(310.0, 1.5, n),
        "rotational_speed": np.random.normal(1500.0, 100.0, n),
        "torque": np.random.normal(40.0, 5.0, n),
        "tool_wear": np.random.uniform(0, 150, n)
    })

    detector = ZScoreDetector()
    detector.fit(train_df)

    # Normal sample
    norm_rec = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 75.0
    }
    res_norm = detector.predict_record(norm_rec)
    assert res_norm["pred_zscore"] == 0

    # Extreme outlier (6-sigma torque)
    outlier_rec = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 75.0,  # 7-sigma above mean 40
        "tool_wear": 75.0
    }
    res_out = detector.predict_record(outlier_rec)
    assert res_out["pred_zscore"] == 1
    assert res_out["max_abs_z"] > 3.0


def test_isolation_forest_detector():
    np.random.seed(42)
    n = 300
    train_df = pd.DataFrame({
        "air_temperature": np.random.normal(300.0, 2.0, n),
        "process_temperature": np.random.normal(310.0, 1.5, n),
        "rotational_speed": np.random.normal(1500.0, 100.0, n),
        "torque": np.random.normal(40.0, 5.0, n),
        "tool_wear": np.random.uniform(0, 150, n)
    })

    detector = IsolationForestDetector()
    detector.fit(train_df)
    assert detector.fitted is True

    # Test batch prediction
    batch = train_df.head(10).copy()
    pred_df = detector.predict_batch(batch)
    assert "pred_iforest" in pred_df.columns
    assert "score_iforest" in pred_df.columns
    assert len(pred_df) == 10
