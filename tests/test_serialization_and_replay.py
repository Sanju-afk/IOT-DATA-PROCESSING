"""
Unit tests for synthetic generator determinism, JSON serialization, and model persistence.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.synthetic_generator import IndustrialTelemetryGenerator
from src.models.detectors import ThresholdDetector, ZScoreDetector


def test_synthetic_generator_determinism():
    gen1 = IndustrialTelemetryGenerator(seed=123)
    rec1 = gen1.generate_record(1)

    gen2 = IndustrialTelemetryGenerator(seed=123)
    rec2 = gen2.generate_record(1)

    assert rec1["air_temperature"] == rec2["air_temperature"]
    assert rec1["rotational_speed"] == rec2["rotational_speed"]
    assert rec1["torque"] == rec2["torque"]
    assert rec1["machine_failure"] == rec2["machine_failure"]


def test_detector_serialization(tmp_path: Path):
    t_path = tmp_path / "thresholds.json"
    z_path = tmp_path / "zscore.json"

    # Threshold save & load
    t_det = ThresholdDetector()
    t_det.save(t_path)
    assert t_path.exists()

    t_loaded = ThresholdDetector()
    t_loaded.load(t_path)
    assert t_loaded.thresholds["hdf_temp_diff_min"] == 8.6

    # Z-Score save & load
    z_det = ZScoreDetector()
    z_det.means = {"torque": 40.0}
    z_det.stds = {"torque": 10.0}
    z_det.fitted = True
    z_det.save(z_path)
    assert z_path.exists()

    z_loaded = ZScoreDetector()
    z_loaded.load(z_path)
    assert z_loaded.means["torque"] == 40.0
