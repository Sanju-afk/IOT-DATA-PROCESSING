"""
Unit tests for data schema validation, clean column mapping, and physics feature engineering.
"""

import math
import pytest
import pandas as pd
import numpy as np

from src.data_loader import compute_derived_features
from src.config import NUMERIC_FEATURES, DERIVED_FEATURES


def test_compute_derived_features():
    sample_data = {
        "air_temperature": [300.0, 305.0],
        "process_temperature": [310.0, 313.0],
        "rotational_speed": [1500, 1200],
        "torque": [40.0, 60.0],
        "tool_wear": [100, 210]
    }
    df = pd.DataFrame(sample_data)
    enriched = compute_derived_features(df)

    # 1. Check temp_diff
    assert "temp_diff" in enriched.columns
    assert enriched["temp_diff"].iloc[0] == pytest.approx(10.0, rel=1e-3)
    assert enriched["temp_diff"].iloc[1] == pytest.approx(8.0, rel=1e-3)

    # 2. Check power_watts = torque * (rotational_speed * 2 * pi / 60)
    expected_power_0 = 40.0 * 1500.0 * (2.0 * math.pi / 60.0)
    assert enriched["power_watts"].iloc[0] == pytest.approx(expected_power_0, rel=1e-3)

    # 3. Check overstrain_val = tool_wear * torque
    expected_overstrain_1 = 210 * 60.0
    assert enriched["overstrain_val"].iloc[1] == pytest.approx(expected_overstrain_1, rel=1e-3)


def test_null_and_boundary_handling():
    sample_data = {
        "air_temperature": [np.nan, 300.0],
        "process_temperature": [310.0, np.nan],
        "rotational_speed": [1500, 1400],
        "torque": [40.0, 45.0],
        "tool_wear": [0, 50]
    }
    df = pd.DataFrame(sample_data)
    enriched = compute_derived_features(df)
    assert len(enriched) == 2
    assert pd.isna(enriched["temp_diff"].iloc[0])
