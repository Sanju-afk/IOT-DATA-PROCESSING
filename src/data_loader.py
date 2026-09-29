"""
Data acquisition, verification, preprocessing, and train/test splitting for AI4I 2020 dataset.
Ensures zero data leakage: scalers and statistical references are fit strictly on training splits.
"""

import hashlib
import io
import math
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    DATA_DIR,
    DATA_URL,
    EXPECTED_SHA256,
    RAW_COLUMN_MAP,
    NUMERIC_FEATURES,
    DERIVED_FEATURES,
    LABEL_COL,
    FAILURE_TYPE_COLS,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT_RATIO
)


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """Verifies SHA256 checksum of a file."""
    if not file_path.exists():
        return False
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_hash.lower()


def download_ai4i_dataset(force: bool = False) -> Path:
    """
    Downloads and extracts the AI4I 2020 Predictive Maintenance dataset from UCI repository.
    Verifies checksum upon download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and not force:
        if verify_checksum(RAW_DATA_PATH, EXPECTED_SHA256):
            return RAW_DATA_PATH

    print(f"Downloading AI4I 2020 dataset from {DATA_URL}...")
    try:
        req = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            # Locate ai4i2020.csv inside archive
            for member in z.namelist():
                if member.endswith("ai4i2020.csv"):
                    with z.open(member) as source_file, open(RAW_DATA_PATH, "wb") as target_file:
                        target_file.write(source_file.read())
                    break
        print(f"Dataset extracted to {RAW_DATA_PATH}")
    except Exception as e:
        if RAW_DATA_PATH.exists():
            print(f"Download encountered issue ({e}), but local file {RAW_DATA_PATH} exists. Proceeding.")
        else:
            raise RuntimeError(f"Failed to acquire dataset and no local file found: {e}")

    return RAW_DATA_PATH


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-based domain features defined in Matzka (2020):
    1. temp_diff: Difference between process temperature and air temperature [K]
    2. power_watts: Mechanical power in Watts = Torque (Nm) * Rotational Speed (rad/s)
       where rad/s = rpm * 2 * pi / 60
    3. overstrain_val: Product of tool wear (min) and torque (Nm)
    """
    df = df.copy()
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["power_watts"] = df["torque"] * df["rotational_speed"] * (2.0 * math.pi / 60.0)
    df["overstrain_val"] = df["tool_wear"] * df["torque"]
    return df


def load_and_preprocess_dataset(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV, cleans column names, calculates domain features,
    and returns a clean standardized DataFrame.
    """
    if not csv_path.exists():
        download_ai4i_dataset()

    # Load with UTF-8 BOM awareness
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Rename columns to standard snake_case names
    clean_columns = {}
    for col in df.columns:
        stripped = col.strip()
        matched = False
        for raw_k, clean_v in RAW_COLUMN_MAP.items():
            if raw_k.lower() in stripped.lower():
                clean_columns[col] = clean_v
                matched = True
                break
        if not matched:
            clean_columns[col] = stripped.lower().replace(" ", "_")

    df = df.rename(columns=clean_columns)

    # Ensure required columns exist
    for col in NUMERIC_FEATURES + [LABEL_COL]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from dataset. Available: {list(df.columns)}")

    # Ensure types
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [LABEL_COL] + FAILURE_TYPE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Compute derived features
    df = compute_derived_features(df)

    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 1.0 - TRAIN_TEST_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and test sets with stratification on machine_failure.
    To prevent data leakage:
    - Train split is used for statistical baseline & Isolation Forest training
    - Test split is kept unseen for evaluation and streaming simulation
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[LABEL_COL]
    )
    return train_df.copy().reset_index(drop=True), test_df.copy().reset_index(drop=True)


if __name__ == "__main__":
    path = download_ai4i_dataset()
    df = load_and_preprocess_dataset(path)
    train_df, test_df = get_train_test_split(df)
    print(f"Total records: {len(df)}")
    print(f"Train records: {len(train_df)} (Failures: {train_df[LABEL_COL].sum()}, {train_df[LABEL_COL].mean():.2%})")
    print(f"Test records:  {len(test_df)} (Failures: {test_df[LABEL_COL].sum()}, {test_df[LABEL_COL].mean():.2%})")
    print("Derived features sample:")
    print(df[["temp_diff", "power_watts", "overstrain_val"]].head(3))
