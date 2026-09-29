"""
Configuration settings for the Industrial IoT Anomaly Detection Pipeline.
Centralizes Kafka settings, file paths, feature definitions, and detector thresholds.
"""

from pathlib import Path
import os

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"

# Model and Artifact Paths
MODELS_DIR = BASE_DIR / "models"
IFOREST_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "scaler.joblib"
DETECTOR_METADATA_PATH = MODELS_DIR / "detector_metadata.json"

# Output Paths
OUTPUT_DIR = BASE_DIR / "output"
STREAM_OUTPUT_DIR = OUTPUT_DIR / "stream_results"
STREAM_PARQUET_DIR = STREAM_OUTPUT_DIR / "parquet"
STREAM_CSV_DIR = STREAM_OUTPUT_DIR / "csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmarks"

# Kafka Settings
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TELEMETRY = "iot_telemetry"
TOPIC_ANOMALIES = "iot_anomalies"
TOPIC_AGGREGATES = "iot_aggregates"
KAFKA_PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))

# Schema and Feature Definitions
# Raw CSV Column names (handles optional BOM in 'UDI')
RAW_COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "product_type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf"
}

NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]

DERIVED_FEATURES = [
    "temp_diff",    # process_temperature - air_temperature
    "power_watts",  # torque * (2 * pi / 60) * rotational_speed
    "overstrain_val"# tool_wear * torque
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + DERIVED_FEATURES

LABEL_COL = "machine_failure"
FAILURE_TYPE_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]

# Reproducibility Seeds
RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test

# Isolation Forest Hyperparameters
IFOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.035,  # Matches the ~3.39% empirical machine failure rate in AI4I
    "max_samples": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Physical Domain Thresholds (Based on Matzka 2020 AI4I specification)
DOMAIN_THRESHOLDS = {
    # Temperature differences < 8.6K with speed < 1380 rpm cause Heat Dissipation Failure (HDF)
    "hdf_temp_diff_min": 8.6,
    "hdf_speed_max": 1380.0,
    # Power bounds (PWF): Power = Torque (Nm) * Speed (rad/s) = Torque * Speed * 2*pi / 60
    # Normal power is between 3500 W and 9000 W
    "power_min_watts": 3500.0,
    "power_max_watts": 9000.0,
    # Tool wear (TWF): Failure occurs between 200 and 240 minutes
    "tool_wear_max_min": 200.0,
    # Overstrain product (OSF): Tool wear * Torque
    "overstrain_limits": {
        "L": 11000.0,  # 11,000 min*Nm for Low quality variant
        "M": 12000.0,  # 12,000 min*Nm for Medium variant
        "H": 13000.0   # 13,000 min*Nm for High variant
    },
    # Univariate baseline extremes
    "air_temp_high": 304.5,
    "process_temp_high": 313.8,
    "rotational_speed_low": 1200.0,
    "rotational_speed_high": 2700.0,
    "torque_high": 68.0,
    "torque_low": 12.0
}

# Z-score detector settings
ZSCORE_THRESHOLD = 3.0  # 3 standard deviations
