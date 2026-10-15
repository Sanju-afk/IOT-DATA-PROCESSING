"""
Integration test for the Hive warehouse sink: scored micro-batches land in a partitioned
Hive table and are queryable through the downstream views.
"""

import math

import pandas as pd
import pytest
from pyspark.sql import SparkSession

import src.warehouse as warehouse
from src.config import HIVE_DATABASE, HIVE_SCORED_TABLE, RAW_DATA_PATH, RAW_COLUMN_MAP
from src.streaming.spark_stream_processor import load_offline_detectors


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    root = tmp_path_factory.mktemp("hive")
    mp = pytest.MonkeyPatch()
    mp.setattr(warehouse, "HIVE_WAREHOUSE_DIR", str(root / "warehouse"))
    mp.setattr(warehouse, "HIVE_METASTORE_DB_DIR", root / "metastore")
    mp.setattr(warehouse, "HIVE_METASTORE_URIS", "")
    session = warehouse.configure_hive(
        SparkSession.builder.appName("test-hive-warehouse").master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    ).getOrCreate()
    warehouse.ensure_tables(session)
    yield session
    session.stop()
    mp.undo()


def scored_batch(start_row: int, event_times: list) -> pd.DataFrame:
    """Builds a micro-batch shaped exactly like MicroBatchProcessor's scored_pdf."""
    raw = pd.read_csv(RAW_DATA_PATH).rename(columns=RAW_COLUMN_MAP)
    pdf = raw.iloc[start_row:start_row + len(event_times)].reset_index(drop=True)
    pdf["timestamp"] = event_times
    pdf["event_seq"] = range(start_row, start_row + len(pdf))
    pdf["kafka_timestamp"] = pd.to_datetime(event_times, utc=True).tz_localize(None)
    pdf["temp_diff"] = pdf["process_temperature"] - pdf["air_temperature"]
    pdf["power_watts"] = pdf["torque"] * pdf["rotational_speed"] * (2.0 * math.pi / 60.0)
    pdf["overstrain_val"] = pdf["tool_wear"] * pdf["torque"]
    scored = load_offline_detectors().predict_batch(pdf)
    scored["processed_timestamp"] = "2026-10-02T00:00:05+00:00"
    scored["latency_ms"] = 5000.0
    return scored


def test_batches_land_in_partitioned_table_and_views(spark):
    batch0 = scored_batch(0, ["2026-10-01T23:59:58+00:00"] * 150)
    batch1 = scored_batch(150, ["2026-10-02T00:00:01+00:00"] * 100)
    # Inject one guaranteed anomaly so the anomalies view is exercised regardless of model output.
    batch1.loc[0, "pred_ensemble"] = 1

    warehouse.append_scored_batch(spark, batch0, batch_id=0)
    warehouse.append_scored_batch(spark, batch1, batch_id=1)

    partitions = [r[0] for r in spark.sql(f"SHOW PARTITIONS {HIVE_SCORED_TABLE}").collect()]
    assert partitions == ["event_date=2026-10-01", "event_date=2026-10-02"]

    per_batch = {
        r["batch_id"]: r["n"]
        for r in spark.sql(f"SELECT batch_id, COUNT(*) AS n FROM {HIVE_SCORED_TABLE} GROUP BY batch_id").collect()
    }
    assert per_batch == {0: 150, 1: 100}

    row = spark.sql(
        f"SELECT *, CAST(event_time AS STRING) AS event_time_utc "
        f"FROM {HIVE_SCORED_TABLE} WHERE batch_id = 1 AND event_seq = 150"
    ).first()
    assert row["udi"] == batch1.loc[0, "udi"]
    assert row["pred_ensemble"] == 1
    assert row["event_time_utc"] == "2026-10-02 00:00:01"
    assert row["score_ensemble"] == pytest.approx(batch1.loc[0, "score_ensemble"])

    expected_anomalies = int(batch0["pred_ensemble"].sum() + batch1["pred_ensemble"].sum())
    assert spark.table(f"{HIVE_DATABASE}.anomalies").count() == expected_anomalies

    summary = spark.sql(
        f"SELECT SUM(events) AS events, SUM(flagged_ensemble) AS flagged "
        f"FROM {HIVE_DATABASE}.hourly_detector_summary"
    ).first()
    assert summary["events"] == 250
    assert summary["flagged"] == expected_anomalies
