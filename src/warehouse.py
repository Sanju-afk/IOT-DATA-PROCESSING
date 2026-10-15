"""
Hive warehouse sink for scored stream results.

Every Spark micro-batch is appended to the Hive table `iot_analytics.telemetry_scored`
(Parquet, partitioned by event_date), so downstream consumers can query detector output
with SQL instead of globbing per-batch CSV files. The warehouse lives on the local
filesystem by default and moves to HDFS by pointing HIVE_WAREHOUSE_DIR at an hdfs:// URI
(see src/config.py).

Downstream querying:
    python -m src.warehouse --report
    python -m src.warehouse --sql "SELECT product_type, COUNT(*) FROM iot_analytics.anomalies GROUP BY 1"

The default metastore is embedded Derby, which allows a single JVM at a time: query it
after the streaming job has stopped, or run the shared Hive Metastore for concurrent access.
"""

import argparse

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import (
    HIVE_DATABASE,
    HIVE_METASTORE_DB_DIR,
    HIVE_METASTORE_URIS,
    HIVE_SCORED_TABLE,
    HIVE_WAREHOUSE_DIR,
)

# (warehouse column, Hive type, column in the scored pandas micro-batch)
SCORED_COLUMNS = [
    ("udi", "INT", "udi"),
    ("product_id", "STRING", "product_id"),
    ("product_type", "STRING", "product_type"),
    ("air_temperature", "DOUBLE", "air_temperature"),
    ("process_temperature", "DOUBLE", "process_temperature"),
    ("rotational_speed", "INT", "rotational_speed"),
    ("torque", "DOUBLE", "torque"),
    ("tool_wear", "INT", "tool_wear"),
    ("machine_failure", "INT", "machine_failure"),
    ("twf", "INT", "twf"),
    ("hdf", "INT", "hdf"),
    ("pwf", "INT", "pwf"),
    ("osf", "INT", "osf"),
    ("rnf", "INT", "rnf"),
    ("event_seq", "BIGINT", "event_seq"),
    ("event_time", "TIMESTAMP", "timestamp"),
    ("kafka_time", "TIMESTAMP", "kafka_timestamp"),
    ("processed_time", "TIMESTAMP", "processed_timestamp"),
    ("latency_ms", "DOUBLE", "latency_ms"),
    ("temp_diff", "DOUBLE", "temp_diff"),
    ("power_watts", "DOUBLE", "power_watts"),
    ("overstrain_val", "DOUBLE", "overstrain_val"),
    ("pred_threshold", "INT", "pred_threshold"),
    ("score_threshold", "DOUBLE", "score_threshold"),
    ("reason_threshold", "STRING", "reason_threshold"),
    ("pred_zscore", "INT", "pred_zscore"),
    ("score_zscore", "DOUBLE", "score_zscore"),
    ("max_abs_z", "DOUBLE", "max_abs_z"),
    ("pred_iforest", "INT", "pred_iforest"),
    ("score_iforest", "DOUBLE", "score_iforest"),
    ("iforest_threshold", "DOUBLE", "iforest_threshold"),
    ("pred_majority_vote", "INT", "pred_majority_vote"),
    ("pred_any", "INT", "pred_any"),
    ("score_ensemble", "DOUBLE", "score_ensemble"),
    ("pred_ensemble", "INT", "pred_ensemble"),
    ("batch_id", "BIGINT", None),
]
PARTITION_COLUMN = ("event_date", "DATE")

VIEWS = {
    f"{HIVE_DATABASE}.anomalies": f"""
        SELECT * FROM {HIVE_SCORED_TABLE} WHERE pred_ensemble = 1
    """,
    f"{HIVE_DATABASE}.hourly_detector_summary": f"""
        SELECT
            date_trunc('HOUR', event_time) AS event_hour,
            product_type,
            COUNT(*)                AS events,
            SUM(machine_failure)    AS labeled_failures,
            SUM(pred_threshold)     AS flagged_threshold,
            SUM(pred_zscore)        AS flagged_zscore,
            SUM(pred_iforest)       AS flagged_iforest,
            SUM(pred_ensemble)      AS flagged_ensemble,
            SUM(CASE WHEN pred_ensemble = 1 AND machine_failure = 1 THEN 1 ELSE 0 END) AS ensemble_true_positives,
            ROUND(AVG(latency_ms), 1) AS avg_latency_ms
        FROM {HIVE_SCORED_TABLE}
        GROUP BY date_trunc('HOUR', event_time), product_type
    """,
}

REPORT_QUERIES = {
    "Partitions": f"SHOW PARTITIONS {HIVE_SCORED_TABLE}",
    "Rows and anomalies per partition": f"""
        SELECT event_date, COUNT(*) AS events, SUM(pred_ensemble) AS anomalies,
               SUM(machine_failure) AS labeled_failures
        FROM {HIVE_SCORED_TABLE} GROUP BY event_date ORDER BY event_date
    """,
    "Hourly detector summary": f"SELECT * FROM {HIVE_DATABASE}.hourly_detector_summary ORDER BY event_hour, product_type",
    "Top 10 ensemble anomalies": f"""
        SELECT event_time, udi, product_id, product_type, ROUND(score_ensemble, 4) AS score_ensemble,
               reason_threshold, machine_failure
        FROM {HIVE_DATABASE}.anomalies ORDER BY score_ensemble DESC LIMIT 10
    """,
}


def configure_hive(builder: SparkSession.Builder) -> SparkSession.Builder:
    """Points a SparkSession builder at the project's Hive metastore and warehouse."""
    builder = (
        builder
        .enableHiveSupport()
        .config("spark.sql.warehouse.dir", HIVE_WAREHOUSE_DIR)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.hive.exec.dynamic.partition", "true")
        .config("spark.hadoop.hive.exec.dynamic.partition.mode", "nonstrict")
    )
    if HIVE_METASTORE_URIS:
        return builder.config("spark.hadoop.hive.metastore.uris", HIVE_METASTORE_URIS)

    HIVE_METASTORE_DB_DIR.mkdir(parents=True, exist_ok=True)
    return (
        builder
        .config(
            "spark.hadoop.javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={HIVE_METASTORE_DB_DIR / 'metastore_db'};create=true",
        )
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={HIVE_METASTORE_DB_DIR}")
    )


def ensure_tables(spark: SparkSession) -> None:
    """Creates the database, scored-telemetry table, and query views if absent."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DATABASE}")
    columns = ",\n    ".join(f"{name} {hive_type}" for name, hive_type, _ in SCORED_COLUMNS)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {HIVE_SCORED_TABLE} (
            {columns}
        )
        PARTITIONED BY ({PARTITION_COLUMN[0]} {PARTITION_COLUMN[1]})
        STORED AS PARQUET
    """)
    for view_name, query in VIEWS.items():
        spark.sql(f"CREATE OR REPLACE VIEW {view_name} AS {query}")


def to_warehouse_frame(spark: SparkSession, scored_pdf: pd.DataFrame, batch_id: int) -> DataFrame:
    """Casts a scored pandas micro-batch to the table's column order and types."""
    sources = [src for _, _, src in SCORED_COLUMNS if src is not None]
    pdf = scored_pdf.reindex(columns=sources)
    # Spark's Arrow conversion cannot infer a type for an all-null object column.
    for col in pdf.columns:
        if pdf[col].isna().all():
            pdf[col] = pdf[col].astype("float64")

    df = spark.createDataFrame(pdf)
    selected = [
        (F.lit(batch_id) if src is None else F.col(f"`{src}`")).cast(hive_type).alias(name)
        for name, hive_type, src in SCORED_COLUMNS
    ]
    selected.append(F.to_date(F.col("`timestamp`").cast("timestamp")).alias(PARTITION_COLUMN[0]))
    return df.select(*selected)


def append_scored_batch(spark: SparkSession, scored_pdf: pd.DataFrame, batch_id: int) -> None:
    """Appends one scored micro-batch to the Hive table (one Parquet file per date partition)."""
    frame = to_warehouse_frame(spark, scored_pdf, batch_id)
    frame.repartition(PARTITION_COLUMN[0]).write.insertInto(HIVE_SCORED_TABLE)


def main():
    parser = argparse.ArgumentParser(description="Query the Hive warehouse of scored IoT telemetry")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sql", help="Run an ad-hoc Spark SQL / HiveQL query")
    group.add_argument("--report", action="store_true", help="Print canned downstream summary queries")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print per query")
    args = parser.parse_args()

    spark = configure_hive(SparkSession.builder.appName("IoT-Warehouse-Query").master("local[*]")).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        ensure_tables(spark)
        queries = {"Query": args.sql} if args.sql else REPORT_QUERIES
        for title, query in queries.items():
            print(f"\n=== {title} ===")
            spark.sql(query).show(args.limit, truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
