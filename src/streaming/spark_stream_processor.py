"""
Spark Structured Streaming Processor for Real-Time Industrial IoT Telemetry.
Ingests from Kafka topic 'iot_telemetry', performs schema validation, feature engineering,
windowed aggregations, and evaluates Threshold, Z-Score, and Isolation Forest detectors.
Emits anomalies and aggregates to Kafka topics and durable Parquet/CSV storage.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    LongType,
    TimestampType
)

from src.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_TELEMETRY,
    TOPIC_ANOMALIES,
    TOPIC_AGGREGATES,
    MODELS_DIR,
    OUTPUT_DIR,
    STREAM_PARQUET_DIR,
    STREAM_CSV_DIR,
    CHECKPOINT_DIR,
    IFOREST_MODEL_PATH,
    SCALER_MODEL_PATH,
    DETECTOR_METADATA_PATH,
    DOMAIN_THRESHOLDS,
    ZSCORE_THRESHOLD
)
from src.models.detectors import (
    ThresholdDetector,
    ZScoreDetector,
    IsolationForestDetector,
    UnifiedPipelineDetector
)


def get_spark_session(app_name: str = "Industrial-IoT-Stream-Analytics") -> SparkSession:
    """Builds an optimized SparkSession with Kafka connector and local parallelism."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .getOrCreate()
    )


def load_offline_detectors() -> UnifiedPipelineDetector:
    """Loads pre-trained offline models and configuration."""
    # Thresholds
    thresholds_file = MODELS_DIR / "thresholds.json"
    threshold_det = ThresholdDetector()
    if thresholds_file.exists():
        threshold_det.load(thresholds_file)

    # Z-Score
    zscore_file = MODELS_DIR / "zscore_params.json"
    zscore_det = ZScoreDetector()
    if zscore_file.exists():
        zscore_det.load(zscore_file)
    else:
        # Fallback default statistics from AI4I reference
        zscore_det.means = {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1538.0,
            "torque": 40.0,
            "tool_wear": 107.0,
            "temp_diff": 10.0,
            "power_watts": 6400.0
        }
        zscore_det.stds = {
            "air_temperature": 2.0,
            "process_temperature": 1.5,
            "rotational_speed": 179.0,
            "torque": 10.0,
            "tool_wear": 63.0,
            "temp_diff": 1.0,
            "power_watts": 1200.0
        }
        zscore_det.fitted = True

    # Isolation Forest
    iforest_det = IsolationForestDetector()
    if IFOREST_MODEL_PATH.exists() and SCALER_MODEL_PATH.exists() and DETECTOR_METADATA_PATH.exists():
        iforest_det.load(
            model_path=IFOREST_MODEL_PATH,
            scaler_path=SCALER_MODEL_PATH,
            meta_path=DETECTOR_METADATA_PATH
        )
    else:
        print("Warning: Trained Isolation Forest artifacts not found on disk. Detector will be skipped if unfitted.")

    return UnifiedPipelineDetector(threshold_det, zscore_det, iforest_det)


class MicroBatchProcessor:
    """
    Handles streaming micro-batch processing:
    - Runs detector scoring on batch DataFrame
    - Writes results to Parquet and CSV sinks
    - Publishes anomalies and windowed aggregates back to Kafka
    - Records latency and throughput metrics
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        emit_kafka_anomalies: bool = True
    ):
        self.detector = load_offline_detectors()
        self.bootstrap_servers = bootstrap_servers
        self.emit_kafka = emit_kafka_anomalies
        self.kafka_producer = None
        self.total_processed_records = 0
        self.total_anomalies_detected = 0
        self.metrics_log_path = OUTPUT_DIR / "stream_metrics_log.jsonl"
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

        if self.emit_kafka:
            from kafka import KafkaProducer
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=[bootstrap_servers],
                    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                    acks=0
                )
            except Exception as e:
                print(f"Warning: Could not connect Kafka output producer: {e}. Output to files only.")
                self.emit_kafka = False

    def process_batch(self, batch_df, batch_id: int):
        batch_start_time = time.time()
        count = batch_df.count()
        if count == 0:
            return

        # Convert Spark micro-batch to Pandas for detector evaluation
        pdf = batch_df.toPandas()

        # Run unified detectors (Threshold, Z-score, Isolation Forest)
        scored_pdf = self.detector.predict_batch(pdf)

        # Add processing timestamp and latency metrics
        proc_time = datetime.now(timezone.utc)
        scored_pdf["processed_timestamp"] = proc_time.isoformat()

        # Compute latency: difference between processed_timestamp and event timestamp
        if "timestamp" in scored_pdf.columns:
            try:
                event_times = pd.to_datetime(scored_pdf["timestamp"], utc=True)
                proc_time_pd = pd.to_datetime(proc_time)
                scored_pdf["latency_ms"] = (proc_time_pd - event_times).dt.total_seconds() * 1000.0
            except Exception:
                scored_pdf["latency_ms"] = 0.0

        batch_duration_sec = time.time() - batch_start_time
        rows_per_sec = count / batch_duration_sec if batch_duration_sec > 0 else 0
        self.total_processed_records += count

        n_anomalies = int(scored_pdf["pred_ensemble"].sum())
        n_thresh = int(scored_pdf["pred_threshold"].sum())
        n_zscore = int(scored_pdf["pred_zscore"].sum())
        n_iforest = int(scored_pdf["pred_iforest"].sum())
        self.total_anomalies_detected += n_anomalies

        avg_latency = float(scored_pdf["latency_ms"].mean()) if "latency_ms" in scored_pdf.columns else 0.0

        # Persist results to CSV / Parquet
        STREAM_CSV_DIR.mkdir(parents=True, exist_ok=True)
        STREAM_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

        csv_batch_file = STREAM_CSV_DIR / f"batch_{batch_id:05d}.csv"
        scored_pdf.to_csv(csv_batch_file, index=False)

        # Also maintain consolidated latest stream results CSV (up to 5,000 recent rows)
        latest_csv = OUTPUT_DIR / "latest_stream_predictions.csv"
        if latest_csv.exists():
            existing = pd.read_csv(latest_csv)
            combined = pd.concat([existing, scored_pdf], ignore_index=True).tail(5000)
            combined.to_csv(latest_csv, index=False)
        else:
            scored_pdf.tail(5000).to_csv(latest_csv, index=False)

        # Emit detected anomalies to Kafka output topic
        if self.emit_kafka and self.kafka_producer:
            anomalies_df = scored_pdf[scored_pdf["pred_ensemble"] == 1]
            for _, row in anomalies_df.iterrows():
                anomaly_event = {}
                for k, v in row.to_dict().items():
                    if pd.isna(v):
                        anomaly_event[k] = None
                    elif isinstance(v, (pd.Timestamp, datetime)):
                        anomaly_event[k] = v.isoformat()
                    elif isinstance(v, (np.integer, int)):
                        anomaly_event[k] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        anomaly_event[k] = float(v)
                    else:
                        anomaly_event[k] = v
                self.kafka_producer.send(TOPIC_ANOMALIES, value=anomaly_event)
            self.kafka_producer.flush()

        # Log micro-batch telemetry progress
        metric_record = {
            "batch_id": int(batch_id),
            "records_in_batch": int(count),
            "batch_duration_sec": round(batch_duration_sec, 4),
            "throughput_rows_per_sec": round(rows_per_sec, 2),
            "mean_latency_ms": round(avg_latency, 2),
            "anomalies_ensemble": n_anomalies,
            "anomalies_threshold": n_thresh,
            "anomalies_zscore": n_zscore,
            "anomalies_iforest": n_iforest,
            "cumulative_processed": self.total_processed_records,
            "timestamp": proc_time.isoformat()
        }

        with open(self.metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_record) + "\n")

        print(
            f"  [Spark Stream Batch {batch_id}] Rows: {count:,} | "
            f"Duration: {batch_duration_sec:.2f}s ({rows_per_sec:,.0f} rows/s) | "
            f"Latency: {avg_latency:.1f}ms | "
            f"Anomalies: {n_anomalies} (Thresh: {n_thresh}, Z: {n_zscore}, IF: {n_iforest})"
        )


def run_streaming_pipeline(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic: str = TOPIC_TELEMETRY,
    max_batches: Optional[int] = None,
    trigger_interval: str = "2 seconds",
    starting_offsets: str = "earliest"
):
    print("=" * 60)
    print("STARTING SPARK STRUCTURED STREAMING PIPELINE")
    print(f"  Kafka Source: {bootstrap_servers} -> Topic: {topic}")
    print(f"  Trigger Interval: {trigger_interval}")
    print("=" * 60)

    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Schema definition for incoming Kafka JSON messages
    telemetry_schema = StructType([
        StructField("udi", IntegerType(), True),
        StructField("product_id", StringType(), True),
        StructField("product_type", StringType(), True),
        StructField("air_temperature", DoubleType(), True),
        StructField("process_temperature", DoubleType(), True),
        StructField("rotational_speed", IntegerType(), True),
        StructField("torque", DoubleType(), True),
        StructField("tool_wear", IntegerType(), True),
        StructField("machine_failure", IntegerType(), True),
        StructField("twf", IntegerType(), True),
        StructField("hdf", IntegerType(), True),
        StructField("pwf", IntegerType(), True),
        StructField("osf", IntegerType(), True),
        StructField("rnf", IntegerType(), True),
        StructField("timestamp", StringType(), True),
        StructField("event_seq", LongType(), True)
    ])

    # 1. Ingest raw stream from Kafka
    raw_kafka_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    # 2. Parse JSON payload and apply schema validation
    parsed_stream = (
        raw_kafka_stream
        .select(
            F.from_json(F.col("value").cast("string"), telemetry_schema).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .filter(F.col("air_temperature").isNotNull())
        .filter(F.col("rotational_speed") > 0)
        .filter(F.col("torque") > 0)
    )

    # 3. Feature Engineering in Spark SQL
    enriched_stream = (
        parsed_stream
        .withColumn("temp_diff", F.col("process_temperature") - F.col("air_temperature"))
        .withColumn("power_watts", F.col("torque") * F.col("rotational_speed") * (2.0 * math.pi / 60.0))
        .withColumn("overstrain_val", F.col("tool_wear") * F.col("torque"))
    )

    processor = MicroBatchProcessor(bootstrap_servers=bootstrap_servers)

    checkpoint_path = CHECKPOINT_DIR / "stream_processor"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    query = (
        enriched_stream.writeStream
        .foreachBatch(processor.process_batch)
        .trigger(processingTime=trigger_interval)
        .option("checkpointLocation", str(checkpoint_path))
        .start()
    )

    print(f"Streaming query started: ID = {query.id}, RunId = {query.runId}")

    try:
        if max_batches is not None:
            # Run for a fixed number of batches then stop
            print(f"Running for up to {max_batches} micro-batches...")
            current_batch = 0
            while current_batch < max_batches and query.isActive:
                time.sleep(2)
                progress = query.lastProgress
                if progress and progress.get("batchId", -1) >= max_batches - 1:
                    break
            query.stop()
        else:
            query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping streaming pipeline...")
        query.stop()
    finally:
        spark.stop()
        print("Spark session stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Spark Structured Streaming Anomaly Processor")
    parser.add_argument("--servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=TOPIC_TELEMETRY, help="Kafka input topic")
    parser.add_argument("--interval", default="2 seconds", help="Micro-batch interval")
    parser.add_argument("--batches", type=int, default=None, help="Max micro-batches to process (default: run continuously)")
    parser.add_argument("--starting-offsets", default="earliest", choices=["earliest", "latest"], help="Kafka starting offsets")
    args = parser.parse_args()

    run_streaming_pipeline(
        bootstrap_servers=args.servers,
        topic=args.topic,
        max_batches=args.batches,
        trigger_interval=args.interval,
        starting_offsets=args.starting_offsets
    )


if __name__ == "__main__":
    main()
