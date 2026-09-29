"""
Modernized execution of the original baseline repository logic:
https://github.com/yugokato/Spark-and-Kafka_IoT-Data-Processing-and-Analytics

The original repository was written in Python 2 using deprecated Spark 1.x/2.0 DStreams
(KafkaUtils.createDirectStream) and single-variable temperature sensor simulation.
This module executes the exact analytical workload of the original project
(State-level average temperature, message counts, device counts) using modern Spark
to allow a fair, scientifically rigorous performance comparison against our extension.
"""

import json
import time
from typing import Dict, Any, List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    LongType
)

from src.config import KAFKA_BOOTSTRAP_SERVERS


def run_baseline_benchmark(
    records: List[Dict[str, Any]],
    spark: SparkSession
) -> Dict[str, Any]:
    """
    Executes the 4 analyses from the original project on a batch of records:
    1. Average temperature in each state (sorted descending)
    2. Total message count
    3. Number of sensors in each state (sorted ascending)
    4. Total number of sensors
    """
    start_time = time.time()
    n_records = len(records)

    # Convert records into Spark DataFrame matching original JSON schema:
    # {"guid": ..., "state": ..., "payload": {"data": {"temperature": ...}}}
    json_rows = []
    for r in records:
        json_rows.append((
            str(r.get("product_id", r.get("guid", "0-ZZZ12345678-01A"))),
            str(r.get("product_type", r.get("state", "CA"))),
            float(r.get("air_temperature", r.get("temperature", 300.0)))
        ))

    schema = StructType([
        StructField("guid", StringType(), True),
        StructField("state", StringType(), True),
        StructField("temperature", DoubleType(), True)
    ])

    df = spark.createDataFrame(json_rows, schema)

    # 1. Average temperature by state
    avg_temp = (
        df.groupBy("state")
        .agg(F.avg("temperature").alias("avg_temp"))
        .sort(F.col("avg_temp").desc())
        .collect()
    )

    # 2. Total message count
    total_messages = df.count()

    # 3. Number of sensors by state
    sensors_by_state = (
        df.select("state", "guid")
        .distinct()
        .groupBy("state")
        .agg(F.count("guid").alias("num_sensors"))
        .sort(F.col("state").asc())
        .collect()
    )

    # 4. Total number of unique sensors
    total_sensors = df.select("guid").distinct().count()

    duration = time.time() - start_time
    throughput = n_records / duration if duration > 0 else 0

    return {
        "workload": "Original Baseline (State Temperature Aggregations)",
        "records_processed": n_records,
        "processing_duration_sec": round(duration, 4),
        "throughput_rows_per_sec": round(throughput, 2),
        "state_averages_count": len(avg_temp),
        "total_messages": total_messages,
        "total_sensors": total_sensors
    }
