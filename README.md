# Real-Time Industrial IoT Anomaly Detection and Scalable Stream Analytics Using Apache Kafka and Apache Spark

## Quickstart & Execution Guide

### Prerequisites
- Linux or WSL 2 (Ubuntu 22.04 / 24.04 / 26.04)
- Java 21 or Java 17 OpenJDK (`java -version`)
- Python 3.10+ (tested on Python 3.14.4)

### Step 1: Environment Setup
```bash
# Clone and enter directory
cd /home/srajimon/big_data_project

# Activate existing virtualenv or create a new one
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Apache Kafka (Standalone KRaft Mode)
No Docker daemon or root privileges required:
```bash
bash scripts/start_kafka.sh
```
*(Alternatively, if Docker Compose is available: `docker compose up -d`)*

### Step 3: Run Offline Training & Holdout Evaluation
```bash
./venv/bin/python -m src.models.train_offline
```

### Step 4: Run Automated Tests
```bash
./venv/bin/pytest tests/ -v
```

### Step 5: Execute End-to-End Streaming Pipeline
```bash
bash scripts/run_pipeline.sh
```

### Step 6: Query Results in the Hive Warehouse
Every scored micro-batch lands in the Hive table `iot_analytics.telemetry_scored` (Parquet, partitioned by `event_date`). Two views sit on top of it: `iot_analytics.anomalies` (rows where `pred_ensemble = 1`) and `iot_analytics.hourly_detector_summary` (per-hour, per-product-type detector counts and latency).
```bash
scripts/query_warehouse                     # canned downstream report
scripts/query_warehouse "SELECT product_type, COUNT(*) FROM iot_analytics.anomalies GROUP BY product_type"
```
By default the warehouse is on the local filesystem (`output/hive_warehouse/`) with an embedded Derby metastore (`output/hive_metastore/`). Derby allows one Spark process at a time, so while the streaming job is running the query CLI skips the metastore and reads the table's Parquet files directly, under the same table and view names. To land the tables on HDFS behind a shared Hive Metastore, set `HIVE_WAREHOUSE_DIR=hdfs://<namenode>:<port>/user/hive/warehouse` and `HIVE_METASTORE_URIS=thrift://<metastore>:9083` before running the pipeline.

> If your shell exports `SPARK_HOME` for a different Spark version, unset it: PySpark 4.2 from the venv must use its own jars. The scripts in `scripts/` do this already.

### Step 7: Run Scalability Benchmarks & Generate Plots
```bash
bash scripts/run_benchmarks.sh
```

### Step 8: Launch Interactive Streamlit Dashboard
```bash
bash scripts/run_dashboard.sh
# Open http://localhost:8501 in your browser
```

### Live Streaming Demo
Runs the producer, the Spark streaming job, and the dashboard together. The dashboard reruns its status bar and telemetry tab every 2 seconds, showing **LIVE** while micro-batches arrive.
```bash
bash scripts/run_live_demo.sh        # 20 events/sec; pass a different rate as the first argument
# Open http://localhost:8501; Ctrl+C stops all three processes
```
The demo uses its own Spark checkpoint (`output/checkpoints/live_demo`) and starts from the latest Kafka offset, so it skips any backlog on the topic. It also clears `output/latest_stream_predictions.csv` so the dashboard shows only this run. Logs go to `output/demo_logs/`. Set `DASHBOARD_PORT` if port 8501 is taken. At 20 events/sec, expect about 40 events per 2-second micro-batch and 1.5–2 s from event to dashboard.
