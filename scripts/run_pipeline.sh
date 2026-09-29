#!/usr/bin/env bash
# Runs end-to-end Industrial IoT Anomaly Detection Pipeline
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=== Industrial IoT Stream Analytics Pipeline ==="

# 1. Ensure Kafka is active
bash "$DIR/scripts/start_kafka.sh"

# 2. Train offline detectors if not already trained
if [ ! -f "$DIR/models/isolation_forest.joblib" ]; then
    echo "Training offline detectors on AI4I 2020 dataset..."
    "$DIR/venv/bin/python" -m src.models.train_offline
fi

# 3. Launch producer with 2,000 events from test split
echo "Streaming 2,000 telemetry events to Kafka..."
"$DIR/venv/bin/python" -m src.producer --events 2000 --rate 200 --mode replay

# 4. Process streaming records in Spark for 3 microbatches
echo "Processing micro-batches in Spark Structured Streaming..."
"$DIR/venv/bin/python" -m src.streaming.spark_stream_processor --batches 3 --starting-offsets earliest

echo "Pipeline slice executed successfully. Results written to output/stream_results/."
