#!/usr/bin/env bash
# Live demo: producer -> Kafka -> Spark Structured Streaming -> Streamlit (auto-refreshing).
# Usage: bash scripts/run_live_demo.sh [events_per_sec]   (default 20)
# Ctrl+C stops the dashboard, the Spark job, and the producer.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Use the venv's PySpark jars, not a system Spark install of a different version
unset SPARK_HOME
# Log lines must reach output/demo_logs/ immediately; the startup check below greps for them
export PYTHONUNBUFFERED=1

RATE="${1:-20}"
PORT="${DASHBOARD_PORT:-8501}"
PY="$DIR/venv/bin/python"

if nc -z localhost "$PORT" 2>/dev/null; then
    echo "Error: port $PORT is already in use (another dashboard running?). Stop it or set DASHBOARD_PORT."
    exit 1
fi
LOG_DIR="$DIR/output/demo_logs"
mkdir -p "$LOG_DIR"

bash "$DIR/scripts/start_kafka.sh"

# Fresh demo checkpoint so Spark starts at the latest offset instead of replaying the topic backlog,
# and an empty live view so the dashboard shows only this run.
rm -rf "$DIR/output/checkpoints/live_demo"
rm -f "$DIR/output/latest_stream_predictions.csv"

PIDS=()
cleanup() {
    echo
    echo "Stopping live demo..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Stopped. Scored rows from this run are in Hive: bash scripts/query_warehouse.sh"
}
trap cleanup EXIT INT TERM

echo "Starting Spark Structured Streaming job (log: output/demo_logs/stream.log)..."
"$PY" -m src.streaming.spark_stream_processor \
    --starting-offsets latest --checkpoint-name live_demo \
    > "$LOG_DIR/stream.log" 2>&1 &
PIDS+=($!)

echo "Waiting for the streaming query to start..."
for _ in {1..120}; do
    if grep -q "Streaming query started" "$LOG_DIR/stream.log"; then
        break
    fi
    if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
        echo "Error: streaming job exited. See output/demo_logs/stream.log."
        exit 1
    fi
    sleep 1
done
if ! grep -q "Streaming query started" "$LOG_DIR/stream.log"; then
    echo "Error: streaming query did not start within 120s. See output/demo_logs/stream.log."
    exit 1
fi

echo "Starting producer at ${RATE} events/sec (log: output/demo_logs/producer.log)..."
"$PY" -m src.producer --mode replay --rate "$RATE" --events 100000000 \
    > "$LOG_DIR/producer.log" 2>&1 &
PIDS+=($!)

echo "Starting dashboard at http://localhost:$PORT (Ctrl+C to stop everything)"
"$DIR/venv/bin/streamlit" run "$DIR/dashboard/app.py" --server.port "$PORT" --server.headless true &
PIDS+=($!)
wait "${PIDS[2]}"
