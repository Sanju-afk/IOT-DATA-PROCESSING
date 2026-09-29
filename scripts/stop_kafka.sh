#!/usr/bin/env bash
# Stops local Kafka KRaft broker
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

if [ -f "$DIR/output/kafka_server.pid" ]; then
    PID=$(cat "$DIR/output/kafka_server.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping Kafka broker (PID: $PID)..."
        kill "$PID"
        rm -f "$DIR/output/kafka_server.pid"
        echo "Kafka stopped."
        exit 0
    fi
fi

# Fallback pattern kill for kafka server process started from kafka_dist
PID=$(pgrep -f "kafka.Kafka.*kafka_kraft.properties" || true)
if [ -n "$PID" ]; then
    echo "Stopping Kafka process $PID..."
    kill $PID
    echo "Kafka stopped."
else
    echo "No running Kafka broker detected."
fi
