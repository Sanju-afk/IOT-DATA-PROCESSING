#!/usr/bin/env bash
# Starts local Kafka KRaft broker if not already running on port 9092
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

if nc -z localhost 9092 2>/dev/null; then
    echo "Kafka is already running on localhost:9092."
    exit 0
fi

echo "Initializing Kafka KRaft Storage..."
mkdir -p kafka_data
if [ ! -f kafka_data/meta.properties ]; then
    CLUSTER_ID=$("$DIR/kafka_dist/bin/kafka-storage.sh" random-uuid)
    "$DIR/kafka_dist/bin/kafka-storage.sh" format --standalone -t "$CLUSTER_ID" -c "$DIR/config/kafka_kraft.properties"
fi

echo "Starting Kafka Server in background..."
nohup "$DIR/kafka_dist/bin/kafka-server-start.sh" "$DIR/config/kafka_kraft.properties" > "$DIR/output/kafka_server.log" 2>&1 &
echo $! > "$DIR/output/kafka_server.pid"

echo "Waiting for Kafka broker to listen on port 9092..."
for i in {1..30}; do
    if nc -z localhost 9092 2>/dev/null; then
        echo "Kafka broker is active and listening on localhost:9092!"
        exit 0
    fi
    sleep 1
done

echo "Error: Kafka failed to start within 30 seconds. Check output/kafka_server.log."
exit 1
