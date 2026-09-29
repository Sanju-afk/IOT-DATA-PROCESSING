#!/usr/bin/env bash
# Runs the complete Big Data Benchmark suite and generates plots
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=== Running Big Data Benchmarks ==="
bash "$DIR/scripts/start_kafka.sh"

"$DIR/venv/bin/python" -m src.benchmark --scales 10000 50000 100000

echo "Benchmark suite completed. Results available in output/benchmarks/."
