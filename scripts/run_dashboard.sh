#!/usr/bin/env bash
# Launches the interactive Streamlit Dashboard
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "Starting Streamlit Dashboard on port 8501..."
"$DIR/venv/bin/streamlit" run "$DIR/dashboard/app.py" --server.port 8501 --server.headless true
