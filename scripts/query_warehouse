#!/usr/bin/env bash
# Queries the Hive warehouse of scored stream results.
# Usage: bash scripts/query_warehouse.sh                  (canned report)
#        bash scripts/query_warehouse.sh "SELECT ..."     (ad-hoc HiveQL / Spark SQL)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Use the venv's PySpark jars, not a system Spark install of a different version
unset SPARK_HOME

if [ $# -gt 0 ]; then
    "$DIR/venv/bin/python" -m src.warehouse --sql "$1"
else
    "$DIR/venv/bin/python" -m src.warehouse --report
fi
