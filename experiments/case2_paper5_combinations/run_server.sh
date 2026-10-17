#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" launch_batch.py --run --workers "${WORKERS:-28}" --config config.json --output runs
"$PYTHON" report_results.py runs
