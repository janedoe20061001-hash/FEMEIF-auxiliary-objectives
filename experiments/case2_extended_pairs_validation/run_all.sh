#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-.venv/bin/python}"
WORKERS="${WORKERS:-28}"

"$PYTHON" launch_batch.py --run --workers "$WORKERS" --config config_pair_screen.json --output pair_screen_runs
"$PYTHON" report_results.py pair_screen_runs
"$PYTHON" launch_batch.py --run --workers "$WORKERS" --config config_validation.json --output validation_runs
"$PYTHON" report_results.py validation_runs
