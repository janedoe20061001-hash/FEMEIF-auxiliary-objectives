#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--run" ]]; then
  echo "No experiment started. Inspect with: python experiment.py plan"
  echo "To start the formal batch explicitly: bash run_server.sh --run"
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" launch_batch.py --run --workers "${WORKERS:-28}" --output runs
"$PYTHON" report_results.py runs
