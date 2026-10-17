#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python launch_batch.py --workers 28 --output runs
.venv/bin/python report_results.py runs
