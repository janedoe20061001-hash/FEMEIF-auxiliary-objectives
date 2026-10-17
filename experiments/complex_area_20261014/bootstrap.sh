#!/usr/bin/env bash
set -eu
TASK_ROOT=/root/autodl-tmp/femeif_complex_area_20261014_01
mkdir -p "$TASK_ROOT"
/root/miniconda3/bin/python - <<'PY'
import json,os
from pathlib import Path
def read(p):
 try:return Path(p).read_text().strip()
 except OSError:return None
print(json.dumps(dict(affinity=len(os.sched_getaffinity(0)),cpu_quota=read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),cpu_period=read('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),cpu_usage=read('/sys/fs/cgroup/cpuacct/cpuacct.usage'),memory_limit=read('/sys/fs/cgroup/memory/memory.limit_in_bytes')),indent=2))
PY
if [ ! -e "$TASK_ROOT/runtime_setup.started" ]; then
  touch "$TASK_ROOT/runtime_setup.started"
  nohup bash -c '/root/miniconda3/bin/python -m venv /root/autodl-tmp/femeif_complex_area_20261014_01/.venv && /root/autodl-tmp/femeif_complex_area_20261014_01/.venv/bin/python -m pip install numpy==2.4.3 scipy==1.18.1 pymoo==0.6.1.6 moocore==0.3.2 matplotlib==3.11.2 psutil==7.0.0 && /root/autodl-tmp/femeif_complex_area_20261014_01/.venv/bin/python -m pip freeze > /root/autodl-tmp/femeif_complex_area_20261014_01/requirements-installed.txt && touch /root/autodl-tmp/femeif_complex_area_20261014_01/runtime_setup.complete' > "$TASK_ROOT/runtime_setup.log" 2>&1 < /dev/null &
  printf 'runtime_setup_pid=%s\n' "$!"
else
  printf 'Runtime setup already initiated; no duplicate start.\n'
fi
