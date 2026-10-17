#!/usr/bin/env bash
set -eu
TASK_ROOT=/root/autodl-tmp/femeif_complex_joint_20261012_01
cd "$TASK_ROOT"
test -e runtime_setup.complete
if [ -e pipeline.started ]; then echo 'Pipeline already initiated; no duplicate launch'; exit 0; fi
.venv/bin/python -m pip install mpmath==1.3.0
.venv/bin/python -m pip freeze > requirements-installed.txt
gcc -O3 -shared -fPIC experiments/complex_joint_20261012/frozen/native.c -o experiments/complex_joint_20261012/frozen/native.so -lm
gcc -O3 -shared -fPIC experiments/complex_joint_20261012/frechet.c -o experiments/complex_joint_20261012/frechet.so -lm
touch pipeline.started
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup .venv/bin/python -u experiments/complex_joint_20261012/pipeline.py > pipeline.log 2>&1 < /dev/null &
printf '%s\n' "$!" > pipeline.pid
printf 'Pipeline PID: %s\n' "$!"
