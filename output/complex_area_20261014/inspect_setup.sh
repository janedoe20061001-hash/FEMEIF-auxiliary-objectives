set -eu
TASK_ROOT=/root/autodl-tmp/femeif_complex_area_20261014_01
ls -l "$TASK_ROOT"/runtime_setup.*
tail -n 35 "$TASK_ROOT/runtime_setup.log"
ps -eo pid,etime,comm | tail -n 15
