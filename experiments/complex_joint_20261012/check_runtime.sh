set -eu
TASK_ROOT=/root/autodl-tmp/femeif_complex_joint_20261012_01
tail -n 10 "$TASK_ROOT/runtime_setup.log"
if [ -e "$TASK_ROOT/runtime_setup.complete" ]; then echo RUNTIME_COMPLETE; fi
