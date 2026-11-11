#!/bin/bash
# HTCondor job wrapper: runs one MAIA training task on a worker node.
# Usage: train_condor.sh <task-index> <profile> [device] [results-tag]
#   results-tag: "none" (default) keeps $MAIA_RESULT_ROOT; anything else sends
#   results to ${MAIA_RESULT_ROOT}_<results-tag>/<alias>/
set -euo pipefail
TASK="${1:?task index}"; PROFILE="${2:-pilot}"; DEVICE="${3:-cuda}"; RTAG="${4:-none}"
HERE=/home/agarciac/projects/mcc_smartpix/SmartPixels/maiaStudies
source "$HERE/set_paths.sh"
if [ "$RTAG" != "none" ]; then
    export MAIA_RESULT_ROOT="${MAIA_RESULT_ROOT}_${RTAG}"
fi
PY=/home/agarciac/miniforge3/envs/smart_pixels/bin/python

echo "[job] host=$(hostname) task=$TASK profile=$PROFILE device=$DEVICE results=$MAIA_RESULT_ROOT"
nvidia-smi -L 2>&1 || echo "[job] no nvidia-smi"
"$PY" -c "import torch; print('[job] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
exec "$PY" "$HERE/maia_run.py" train --task-index "$TASK" --profile "$PROFILE" --device "$DEVICE"
