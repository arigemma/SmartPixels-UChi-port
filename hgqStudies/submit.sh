#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

if [[ ! -f hgqStudies/train.slurm || ! -f config/10783560/v3.yml ]]; then
  echo "[ERROR] Could not locate the HGQ campaign from ${REPOSITORY_ROOT}." >&2
  exit 1
fi

export KERAS_BACKEND=torch
python - <<'PY'
from packaging.version import Version
import hgq
import keras

if Version(hgq.__version__) != Version("0.1.9"):
    raise SystemExit(f"[ERROR] Expected hgq2 0.1.9, found {hgq.__version__}.")
if Version(keras.__version__) < Version("3.11"):
    raise SystemExit(f"[ERROR] Expected Keras >=3.11, found {keras.__version__}.")
if keras.backend.backend() != "torch":
    raise SystemExit(f"[ERROR] Expected Keras torch backend, found {keras.backend.backend()}.")
print(f"[ENV] HGQ2 {hgq.__version__}; Keras {keras.__version__}; backend={keras.backend.backend()}")
PY

RESULT_ROOT="${1:-results/SLURM/$(date +%Y_%m_%d)_hgq2_model2_pareto_qkeras_matched_6k}"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "[ERROR] Refusing to reuse existing campaign root: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p logs "$RESULT_ROOT"

python -m hgqStudies.campaign --write-manifest "$RESULT_ROOT/campaign_manifest.json"
TASK_COUNT="$(python -m hgqStudies.campaign --print-task-count)"
LAST_TASK=$((TASK_COUNT - 1))
MAX_PARALLEL="${HGQ_MAX_PARALLEL:-4}"
ABSOLUTE_RESULT_ROOT="$(realpath "$RESULT_ROOT")"

TRAIN_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --array="0-${LAST_TASK}%${MAX_PARALLEL}" \
    --export="ALL,HGQ_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT}" \
    "$REPOSITORY_ROOT/hgqStudies/train.slurm"
})"
TRAIN_JOB="${TRAIN_JOB%%;*}"

SUMMARY_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --dependency="afterany:${TRAIN_JOB}" \
    --export="ALL,HGQ_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT}" \
    "$REPOSITORY_ROOT/hgqStudies/summarize.slurm"
})"
SUMMARY_JOB="${SUMMARY_JOB%%;*}"

echo "Submitted HGQ2 Model-2 training array: ${TRAIN_JOB} (tasks 0-${LAST_TASK}, max ${MAX_PARALLEL} concurrent)"
echo "Submitted strict summary job:          ${SUMMARY_JOB}"
echo "Campaign root:                        ${ABSOLUTE_RESULT_ROOT}"
echo "Git commit:                           $(git rev-parse HEAD)"
echo
echo "Monitor: squeue -j ${TRAIN_JOB},${SUMMARY_JOB}"
echo "Results: column -s $'\\t' -t < ${ABSOLUTE_RESULT_ROOT}/summary/balanced_accuracy_summary.tsv"
