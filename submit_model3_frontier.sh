#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

RESULT_ROOT="${1:-results/SLURM/2026_08_24_model3lgnConv_frontier_extension}"
EVAL_ROOT="${2:-results/evaluation/2026_08_24_model3Frontier_discrete_balacc}"

for ROOT in "$RESULT_ROOT" "$EVAL_ROOT"; do
  if [[ -e "$ROOT" ]]; then
    echo "[ERROR] Refusing to reuse existing output: ${ROOT}" >&2
    exit 2
  fi
done

mkdir -p logs
ABSOLUTE_RESULT_ROOT="$(realpath -m "$RESULT_ROOT")"
ABSOLUTE_EVAL_ROOT="$(realpath -m "$EVAL_ROOT")"

TRAIN_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --export="ALL,MODEL3_FRONTIER_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT}" \
    "$REPOSITORY_ROOT/train_model3_frontier.slurm"
})"
TRAIN_JOB="${TRAIN_JOB%%;*}"

# The evaluator is deliberately strict and will fail if any training task did
# not publish a checkpoint. afterany also permits checkpoints saved after the
# advance signal even if Slurm records the corresponding task as TIMEOUT.
EVAL_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --dependency="afterany:${TRAIN_JOB}" \
    --export="ALL,MODEL3_FRONTIER_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT},MODEL3_FRONTIER_EVAL_ROOT=${ABSOLUTE_EVAL_ROOT}" \
    "$REPOSITORY_ROOT/eval_model3_frontier.slurm"
})"
EVAL_JOB="${EVAL_JOB%%;*}"

echo "Submitted Model 3 frontier training: ${TRAIN_JOB} (30 tasks, max 6 concurrent)"
echo "Submitted strict dependent evaluation: ${EVAL_JOB}"
echo "Training root: ${ABSOLUTE_RESULT_ROOT}"
echo "Evaluation root: ${ABSOLUTE_EVAL_ROOT}"
echo "Git commit: $(git rev-parse HEAD)"
echo
echo "Monitor: squeue -j ${TRAIN_JOB},${EVAL_JOB}"
echo "Results: column -s $'\\t' -t < ${ABSOLUTE_EVAL_ROOT}/best_by_size.tsv"
