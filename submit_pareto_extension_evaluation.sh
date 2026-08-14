#!/bin/bash
set -euo pipefail

DENSE_ROOT="${1:-results/SLURM/2026_08_14_model2lgnFull_left_extension_fixed}"
CONV_ROOT="${2:-results/SLURM/2026_08_14_model3lgnConv_curve_extension_fixed}"
EVAL_ROOT="${3:-results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc}"

count_checkpoints() {
  local root="$1"
  find "$root" -mindepth 3 -maxdepth 3 \
    -path '*/tau_*/best_model.pth' -size +0c | wc -l
}

DENSE_COUNT="$(count_checkpoints "$DENSE_ROOT")"
CONV_COUNT="$(count_checkpoints "$CONV_ROOT")"
if [[ "$DENSE_COUNT" -ne 21 || "$CONV_COUNT" -ne 30 ]]; then
  echo "Refusing partial evaluation: dense=${DENSE_COUNT}/21 conv=${CONV_COUNT}/30" >&2
  exit 2
fi

mkdir -p logs

JOB_ID="$(
  sbatch \
    --parsable \
    --export="ALL,PARETO_DENSE_RESULT_ROOT=${DENSE_ROOT},PARETO_CONV_RESULT_ROOT=${CONV_ROOT},PARETO_EXTENSION_EVAL_ROOT=${EVAL_ROOT}" \
    eval_lgn_pareto_extensions.slurm
)"
JOB_ID="${JOB_ID%%;*}"

echo "Submitted dense/conv discrete-evaluation array ${JOB_ID} (tasks 0-1)"
echo "Evaluation root: ${EVAL_ROOT}"
echo "Dense checkpoints: ${DENSE_COUNT}/21"
echo "Conv checkpoints: ${CONV_COUNT}/30"
