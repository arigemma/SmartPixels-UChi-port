#!/bin/bash
set -euo pipefail

DENSE_ROOT="${1:-results/SLURM/2026_08_14_model2lgnFull_left_extension_fixed}"
CONV_ROOT="${2:-results/SLURM/2026_08_14_model3lgnConv_curve_extension_fixed}"

for ROOT in "$DENSE_ROOT" "$CONV_ROOT"; do
  if [[ -e "$ROOT" ]]; then
    echo "Refusing to reuse an existing training root: ${ROOT}" >&2
    exit 2
  fi
done

mkdir -p logs

JOB_ID="$(
  sbatch \
    --parsable \
    --export="ALL,PARETO_DENSE_OUTPUT_BASE=${DENSE_ROOT},PARETO_CONV_OUTPUT_BASE=${CONV_ROOT}" \
    run.slurm
)"
JOB_ID="${JOB_ID%%;*}"

echo "Submitted fixed Pareto-extension training array ${JOB_ID} (tasks 0-50)"
echo "Dense root: ${DENSE_ROOT}"
echo "Conv root: ${CONV_ROOT}"
echo "Git commit: $(git rev-parse HEAD)"
