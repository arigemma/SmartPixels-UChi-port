#!/bin/bash
set -euo pipefail

if [[ ! -f internalBitOverwriteStudy/submit.sh ]]; then
  echo "Run this command from the SmartPixels repository root." >&2
  exit 2
fi

export INTERNAL_OVERWRITE_TRAINING_PERCENTAGES="${INTERNAL_OVERWRITE_TRAINING_PERCENTAGES:-0 1 3 10}"
INTERNAL_OVERWRITE_ROOT="${1:-results/SLURM/$(date +%Y_%m_%d)_selected_internal_random_overwrite_0_1_3_10pct}"
if [[ -e "$INTERNAL_OVERWRITE_ROOT" ]]; then
  echo "Refusing to reuse existing campaign root: ${INTERNAL_OVERWRITE_ROOT}" >&2
  exit 2
fi

TRAINING_TASKS="$(python -m internalBitOverwriteStudy.campaign --print-training-task-count)"
EVALUATION_TASKS="$(python -m internalBitOverwriteStudy.campaign --print-evaluation-task-count)"
MODEL_GRIDS="$(python -m internalBitOverwriteStudy.campaign --print-model-grids)"
GIT_COMMIT="$(git rev-parse HEAD)"

python -m internalBitOverwriteStudy.campaign \
  --write-campaign-config --root "$INTERNAL_OVERWRITE_ROOT" --git-commit "$GIT_COMMIT"
mkdir -p logs

export INTERNAL_OVERWRITE_ROOT
export PYTHONUNBUFFERED=1

TRAINING_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --array="0-$((TRAINING_TASKS - 1))%8" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/run_training.slurm"
)"
TRAINING_JOB_ID="${TRAINING_JOB_ID%%;*}"

EVALUATION_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --array="0-$((EVALUATION_TASKS - 1))%12" \
    --dependency="afterok:${TRAINING_JOB_ID}" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/run_evaluation.slurm"
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"

COMBINE_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/combine.slurm"
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"

{
  printf 'training_job_id=%s\n' "$TRAINING_JOB_ID"
  printf 'evaluation_job_id=%s\n' "$EVALUATION_JOB_ID"
  printf 'combine_job_id=%s\n' "$COMBINE_JOB_ID"
} > "${INTERNAL_OVERWRITE_ROOT}/submitted_jobs.txt"

echo "Training array:   ${TRAINING_JOB_ID} (${TRAINING_TASKS} tasks)"
echo "Evaluation array: ${EVALUATION_JOB_ID} (${EVALUATION_TASKS} tasks; waits for training)"
echo "Final plot job:   ${COMBINE_JOB_ID} (waits for all evaluations)"
echo "Campaign root:    ${INTERNAL_OVERWRITE_ROOT}"
echo "Training rates:   ${INTERNAL_OVERWRITE_TRAINING_PERCENTAGES}%"
echo "Internal-site grids:"
echo "$MODEL_GRIDS"
echo
echo "Final outputs after the dependency chain succeeds:"
echo "  ${INTERNAL_OVERWRITE_ROOT}/lnn_vs_qkeras_internal_random_overwrite.png"
echo "  ${INTERNAL_OVERWRITE_ROOT}/lnn_vs_qkeras_internal_random_overwrite.pdf"
echo "  ${INTERNAL_OVERWRITE_ROOT}/internal_random_overwrite_plot_points.csv"
echo "  ${INTERNAL_OVERWRITE_ROOT}/internal_random_overwrite_raw_results.csv"
