#!/bin/bash
set -euo pipefail

INPUT_FAULT_ROOT="${1:-results/SLURM/$(date +%Y_%m_%d)_selected_input_bit_flip_1to10pct_eval60_linear}"

if [[ -e "$INPUT_FAULT_ROOT" ]]; then
  echo "Refusing to reuse an existing campaign root: ${INPUT_FAULT_ROOT}" >&2
  exit 2
fi

TRAINING_TASKS="$(
  python -m inputBitFlipStudy.campaign --print-training-task-count
)"
EVALUATION_TASKS="$(
  python -m inputBitFlipStudy.campaign --print-evaluation-task-count
)"
MODEL_GRIDS="$(
  python -m inputBitFlipStudy.campaign --print-model-grids
)"
GIT_COMMIT="$(git rev-parse HEAD)"

python -m inputBitFlipStudy.campaign \
  --write-campaign-config \
  --root "$INPUT_FAULT_ROOT" \
  --git-commit "$GIT_COMMIT"
mkdir -p logs

export INPUT_FAULT_ROOT
export PYTHONUNBUFFERED=1

TRAINING_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((TRAINING_TASKS - 1))" \
    --export=ALL \
    inputBitFlipStudy/run_training.slurm
)"
TRAINING_JOB_ID="${TRAINING_JOB_ID%%;*}"
echo "Submitted selected-model training array ${TRAINING_JOB_ID}"

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((EVALUATION_TASKS - 1))" \
    --dependency="afterok:${TRAINING_JOB_ID}" \
    --export=ALL \
    inputBitFlipStudy/run_evaluation.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted dependent input-fault evaluation array ${EVALUATION_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    inputBitFlipStudy/combine.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent combine/plot job ${COMBINE_JOB_ID}"

echo "Campaign root: ${INPUT_FAULT_ROOT}"
echo "Git commit: ${GIT_COMMIT}"
echo "Training tasks: ${TRAINING_TASKS}"
echo "Evaluation tasks: ${EVALUATION_TASKS}"
echo "Training percentages: baseline clean, then 1 2 3 4 5 6 7 8 9 10%"
echo "Evaluation target grid: 0 5 10 15 20 25 30 35 40 45 50 55 60%"
echo "Model evaluation grids:"
echo "$MODEL_GRIDS"
