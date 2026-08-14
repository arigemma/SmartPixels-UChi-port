#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash inputBitFlipStudy/resume_failed_evaluation.sh CAMPAIGN_ROOT [ARRAY_SPEC]" >&2
  exit 2
fi

INPUT_FAULT_ROOT="$1"
FAILED_TASKS="${2:-0-43,84-87}"

if [[ ! -f "${INPUT_FAULT_ROOT}/campaign_config.json" ]]; then
  echo "Campaign config does not exist: ${INPUT_FAULT_ROOT}/campaign_config.json" >&2
  exit 2
fi

EXPECTED_TRAINING_TASKS="$(
  python -m inputBitFlipStudy.campaign --print-training-task-count
)"
COMPLETED_TRAINING_TASKS="$(
  find "$INPUT_FAULT_ROOT" -name training_complete.json -type f | wc -l
)"
if [[ "$COMPLETED_TRAINING_TASKS" -ne "$EXPECTED_TRAINING_TASKS" ]]; then
  echo "Refusing to resume evaluation: found ${COMPLETED_TRAINING_TASKS}/${EXPECTED_TRAINING_TASKS} completed training tasks." >&2
  exit 2
fi

mkdir -p logs
export INPUT_FAULT_ROOT
export PYTHONUNBUFFERED=1

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="$FAILED_TASKS" \
    --export=ALL \
    inputBitFlipStudy/run_evaluation.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted resumed evaluation array ${EVALUATION_JOB_ID} (${FAILED_TASKS})"

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
