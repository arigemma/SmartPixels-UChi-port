#!/bin/bash
set -euo pipefail

if [[ ! -f internalBitOverwriteStudy/recover_qkeras.sh ]]; then
  echo "Run this command from the SmartPixels repository root." >&2
  exit 2
fi

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CAMPAIGN_ROOT ORIGINAL_TRAINING_ARRAY_JOB_ID" >&2
  exit 2
fi

export INTERNAL_OVERWRITE_TRAINING_PERCENTAGES="${INTERNAL_OVERWRITE_TRAINING_PERCENTAGES:-0 1 3 10}"
export INTERNAL_OVERWRITE_ROOT="$1"
ORIGINAL_TRAINING_JOB_ID="$2"

if [[ ! -f "${INTERNAL_OVERWRITE_ROOT}/campaign_config.json" ]]; then
  echo "Campaign config not found: ${INTERNAL_OVERWRITE_ROOT}/campaign_config.json" >&2
  exit 2
fi
if [[ ! "$ORIGINAL_TRAINING_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Original training job ID must be numeric." >&2
  exit 2
fi

python -m internalBitOverwriteStudy.campaign --qkeras-preflight

EXPECTED_TRAINING_TASKS="$(
  python -m internalBitOverwriteStudy.campaign --print-training-task-count
)"
EXPECTED_EVALUATION_TASKS="$(
  python -m internalBitOverwriteStudy.campaign --print-evaluation-task-count
)"
if [[ "$EXPECTED_TRAINING_TASKS" -ne 8 || "$EXPECTED_EVALUATION_TASKS" -ne 40 ]]; then
  echo "Recovery is restricted to the original 8-training/40-evaluation grid." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
mkdir -p logs

QKERAS_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --array="4-7%4" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/run_training.slurm"
)"
QKERAS_JOB_ID="${QKERAS_JOB_ID%%;*}"

EVALUATION_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --array="0-39%12" \
    --dependency="afterany:${ORIGINAL_TRAINING_JOB_ID},afterok:${QKERAS_JOB_ID}" \
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
  printf 'original_training_job_id=%s\n' "$ORIGINAL_TRAINING_JOB_ID"
  printf 'qkeras_recovery_job_id=%s\n' "$QKERAS_JOB_ID"
  printf 'evaluation_recovery_job_id=%s\n' "$EVALUATION_JOB_ID"
  printf 'combine_recovery_job_id=%s\n' "$COMBINE_JOB_ID"
  printf 'recovery_git_commit=%s\n' "$(git rev-parse HEAD)"
} > "${INTERNAL_OVERWRITE_ROOT}/recovery_jobs.txt"

echo "QKeras recovery array: ${QKERAS_JOB_ID} (tasks 4-7 only)"
echo "Evaluation array:      ${EVALUATION_JOB_ID}"
echo "Final plot job:        ${COMBINE_JOB_ID}"
echo "The evaluation waits for original LNN tasks 0-3 and the new QKeras array."
echo "Recorded in: ${INTERNAL_OVERWRITE_ROOT}/recovery_jobs.txt"
