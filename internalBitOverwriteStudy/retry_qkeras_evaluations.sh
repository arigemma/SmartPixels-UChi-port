#!/bin/bash
set -euo pipefail

if [[ ! -f internalBitOverwriteStudy/retry_qkeras_evaluations.sh ]]; then
  echo "Run this command from the SmartPixels repository root." >&2
  exit 2
fi
if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CAMPAIGN_ROOT PRIOR_EVALUATION_ARRAY_JOB_ID" >&2
  exit 2
fi

export INTERNAL_OVERWRITE_TRAINING_PERCENTAGES="${INTERNAL_OVERWRITE_TRAINING_PERCENTAGES:-0 1 3 10}"
export INTERNAL_OVERWRITE_ROOT="$1"
PRIOR_EVALUATION_JOB_ID="$2"
if [[ ! -f "${INTERNAL_OVERWRITE_ROOT}/campaign_config.json" ]]; then
  echo "Campaign config not found: ${INTERNAL_OVERWRITE_ROOT}/campaign_config.json" >&2
  exit 2
fi
if [[ ! "$PRIOR_EVALUATION_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Prior evaluation array job ID must be numeric." >&2
  exit 2
fi

bash internalBitOverwriteStudy/prepare_qkeras_runtime.sh
source internalBitOverwriteStudy/qkeras_runtime.sh
python -m internalBitOverwriteStudy.campaign --qkeras-preflight
python -m internalBitOverwriteStudy.campaign \
  --root "$INTERNAL_OVERWRITE_ROOT" \
  --write-qkeras-retry-manifest

RETRY_TASKS="$(
  python -m internalBitOverwriteStudy.campaign \
    --root "$INTERNAL_OVERWRITE_ROOT" \
    --print-qkeras-retry-task-count
)"
if [[ "$RETRY_TASKS" -eq 0 ]]; then
  echo "No missing QKeras evaluation points were found."
  exit 0
fi

export PYTHONUNBUFFERED=1
mkdir -p logs
RETRY_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --array="0-$((RETRY_TASKS - 1))%12" \
    --dependency="afterany:${PRIOR_EVALUATION_JOB_ID}" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/run_qkeras_retry_evaluation.slurm"
)"
RETRY_JOB_ID="${RETRY_JOB_ID%%;*}"

COMBINE_JOB_ID="$(
  sbatch --parsable \
    --chdir="$PWD" \
    --dependency="afterok:${RETRY_JOB_ID}" \
    --export=ALL \
    "$PWD/internalBitOverwriteStudy/combine.slurm"
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"

{
  printf 'prior_evaluation_job_id=%s\n' "$PRIOR_EVALUATION_JOB_ID"
  printf 'qkeras_evaluation_retry_job_id=%s\n' "$RETRY_JOB_ID"
  printf 'combine_after_retry_job_id=%s\n' "$COMBINE_JOB_ID"
  printf 'retry_task_count=%s\n' "$RETRY_TASKS"
  printf 'retry_git_commit=%s\n' "$(git rev-parse HEAD)"
} > "${INTERNAL_OVERWRITE_ROOT}/evaluation_retry_jobs.txt"

echo "QKeras point-retry array: ${RETRY_JOB_ID} (${RETRY_TASKS} one-point tasks)"
echo "Final plot job:           ${COMBINE_JOB_ID}"
echo "Recorded in: ${INTERNAL_OVERWRITE_ROOT}/evaluation_retry_jobs.txt"
