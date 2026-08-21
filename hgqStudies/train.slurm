#!/bin/bash
#SBATCH --job-name=hgqM2Pareto
#SBATCH --output=logs/hgqM2Pareto_%A_%a.out
#SBATCH --error=logs/hgqM2Pareto_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-20%4
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f hgqStudies/train.py || ! -f config/10783560/v3.yml ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi
if [[ -z "${HGQ_RESULT_ROOT:-}" ]]; then
  echo "[ERROR] HGQ_RESULT_ROOT was not exported by hgqStudies/submit.sh." >&2
  exit 1
fi

export KERAS_BACKEND=torch
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

python -m hgqStudies.train \
  --task "$SLURM_ARRAY_TASK_ID" \
  --data-config "${HGQ_DATA_CONFIG:-config/10783560/v3.yml}" \
  --output-root "$HGQ_RESULT_ROOT" \
  --device cuda \
  --batch-size "${HGQ_BATCH_SIZE:-1024}" \
  --learning-rate "${HGQ_LEARNING_RATE:-0.001}" \
  --optimizer-steps "${HGQ_OPTIMIZER_STEPS:-6000}" \
  --steps-per-epoch "${HGQ_STEPS_PER_EPOCH:-50}" \
  --validation-every "${HGQ_VALIDATION_EVERY:-10}"
