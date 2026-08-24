#!/bin/bash
#SBATCH --job-name=hgqM2Calib
#SBATCH --output=logs/hgqM2Calib_%A_%a.out
#SBATCH --error=logs/hgqM2Calib_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-20%4
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ -z "${HGQ_RESULT_ROOT:-}" ]]; then
  echo "[ERROR] HGQ_RESULT_ROOT must point to the completed training campaign." >&2
  exit 1
fi

export KERAS_BACKEND=torch
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

python -m hgqStudies.calibrate \
  --task "$SLURM_ARRAY_TASK_ID" \
  --result-root "$HGQ_RESULT_ROOT" \
  --data-config "${HGQ_DATA_CONFIG:-config/10783560/v3.yml}" \
  --device cuda \
  --batch-size "${HGQ_CALIBRATION_BATCH_SIZE:-1024}" \
  --calibration-batches "${HGQ_CALIBRATION_BATCHES:-0}" \
  --minimum-balanced-accuracy "${HGQ_MINIMUM_BALANCED_ACCURACY:-0.60}"
