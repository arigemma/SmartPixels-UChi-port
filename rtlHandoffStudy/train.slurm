#!/bin/bash
#SBATCH --job-name=rtlHandoffTrain
#SBATCH --output=logs/rtlHandoffTrain_%A_%a.out
#SBATCH --error=logs/rtlHandoffTrain_%A_%a.err
#SBATCH --time=05:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-19%6
#SBATCH --signal=B:USR1@300
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f rtlHandoffStudy/train.py || ! -f config/10783560/v3.yml ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
ALIAS="$(python -c 'from models.rtl_handoff import RTL_HANDOFF_CONFIGS; import os; print(RTL_HANDOFF_CONFIGS[int(os.environ["SLURM_ARRAY_TASK_ID"])]["alias"])')"
OUTPUT="$ROOT/$ALIAS"
mkdir -p logs "$OUTPUT"

if [[ -s "$OUTPUT/training_complete.json" ]]; then
  echo "[SKIP] Complete training output: $OUTPUT"
  exit 0
fi

exec python -m rtlHandoffStudy.train \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --data-config config/10783560/v3.yml \
  --output "$OUTPUT" \
  --device cuda \
  --batch-size 1024 \
  --learning-rate 0.01 \
  --minimum-steps 20000 \
  --minimum-training-hours 5.0 \
  --eval-frequency 5000 \
  --compile-model
