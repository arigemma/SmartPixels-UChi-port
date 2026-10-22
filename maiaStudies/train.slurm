#!/bin/bash
#SBATCH --job-name=maiaLnnTrain
#SBATCH --output=logs/maiaLnnTrain_%A_%a.out
#SBATCH --error=logs/maiaLnnTrain_%A_%a.err
#SBATCH --time=06:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-29%6
#SBATCH --signal=B:USR1@600
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

[[ -f maiaStudies/train.py ]] || { echo "Run from the SmartPixels root." >&2; exit 2; }
: "${MAIA_CACHE_ROOT:?MAIA_CACHE_ROOT is required}"
: "${MAIA_RESULT_ROOT:?MAIA_RESULT_ROOT is required}"

CAMPAIGN="${MAIA_CAMPAIGN:-main}"
ALIAS="$(python -m maiaStudies.campaign --campaign "$CAMPAIGN" --task-index "$SLURM_ARRAY_TASK_ID")"
OUTPUT="$MAIA_RESULT_ROOT/$ALIAS"
mkdir -p logs "$OUTPUT"

if [[ -s "$OUTPUT/training_complete.json" ]]; then
  echo "[SKIP] Complete training: $OUTPUT"
  exit 0
fi

if [[ "$ALIAS" == dense* ]]; then
  BATCH_SIZE="${MAIA_DENSE_BATCH_SIZE:-1024}"
else
  BATCH_SIZE="${MAIA_CONV_BATCH_SIZE:-128}"
fi

ARGS=(
  --campaign "$CAMPAIGN"
  --task-index "$SLURM_ARRAY_TASK_ID"
  --cache-dir "$MAIA_CACHE_ROOT"
  --output "$OUTPUT"
  --device cuda
  --batch-size "$BATCH_SIZE"
  --num-workers "${MAIA_NUM_WORKERS:-4}"
  --learning-rate "${MAIA_LEARNING_RATE:-0.01}"
  --minimum-steps "${MAIA_MINIMUM_STEPS:-20000}"
  --minimum-training-hours "${MAIA_MINIMUM_TRAINING_HOURS:-5.0}"
  --eval-frequency "${MAIA_EVAL_FREQUENCY:-5000}"
  --validation-examples "${MAIA_VALIDATION_EXAMPLES:-100000}"
  --seed 42
)

if [[ "${MAIA_TRACK_RELAXED_VALIDATION:-0}" == "1" ]]; then
  ARGS+=(--track-relaxed-validation)
fi

exec python -m maiaStudies.train "${ARGS[@]}"
