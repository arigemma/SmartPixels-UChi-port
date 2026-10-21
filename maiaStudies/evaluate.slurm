#!/bin/bash
#SBATCH --job-name=maiaLnnEval
#SBATCH --output=logs/maiaLnnEval_%A_%a.out
#SBATCH --error=logs/maiaLnnEval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-29%8
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

[[ -f maiaStudies/evaluate.py ]] || { echo "Run from the SmartPixels root." >&2; exit 2; }
: "${MAIA_CACHE_ROOT:?MAIA_CACHE_ROOT is required}"
: "${MAIA_RESULT_ROOT:?MAIA_RESULT_ROOT is required}"

CAMPAIGN="${MAIA_CAMPAIGN:-main}"
ALIAS="$(python -m maiaStudies.campaign --campaign "$CAMPAIGN" --task-index "$SLURM_ARRAY_TASK_ID")"
MODEL_DIR="$MAIA_RESULT_ROOT/$ALIAS"
mkdir -p logs

if [[ -s "$MODEL_DIR/evaluation_complete.json" ]]; then
  echo "[SKIP] Complete evaluation: $MODEL_DIR"
  exit 0
fi

FAMILY="${ALIAS%%_*}"
if [[ "$FAMILY" == dense ]]; then
  BATCH_SIZE="${MAIA_DENSE_EVAL_BATCH_SIZE:-2048}"
else
  BATCH_SIZE="${MAIA_CONV_EVAL_BATCH_SIZE:-256}"
fi

exec python -m maiaStudies.evaluate \
  --campaign "$CAMPAIGN" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --cache-dir "$MAIA_CACHE_ROOT" \
  --model-dir "$MODEL_DIR" \
  --device cuda \
  --batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  --timing-min-ns -0.5 \
  --timing-max-ns 15.0 \
  --maximum-validation-examples "${MAIA_MAXIMUM_VALIDATION_EXAMPLES:-0}" \
  --maximum-test-examples "${MAIA_MAXIMUM_TEST_EXAMPLES:-0}"
