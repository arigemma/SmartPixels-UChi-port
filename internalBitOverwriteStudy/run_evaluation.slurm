#!/bin/bash
#SBATCH --job-name=internalOverwriteEval
#SBATCH --output=logs/internalOverwriteEval_%A_%a.out
#SBATCH --error=logs/internalOverwriteEval_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-39%12
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"
: "${INTERNAL_OVERWRITE_ROOT:?Missing campaign output root}"

IFS=$'\t' read -r FAMILY TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME CHUNK_INDEX COUNT_TEXT MINIMUM_HOURS MINIMUM_STEPS <<< "$(
  python -m internalBitOverwriteStudy.campaign \
    --evaluation-task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a COUNTS <<< "$COUNT_TEXT"
TRAINING_DIR="${INTERNAL_OVERWRITE_ROOT}/${FAMILY}/${TAG}/${MODEL}/training"
OUTPUT_DIR="${INTERNAL_OVERWRITE_ROOT}/${FAMILY}/${TAG}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"
mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating ${FAMILY}/${MODEL}, trained at ${TRAINING_PERCENTAGE}%"
echo "Exact selected internal-site counts: ${COUNTS[*]}"
echo "Each selected site receives an independent random 0/1 target"

if [[ "$FAMILY" == "lnn" ]]; then
  CHECKPOINT="${TRAINING_DIR}/final_model.pth"
  STATUS="${TRAINING_DIR}/training_complete.json"
  python -m lgnStudies.training_completion \
    --checkpoint "$CHECKPOINT" --status "$STATUS" \
    --minimum-hours "$MINIMUM_HOURS" --minimum-steps "$MINIMUM_STEPS"
  python lgnStudies/bitFlipStudy/study.py \
    -c config/10783560/v3.yml --device cuda -bs 1024 \
    --flip-counts "${COUNTS[@]}" --repeats 5 --seed 42 --no-plots \
    --fault-scope internal --fault-operation random_set \
    --model-alias "$MODEL" --registry-name "$REGISTRY_NAME" \
    --clean-model "$CHECKPOINT" --training-status "$STATUS" \
    -o "$OUTPUT_DIR"
elif [[ "$FAMILY" == "qkeras" ]]; then
  CHECKPOINT="${TRAINING_DIR}/final_model.keras"
  STATUS="${TRAINING_DIR}/training_complete.json"
  python -m qkerasStudies.training_completion \
    --checkpoint "$CHECKPOINT" --status "$STATUS" \
    --minimum-hours "$MINIMUM_HOURS" --minimum-steps "$MINIMUM_STEPS"
  python -m qkerasStudies.bitFlipStudy.study \
    -c config/10783560/v3.yml --device cuda -bs 1024 \
    --flip-counts "${COUNTS[@]}" --repeats 5 --seed 42 --no-plots \
    --fault-scope internal --fault-operation random_set \
    --model-alias "$MODEL" --registry-name "$REGISTRY_NAME" \
    --clean-model "$CHECKPOINT" --training-status "$STATUS" \
    -o "$OUTPUT_DIR"
else
  echo "Unknown family: ${FAMILY}" >&2
  exit 2
fi
