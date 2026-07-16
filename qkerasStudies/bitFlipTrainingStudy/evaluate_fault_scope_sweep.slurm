#!/bin/bash
#SBATCH --job-name=qkerasFaultEval
#SBATCH --output=logs/qkerasFaultEval_%A_%a.out
#SBATCH --error=logs/qkerasFaultEval_%A_%a.err
#SBATCH --time=3:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${QKERAS_SCOPE_ROOT:?Missing sweep output root}"
: "${QKERAS_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"
QKERAS_SCOPE_MODELS="${QKERAS_SCOPE_MODELS:-all}"

IFS=$'\t' read -r CONDITION SCOPE_KEY FAULT_SCOPE SCOPE_DIR TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME CHUNK_INDEX FLIP_COUNT_TEXT <<< "$(
  python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
    --models "$QKERAS_SCOPE_MODELS" \
    --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
    --evaluation-task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a CHUNK_COUNTS <<< "$FLIP_COUNT_TEXT"

if [[ "$CONDITION" == "inference_only" ]]; then
  TRAINING_DIR="${QKERAS_SCOPE_ROOT}/clean_training/${MODEL}/training"
else
  TRAINING_DIR="${QKERAS_SCOPE_ROOT}/${SCOPE_DIR}/${TAG}/${MODEL}/training"
fi
CLEAN_MODEL="${TRAINING_DIR}/final_model.keras"
TRAINING_STATUS="${TRAINING_DIR}/training_complete.json"
OUTPUT_DIR="${QKERAS_SCOPE_ROOT}/${SCOPE_DIR}/${TAG}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"

mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating QKeras exact-count activation-bit faults"
echo "Condition: ${CONDITION}"
echo "Fault scope: ${FAULT_SCOPE}"
echo "Training fault percentage: ${TRAINING_PERCENTAGE}"
echo "Model: ${MODEL}"
echo "Evaluation chunk: ${CHUNK_INDEX}"
echo "Inference flip counts: ${CHUNK_COUNTS[*]}"
echo "Checkpoint: ${CLEAN_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python -m qkerasStudies.training_completion \
  --checkpoint "$CLEAN_MODEL" \
  --status "$TRAINING_STATUS" \
  --minimum-hours 3.0 \
  --minimum-steps 6000

python -m qkerasStudies.bitFlipStudy.study \
  -c config/10783560/v3.yml \
  --device cuda \
  -bs 1024 \
  --flip-counts "${CHUNK_COUNTS[@]}" \
  --repeats 5 \
  --seed 42 \
  --no-plots \
  --fault-scope "$FAULT_SCOPE" \
  --model-alias "$MODEL" \
  --registry-name "$REGISTRY_NAME" \
  --clean-model "$CLEAN_MODEL" \
  --training-status "$TRAINING_STATUS" \
  -o "$OUTPUT_DIR"
