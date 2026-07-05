#!/bin/bash
#SBATCH --job-name=model2FaultAwareEval
#SBATCH --output=logs/model2FaultAwareEval_%A_%a.out
#SBATCH --error=logs/model2FaultAwareEval_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-20
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${BIT_FLIP_TRAINING_PERCENTAGE:?Missing training fault percentage}"
: "${BIT_FLIP_TRAINING_ROOT:?Missing campaign output root}"

IFS=$'\t' read -r MODEL REGISTRY_NAME CHUNK_INDEX FLIP_COUNT_TEXT <<< "$(
  python -m lgnStudies.bitFlipTrainingStudy.campaign \
    --evaluation-task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a CHUNK_COUNTS <<< "$FLIP_COUNT_TEXT"

TRAINING_DIR="${BIT_FLIP_TRAINING_ROOT}/${MODEL}/training"
FAULT_AWARE_MODEL="${TRAINING_DIR}/final_model.pth"
TRAINING_STATUS="${TRAINING_DIR}/training_complete.json"
OUTPUT_DIR="${BIT_FLIP_TRAINING_ROOT}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"

mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating a fault-aware model with inference-time faults"
echo "Model: ${MODEL}"
echo "Training fault percentage: ${BIT_FLIP_TRAINING_PERCENTAGE}%"
echo "Evaluation chunk: ${CHUNK_INDEX}"
echo "Inference flip counts: ${CHUNK_COUNTS[*]}"
echo "Checkpoint: ${FAULT_AWARE_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python -m lgnStudies.training_completion \
  --checkpoint "$FAULT_AWARE_MODEL" \
  --status "$TRAINING_STATUS" \
  --minimum-hours 5.0 \
  --minimum-steps 20000

python lgnStudies/bitFlipStudy/study.py \
  -c config/10783560/v3.yml \
  --device cuda \
  -bs 1024 \
  --flip-counts "${CHUNK_COUNTS[@]}" \
  --repeats 5 \
  --seed 42 \
  --no-plots \
  --model-alias "$MODEL" \
  --registry-name "$REGISTRY_NAME" \
  --clean-model "$FAULT_AWARE_MODEL" \
  --training-status "$TRAINING_STATUS" \
  -o "$OUTPUT_DIR"
