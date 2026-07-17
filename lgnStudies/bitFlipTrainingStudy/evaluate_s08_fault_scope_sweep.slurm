#!/bin/bash
#SBATCH --job-name=s08InputFaultEval
#SBATCH --output=logs/s08InputFaultEval_%A_%a.out
#SBATCH --error=logs/s08InputFaultEval_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${S08_SCOPE_ROOT:?Missing sweep output root}"
: "${S08_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"
RESOURCE_ROOT="${RESOURCE_ROOT:-results/SLURM/2026_07_03_model2_resource_precision_5h_20k}"

IFS=$'\t' read -r CONDITION FAULT_SCOPE TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME CHUNK_INDEX FLIP_COUNT_TEXT <<< "$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
    --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
    --evaluation-task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a CHUNK_COUNTS <<< "$FLIP_COUNT_TEXT"

if [[ "$CONDITION" == "inference_only" ]]; then
  CLEAN_MODEL="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
  TRAINING_STATUS="${RESOURCE_ROOT}/${MODEL}/training_complete.json"
else
  TRAINING_DIR="${S08_SCOPE_ROOT}/input_scope/${TAG}/${MODEL}/training"
  CLEAN_MODEL="${TRAINING_DIR}/final_model.pth"
  TRAINING_STATUS="${TRAINING_DIR}/training_complete.json"
fi
OUTPUT_DIR="${S08_SCOPE_ROOT}/input_scope/${TAG}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"

mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating s08 input-scope inference faults"
echo "Condition: ${CONDITION}"
echo "Fault scope: ${FAULT_SCOPE}"
echo "Training fault percentage: ${TRAINING_PERCENTAGE}"
echo "Model: ${MODEL}"
echo "Evaluation chunk: ${CHUNK_INDEX}"
echo "Inference flip counts: ${CHUNK_COUNTS[*]}"
echo "Checkpoint: ${CLEAN_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python -m lgnStudies.training_completion \
  --checkpoint "$CLEAN_MODEL" \
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
  --fault-scope input \
  --model-alias "$MODEL" \
  --registry-name "$REGISTRY_NAME" \
  --clean-model "$CLEAN_MODEL" \
  --training-status "$TRAINING_STATUS" \
  -o "$OUTPUT_DIR"
