#!/bin/bash
#SBATCH --job-name=s08CustomXEval
#SBATCH --output=logs/s08CustomXEval_%A_%a.out
#SBATCH --error=logs/s08CustomXEval_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${S08_CUSTOM_SCOPE:?Missing custom S08 scope}"
: "${S08_CUSTOM_ROOT:?Missing custom S08 root}"
: "${S08_CUSTOM_TRAINING_PERCENTAGES:?Missing custom S08 training percentages}"
: "${S08_CUSTOM_X_PERCENTAGES:?Missing custom S08 x-axis percentages}"
S08_CUSTOM_SKIP_BASELINE="${S08_CUSTOM_SKIP_BASELINE:-0}"
S08_CUSTOM_CHUNK_OFFSET="${S08_CUSTOM_CHUNK_OFFSET:-7}"
RESOURCE_ROOT="${RESOURCE_ROOT:-results/SLURM/2026_07_03_model2_resource_precision_5h_20k}"

SKIP_BASELINE_ARGS=()
if [[ "$S08_CUSTOM_SKIP_BASELINE" == "1" ]]; then
  SKIP_BASELINE_ARGS+=(--skip-baseline)
fi

IFS=$'\t' read -r CONDITION FAULT_SCOPE TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME CHUNK_INDEX FLIP_COUNT_TEXT <<< "$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_custom_xaxis_points \
    --scope "$S08_CUSTOM_SCOPE" \
    --root "$S08_CUSTOM_ROOT" \
    --training-percentages "$S08_CUSTOM_TRAINING_PERCENTAGES" \
    --x-percentages "$S08_CUSTOM_X_PERCENTAGES" \
    --chunk-offset "$S08_CUSTOM_CHUNK_OFFSET" \
    "${SKIP_BASELINE_ARGS[@]}" \
    --evaluation-task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a CHUNK_COUNTS <<< "$FLIP_COUNT_TEXT"

if [[ "$CONDITION" == "inference_only" ]]; then
  TRAINING_DIR="${RESOURCE_ROOT}/${MODEL}"
else
  if [[ "$FAULT_SCOPE" == "input" ]]; then
    TRAINING_DIR="${S08_CUSTOM_ROOT}/input_scope/${TAG}/${MODEL}/training"
  else
    TRAINING_DIR="${S08_CUSTOM_ROOT}/${TAG}/${MODEL}/training"
  fi
fi
CLEAN_MODEL="${TRAINING_DIR}/final_model.pth"
TRAINING_STATUS="${TRAINING_DIR}/training_complete.json"
if [[ "$CONDITION" == "inference_only" ]]; then
  CLEAN_MODEL="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
  TRAINING_STATUS="${RESOURCE_ROOT}/${MODEL}/training_complete.json"
fi

if [[ "$FAULT_SCOPE" == "input" ]]; then
  OUTPUT_DIR="${S08_CUSTOM_ROOT}/input_scope/${TAG}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"
  FAULT_SCOPE_ARGS=(--fault-scope input)
else
  OUTPUT_DIR="${S08_CUSTOM_ROOT}/${TAG}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"
  FAULT_SCOPE_ARGS=()
fi

mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating custom S08 x-axis points"
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
  "${FAULT_SCOPE_ARGS[@]}" \
  --model-alias "$MODEL" \
  --registry-name "$REGISTRY_NAME" \
  --clean-model "$CLEAN_MODEL" \
  --training-status "$TRAINING_STATUS" \
  -o "$OUTPUT_DIR"
