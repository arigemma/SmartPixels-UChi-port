#!/bin/bash
#SBATCH --job-name=inputFaultRefineEval
#SBATCH --output=logs/inputFaultRefineEval_%A_%a.out
#SBATCH --error=logs/inputFaultRefineEval_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-21
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${INPUT_FAULT_ROOT:?Missing campaign output root}"

REFINEMENT_MODE="${INPUT_FAULT_REFINEMENT_MODE:-low}"
if [[ "$REFINEMENT_MODE" == "low" ]]; then
  TASK_ACTION="--refinement-task-id"
  OUTPUT_SUBDIRECTORY="refinement_0to10pct"
elif [[ "$REFINEMENT_MODE" == "paper" ]]; then
  TASK_ACTION="--paper-refinement-task-id"
  OUTPUT_SUBDIRECTORY="paper_refinement_0to25pct"
else
  echo "Unknown refinement mode: ${REFINEMENT_MODE}" >&2
  exit 2
fi

IFS=$'\t' read -r FAMILY CONDITION TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME FLIP_COUNT_TEXT MINIMUM_HOURS MINIMUM_STEPS <<< "$(
  python -m inputBitFlipStudy.campaign \
    "$TASK_ACTION" "$SLURM_ARRAY_TASK_ID"
)"
read -r -a FLIP_COUNTS <<< "$FLIP_COUNT_TEXT"

MODEL_ROOT="${INPUT_FAULT_ROOT}/${FAMILY}/${TAG}/${MODEL}"
TRAINING_DIR="${MODEL_ROOT}/training"
OUTPUT_DIR="${MODEL_ROOT}/${OUTPUT_SUBDIRECTORY}"
mkdir -p logs "$OUTPUT_DIR"

echo "Evaluating additional random input-bit faults"
echo "Refinement mode: ${REFINEMENT_MODE}"
echo "Family: ${FAMILY}"
echo "Condition: ${CONDITION}"
echo "Training input-bit fault percentage: ${TRAINING_PERCENTAGE}"
echo "Model: ${MODEL}"
echo "Additional exact random input-bit counts: ${FLIP_COUNTS[*]}"
echo "Output: ${OUTPUT_DIR}"

if [[ "$FAMILY" == "lgn" ]]; then
  CHECKPOINT="${TRAINING_DIR}/final_model.pth"
  STATUS="${TRAINING_DIR}/training_complete.json"
  python -m lgnStudies.training_completion \
    --checkpoint "$CHECKPOINT" \
    --status "$STATUS" \
    --minimum-hours "$MINIMUM_HOURS" \
    --minimum-steps "$MINIMUM_STEPS"
  python lgnStudies/bitFlipStudy/study.py \
    -c config/10783560/v3.yml \
    --device cuda \
    -bs 1024 \
    --flip-counts "${FLIP_COUNTS[@]}" \
    --repeats 5 \
    --seed 42 \
    --no-plots \
    --fault-scope input \
    --model-alias "$MODEL" \
    --registry-name "$REGISTRY_NAME" \
    --clean-model "$CHECKPOINT" \
    --training-status "$STATUS" \
    -o "$OUTPUT_DIR"
elif [[ "$FAMILY" == "qkeras" ]]; then
  CHECKPOINT="${TRAINING_DIR}/final_model.keras"
  STATUS="${TRAINING_DIR}/training_complete.json"
  python -m qkerasStudies.training_completion \
    --checkpoint "$CHECKPOINT" \
    --status "$STATUS" \
    --minimum-hours "$MINIMUM_HOURS" \
    --minimum-steps "$MINIMUM_STEPS"
  python -m qkerasStudies.bitFlipStudy.study \
    -c config/10783560/v3.yml \
    --device cuda \
    -bs 1024 \
    --flip-counts "${FLIP_COUNTS[@]}" \
    --repeats 5 \
    --seed 42 \
    --no-plots \
    --fault-scope input \
    --model-alias "$MODEL" \
    --registry-name "$REGISTRY_NAME" \
    --clean-model "$CHECKPOINT" \
    --training-status "$STATUS" \
    -o "$OUTPUT_DIR"
else
  echo "Unknown model family: ${FAMILY}" >&2
  exit 2
fi
