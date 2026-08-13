#!/bin/bash
#SBATCH --job-name=selectedInputFaultTrain
#SBATCH --output=logs/selectedInputFaultTrain_%A_%a.out
#SBATCH --error=logs/selectedInputFaultTrain_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-21
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${INPUT_FAULT_ROOT:?Missing campaign output root}"

IFS=$'\t' read -r FAMILY CONDITION TAG TRAINING_PERCENTAGE FAULT_PROBABILITY MODEL REGISTRY_NAME MINIMUM_HOURS MINIMUM_STEPS <<< "$(
  python -m inputBitFlipStudy.campaign \
    --training-task-id "$SLURM_ARRAY_TASK_ID"
)"
OUTPUT_DIR="${INPUT_FAULT_ROOT}/${FAMILY}/${TAG}/${MODEL}/training"

mkdir -p logs "$OUTPUT_DIR"

echo "Running selected-model input-bit-fault training"
echo "Family: ${FAMILY}"
echo "Condition: ${CONDITION}"
echo "Model: ${MODEL}"
echo "Registry: ${REGISTRY_NAME}"
echo "Training input-bit fault percentage: ${TRAINING_PERCENTAGE}"
echo "Training input-bit fault probability: ${FAULT_PROBABILITY}"
echo "Output: ${OUTPUT_DIR}"

if [[ "$FAMILY" == "lgn" ]]; then
  python lgnStudies/bitFlipTrainingStudy/train.py \
    -c config/10783560/v3.yml \
    --device cuda \
    -bs 1024 \
    -lr 0.01 \
    --steps "$MINIMUM_STEPS" \
    --minimum-training-hours "$MINIMUM_HOURS" \
    --eval-freq 5000 \
    --compile-model \
    --fault-scope input \
    --training-fault-probability "$FAULT_PROBABILITY" \
    --model "$MODEL" \
    -o "$OUTPUT_DIR"
elif [[ "$FAMILY" == "qkeras" ]]; then
  python -m qkerasStudies.bitFlipTrainingStudy.train \
    -c config/10783560/v3.yml \
    --device cuda \
    -bs 1024 \
    -lr 0.001 \
    --steps "$MINIMUM_STEPS" \
    --minimum-training-hours "$MINIMUM_HOURS" \
    --eval-freq 2000 \
    --fault-scope input \
    --training-fault-probability "$FAULT_PROBABILITY" \
    --model-alias "$MODEL" \
    -o "$OUTPUT_DIR"
else
  echo "Unknown model family: ${FAMILY}" >&2
  exit 2
fi
