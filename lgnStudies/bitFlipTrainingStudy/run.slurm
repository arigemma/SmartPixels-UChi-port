#!/bin/bash
#SBATCH --job-name=model2FaultAwareTrain
#SBATCH --output=logs/model2FaultAwareTrain_%A_%a.out
#SBATCH --error=logs/model2FaultAwareTrain_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${BIT_FLIP_TRAINING_PERCENTAGE:?Missing training fault percentage}"
: "${BIT_FLIP_TRAINING_ROOT:?Missing campaign output root}"

IFS=$'\t' read -r MODEL REGISTRY_NAME <<< "$(
  python -m lgnStudies.bitFlipTrainingStudy.campaign \
    --training-task-id "$SLURM_ARRAY_TASK_ID"
)"
FAULT_PROBABILITY="$(
  python -c \
    'import sys; print(float(sys.argv[1]) / 100.0)' \
    "$BIT_FLIP_TRAINING_PERCENTAGE"
)"
OUTPUT_DIR="${BIT_FLIP_TRAINING_ROOT}/${MODEL}/training"

mkdir -p logs "$OUTPUT_DIR"

echo "Running model-2 fault-aware training study"
echo "Model: ${MODEL}"
echo "Registry: ${REGISTRY_NAME}"
echo "Training fault percentage: ${BIT_FLIP_TRAINING_PERCENTAGE}%"
echo "Output: ${OUTPUT_DIR}"

python lgnStudies/bitFlipTrainingStudy/train.py \
  -c config/10783560/v3.yml \
  --device cuda \
  -bs 1024 \
  -lr 0.01 \
  --steps 20000 \
  --minimum-training-hours 5.0 \
  --eval-freq 5000 \
  --compile-model \
  --training-fault-probability "$FAULT_PROBABILITY" \
  --model "$MODEL" \
  -o "$OUTPUT_DIR"
