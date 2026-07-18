#!/bin/bash
#SBATCH --job-name=s08InputFaultTrain
#SBATCH --output=logs/s08InputFaultTrain_%A_%a.out
#SBATCH --error=logs/s08InputFaultTrain_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${S08_SCOPE_ROOT:?Missing sweep output root}"
: "${S08_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"

IFS=$'\t' read -r TRAINING_PERCENTAGE FAULT_SCOPE TAG MODEL REGISTRY_NAME <<< "$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
    --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
    --training-task-id "$SLURM_ARRAY_TASK_ID"
)"
FAULT_PROBABILITY="$(
  python -c \
    'import sys; print(float(sys.argv[1]) / 100.0)' \
    "$TRAINING_PERCENTAGE"
)"
OUTPUT_DIR="${S08_SCOPE_ROOT}/input_scope/${TAG}/${MODEL}/training"

mkdir -p logs "$OUTPUT_DIR"

echo "Running s08 input-scope fault-aware training sweep"
echo "Model: ${MODEL}"
echo "Registry: ${REGISTRY_NAME}"
echo "Training fault scope: ${FAULT_SCOPE}"
echo "Training fault percentage: ${TRAINING_PERCENTAGE}%"
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
  --fault-scope input \
  --training-fault-probability "$FAULT_PROBABILITY" \
  --model "$MODEL" \
  -o "$OUTPUT_DIR"
