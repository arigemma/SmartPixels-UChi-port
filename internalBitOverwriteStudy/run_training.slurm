#!/bin/bash
#SBATCH --job-name=internalOverwriteTrain
#SBATCH --output=logs/internalOverwriteTrain_%A_%a.out
#SBATCH --error=logs/internalOverwriteTrain_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7%8
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"
: "${INTERNAL_OVERWRITE_ROOT:?Missing campaign output root}"

IFS=$'\t' read -r FAMILY TAG TRAINING_PERCENTAGE FAULT_PROBABILITY MODEL REGISTRY_NAME MINIMUM_HOURS MINIMUM_STEPS <<< "$(
  python -m internalBitOverwriteStudy.campaign \
    --training-task-id "$SLURM_ARRAY_TASK_ID"
)"
OUTPUT_DIR="${INTERNAL_OVERWRITE_ROOT}/${FAMILY}/${TAG}/${MODEL}/training"
mkdir -p logs "$OUTPUT_DIR"

echo "Training ${FAMILY}/${MODEL} from scratch"
echo "Internal random-overwrite training rate: ${TRAINING_PERCENTAGE}%"
echo "Output: ${OUTPUT_DIR}"

if [[ "$FAMILY" == "lnn" ]]; then
  python lgnStudies/bitFlipTrainingStudy/train.py \
    -c config/10783560/v3.yml \
    --device cuda \
    -bs 1024 \
    -lr 0.01 \
    --steps "$MINIMUM_STEPS" \
    --minimum-training-hours "$MINIMUM_HOURS" \
    --eval-freq 5000 \
    --compile-model \
    --fault-scope internal \
    --fault-operation random_set \
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
    --fault-scope internal \
    --fault-operation random_set \
    --training-fault-probability "$FAULT_PROBABILITY" \
    --model-alias "$MODEL" \
    -o "$OUTPUT_DIR"
else
  echo "Unknown family: ${FAMILY}" >&2
  exit 2
fi
