#!/bin/bash
#SBATCH --job-name=model2BitFlips
#SBATCH --output=logs/model2BitFlips_%A_%a.out
#SBATCH --error=logs/model2BitFlips_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-20
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

IFS=$'\t' read -r MODEL REGISTRY_NAME CHUNK_INDEX FLIP_COUNT_TEXT <<< "$(
  python -m lgnStudies.bitFlipStudy.campaign --task-id "$SLURM_ARRAY_TASK_ID"
)"
read -r -a CHUNK_COUNTS <<< "$FLIP_COUNT_TEXT"

RESOURCE_ROOT="results/SLURM/2026_07_02_model2_resource_precision"
OUTPUT_ROOT="results/SLURM/2026_07_03_model2_bit_flip_inference_dense_sweep"
CLEAN_MODEL="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}/chunks/chunk_${CHUNK_INDEX}"

mkdir -p logs "$OUTPUT_DIR"

echo "Model: ${MODEL}"
echo "Chunk: ${CHUNK_INDEX}"
echo "Flip counts: ${CHUNK_COUNTS[*]}"
echo "Clean checkpoint: ${CLEAN_MODEL}"
echo "Output: ${OUTPUT_DIR}"

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
  --clean-model "$CLEAN_MODEL" \
  -o "$OUTPUT_DIR"
