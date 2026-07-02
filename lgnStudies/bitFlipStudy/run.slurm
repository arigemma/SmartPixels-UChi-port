#!/bin/bash
#SBATCH --job-name=model2BitFlips
#SBATCH --output=logs/model2BitFlips_%A_%a.out
#SBATCH --error=logs/model2BitFlips_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --signal=B:USR1@600
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

MODELS=(
  "s06_p760M_tau40"
  "s08_p1202M_tau40"
  "s09_p1460M_tau40"
)
REGISTRY_NAMES=(
  "lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_40"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_40"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
REGISTRY_NAME="${REGISTRY_NAMES[$SLURM_ARRAY_TASK_ID]}"
RESOURCE_ROOT="results/SLURM/2026_07_02_model2_resource_precision"
OUTPUT_ROOT="results/SLURM/2026_07_02_model2_bit_flip_inference_exact_count"
CLEAN_MODEL="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}"

mkdir -p logs "$OUTPUT_DIR"

echo "Model: ${MODEL}"
echo "Clean checkpoint: ${CLEAN_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python lgnStudies/bitFlipStudy/study.py \
  -c config/10783560/v3.yml \
  --device cuda \
  -bs 1024 \
  --flip-counts 0 1 2 4 8 16 32 64 128 256 512 1024 \
  --repeats 5 \
  --seed 42 \
  --model-alias "$MODEL" \
  --registry-name "$REGISTRY_NAME" \
  --clean-model "$CLEAN_MODEL" \
  -o "$OUTPUT_DIR"
