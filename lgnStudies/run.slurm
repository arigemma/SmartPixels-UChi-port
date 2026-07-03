#!/bin/bash
#SBATCH --job-name=model2ResourcePrecision
#SBATCH --output=logs/model2ResourcePrecision_%A_%a.out
#SBATCH --error=logs/model2ResourcePrecision_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

MODELS=(
  "s06_p760M_tau40"
  "s08_p1202M_tau40"
  "s09_p1460M_tau40"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="results/SLURM/2026_07_02_model2_resource_precision/${MODEL}"

mkdir -p logs "$OUTPUT_DIR"

echo "Running model-2 LGN resource/precision study"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"

python lgnStudies/resource_precision_study.py \
  -c config/10783560/v3.yml \
  --device cuda \
  -bs 1024 \
  -lr 0.01 \
  --steps 2000 \
  --eval-freq 200 \
  --compile-model \
  --model "$MODEL" \
  -o "$OUTPUT_DIR"
