#!/bin/bash
#SBATCH --job-name=model3lgnConv
#SBATCH --output=logs/model3lgnConv_%A_%a.out
#SBATCH --error=logs/model3lgnConv_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-14

TAUS=(20 40 60)
SIZE_LABELS=(
  "s01_p319M"
  "s02_p465M"
  "s03_p639M"
  "s04_p840M"
  "s05_p1069M"
)
MODEL_PREFIXES=(
  "lgn-conv-3-100-orpool-model3Full_s01_p319M"
  "lgn-conv-3-100-orpool-model3Full_s02_p465M"
  "lgn-conv-3-100-orpool-model3Full_s03_p639M"
  "lgn-conv-3-100-orpool-model3Full_s04_p840M"
  "lgn-conv-3-100-orpool-model3Full_s05_p1069M"
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))

TAU="${TAUS[$TAU_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
OUTPUT_BASE="results/SLURM/2026_06_25_model3lgnConv_expanded"
OUTPUT_DIR="${OUTPUT_BASE}/${SIZE_LABEL}/tau_${TAU}"

mkdir -p logs "$OUTPUT_DIR"

echo "Running ${MODEL}"
echo "Size: ${SIZE_LABEL}"
echo "Tau: ${TAU}"
echo "Output: ${OUTPUT_DIR}"
echo "TensorBoard: ${OUTPUT_DIR}/logs"

python train.py \
  -c config/10783560/v3.yml \
  --balance-classes \
  --device cuda \
  -bs 1024 \
  -lf cross_entropy \
  -lr 0.01 \
  --disable-early-stopping \
  --epochs 0 \
  --compile-model \
  -m "$MODEL" \
  -o "$OUTPUT_DIR"
