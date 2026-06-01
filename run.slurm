#!/bin/bash
#SBATCH --job-name=model2lgnFull
#SBATCH --output=logs/model2lgnFull_%A_%a.out
#SBATCH --error=logs/model2lgnFull_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-24

TAUS=(5 10 20 30 40)
SIZE_LABELS=(
  "s01_dsNewLike_p023M"
  "s02_p163M"
  "s03_p298M"
  "s04_p434M"
  "s05_currentWide_p577M"
)
MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnFull_s01_dsNewLike"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s03_p298M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s04_p434M"
  "lgn-dense-2-dense-100_128_-model2lgnFull"
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))

TAU="${TAUS[$TAU_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
OUTPUT_BASE="results/SLURM/2026_06_01_model2lgnFull"
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
