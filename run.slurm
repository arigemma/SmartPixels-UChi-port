#!/bin/bash
#SBATCH --job-name=model2lgnFull
#SBATCH --output=logs/model2lgnFull_%A_%a.out
#SBATCH --error=logs/model2lgnFull_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-17

TAUS=(10 20 40)
SIZE_LABELS=(
  "s06_p760M"
  "s07_p968M"
  "s08_p1202M"
  "s09_p1460M"
  "s10_p1744M"
  "s11_p2054M"
)
MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s07_p968M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s10_p1744M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s11_p2054M"
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))

TAU="${TAUS[$TAU_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
OUTPUT_BASE="results/SLURM/2026_06_22_model2lgnFull_expanded"
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
