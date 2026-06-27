#!/bin/bash
#SBATCH --job-name=model3lgnDeep
#SBATCH --output=logs/model3lgnDeep_%A_%a.out
#SBATCH --error=logs/model3lgnDeep_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-11

# The unchanged s01_p319M baseline is already trained under
# results/SLURM/2026_06_25_model3lgnConv_expanded and is intentionally omitted.
TAUS=(20 40 60)
SIZE_LABELS=(
  "s02_p463M"
  "s03_p607M"
  "s04_p751M"
  "s05_p895M"
)
MODEL_PREFIXES=(
  "lgn-conv-3-100-orpool-model3Deep_s02_p463M"
  "lgn-conv-3-100-orpool-model3Deep_s03_p607M"
  "lgn-conv-3-100-orpool-model3Deep_s04_p751M"
  "lgn-conv-3-100-orpool-model3Deep_s05_p895M"
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))

TAU="${TAUS[$TAU_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"
OUTPUT_BASE="results/SLURM/2026_06_27_model3lgnConv_deeper"
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
