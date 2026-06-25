#!/bin/bash
#SBATCH --job-name=model2lgnFull
#SBATCH --output=logs/model2lgnFull_%A_%a.out
#SBATCH --error=logs/model2lgnFull_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-17

TAUS=(40 60 80)
SIZE_LABELS=(
  "s12_p2388M"
  "s13_p2747M"
  "s14_p3132M"
  "s15_p3542M"
  "s16_p3977M"
  "s17_p4437M"
)
MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnFull_s12_p2388M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s13_p2747M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s14_p3132M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s15_p3542M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s16_p3977M"
  "lgn-dense-2-dense-100_128_-model2lgnFull_s17_p4437M"
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
