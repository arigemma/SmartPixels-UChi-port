#!/bin/bash
#SBATCH --job-name=lgnParetoExtend
#SBATCH --output=logs/lgnParetoExtend_%A_%a.out
#SBATCH --error=logs/lgnParetoExtend_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-50
#SBATCH --signal=B:USR1@300

# One array entry per architecture/tau pair:
#   7 dense left-extension sizes x 3 taus = 21 jobs
#   10 conv curve-extension sizes x 3 taus = 30 jobs
TAUS=(10 20 40)

FAMILIES=(
  "dense" "dense" "dense" "dense" "dense" "dense" "dense"
  "conv" "conv" "conv" "conv" "conv"
  "conv" "conv" "conv" "conv" "conv"
)

SIZE_LABELS=(
  "s01_p005M"
  "s02_p008M"
  "s03_p010M"
  "s04_p013M"
  "s05_p015M"
  "s06_p018M"
  "s07_p020M"
  "s01_p046M"
  "s02_p074M"
  "s03_p109M"
  "s04_p152M"
  "s05_p201M"
  "s06_p1325M"
  "s07_p1608M"
  "s08_p1919M"
  "s09_p2257M"
  "s10_p2623M"
)

MODEL_PREFIXES=(
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s01_p005M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s02_p008M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s03_p010M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s04_p013M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s05_p015M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s06_p018M"
  "lgn-dense-2-dense-100_128_-model2lgnFullLeft_s07_p020M"
  "lgn-conv-3-100-orpool-model3Curve_s01_p046M"
  "lgn-conv-3-100-orpool-model3Curve_s02_p074M"
  "lgn-conv-3-100-orpool-model3Curve_s03_p109M"
  "lgn-conv-3-100-orpool-model3Curve_s04_p152M"
  "lgn-conv-3-100-orpool-model3Curve_s05_p201M"
  "lgn-conv-3-100-orpool-model3Curve_s06_p1325M"
  "lgn-conv-3-100-orpool-model3Curve_s07_p1608M"
  "lgn-conv-3-100-orpool-model3Curve_s08_p1919M"
  "lgn-conv-3-100-orpool-model3Curve_s09_p2257M"
  "lgn-conv-3-100-orpool-model3Curve_s10_p2623M"
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))

TAU="${TAUS[$TAU_INDEX]}"
FAMILY="${FAMILIES[$SIZE_INDEX]}"
SIZE_LABEL="${SIZE_LABELS[$SIZE_INDEX]}"
MODEL="${MODEL_PREFIXES[$SIZE_INDEX]}_${TAU}"

if [[ "$FAMILY" == "dense" ]]; then
  OUTPUT_BASE="${PARETO_DENSE_OUTPUT_BASE:-results/SLURM/2026_08_13_model2lgnFull_left_extension}"
else
  OUTPUT_BASE="${PARETO_CONV_OUTPUT_BASE:-results/SLURM/2026_08_13_model3lgnConv_curve_extension}"
fi
OUTPUT_DIR="${OUTPUT_BASE}/${SIZE_LABEL}/tau_${TAU}"

BATCH_SIZE=1024
if [[ "$FAMILY" == "conv" && "$SIZE_LABEL" == "s10_p2623M" ]]; then
  # This architecture exceeds 40 GB at batch 1024. Use the same safe batch
  # size for all three taus so their comparison remains internally consistent.
  BATCH_SIZE=512
fi

mkdir -p logs "$OUTPUT_DIR"

echo "Running ${MODEL}"
echo "Family: ${FAMILY}"
echo "Size: ${SIZE_LABEL}"
echo "Tau: ${TAU}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "TensorBoard: ${OUTPUT_DIR}/logs"

exec python train.py \
  -c config/10783560/v3.yml \
  --balance-classes \
  --device cuda \
  -bs "$BATCH_SIZE" \
  -lf cross_entropy \
  -lr 0.01 \
  --disable-early-stopping \
  --epochs 0 \
  --compile-model \
  -m "$MODEL" \
  -o "$OUTPUT_DIR"
