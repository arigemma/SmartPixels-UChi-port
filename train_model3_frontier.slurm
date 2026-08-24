#!/bin/bash
#SBATCH --job-name=model3Frontier
#SBATCH --output=logs/model3Frontier_%A_%a.out
#SBATCH --error=logs/model3Frontier_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-29%6
#SBATCH --signal=B:USR1@900
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f train.py || ! -f config/10783560/v3.yml ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi

TAUS=(10 20 40)
SIZES=(
  s01_p003M s02_p007M s03_p012M s04_p018M s05_p026M
  s06_p039M s07_p069M s08_p110M s09_p213M s10_p385M
)

TAU_INDEX=$((SLURM_ARRAY_TASK_ID % ${#TAUS[@]}))
SIZE_INDEX=$((SLURM_ARRAY_TASK_ID / ${#TAUS[@]}))
if [[ "$SIZE_INDEX" -ge "${#SIZES[@]}" ]]; then
  echo "[ERROR] Array task ${SLURM_ARRAY_TASK_ID} is outside the 30-point grid." >&2
  exit 2
fi

TAU="${TAUS[$TAU_INDEX]}"
SIZE="${SIZES[$SIZE_INDEX]}"
MODEL="lgn-conv-3-100-orpool-model3Frontier_${SIZE}_${TAU}"
RESULT_ROOT="${MODEL3_FRONTIER_RESULT_ROOT:-results/SLURM/2026_08_24_model3lgnConv_frontier_extension}"
OUTPUT_DIR="${RESULT_ROOT}/${SIZE}/tau_${TAU}"

mkdir -p logs "$OUTPUT_DIR"

if [[ -s "$OUTPUT_DIR/best_model.pth" ]]; then
  echo "[SKIP] Existing checkpoint: $OUTPUT_DIR/best_model.pth"
  exit 0
fi

echo "Model 3 frontier training"
echo "  task:       ${SLURM_ARRAY_TASK_ID}"
echo "  model:      ${MODEL}"
echo "  size:       ${SIZE}"
echo "  tau:        ${TAU}"
echo "  batch size: 1024"
echo "  output:     ${OUTPUT_DIR}"

exec python train.py \
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
