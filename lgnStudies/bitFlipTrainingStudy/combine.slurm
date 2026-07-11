#!/bin/bash
#SBATCH --job-name=combineFaultAwarePlots
#SBATCH --output=logs/combineFaultAwarePlots_%j.out
#SBATCH --error=logs/combineFaultAwarePlots_%j.err
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${BIT_FLIP_TRAINING_ROOT:?Missing campaign output root}"
: "${BIT_FLIP_TRAINING_PERCENTAGE:?Missing training fault percentage}"

BASELINE_BIT_FLIP_ROOT="${BASELINE_BIT_FLIP_ROOT:-results/SLURM/2026_07_04_model2_bit_flip_5h_20k_dense_sweep}"

python lgnStudies/bitFlipStudy/combine_results.py \
  --root "$BIT_FLIP_TRAINING_ROOT" \
  --title-context "Fault-aware training: ${BIT_FLIP_TRAINING_PERCENTAGE}% per activation site"

if [[ -d "$BASELINE_BIT_FLIP_ROOT" ]]; then
  python -m lgnStudies.bitFlipStudy.compare_training \
    --inference-root "$BASELINE_BIT_FLIP_ROOT" \
    --fault-aware-root "$BIT_FLIP_TRAINING_ROOT" \
    --output-root "$BIT_FLIP_TRAINING_ROOT" \
    --title-context "Fault-aware training: ${BIT_FLIP_TRAINING_PERCENTAGE}% per activation site"
else
  echo "Skipping fault-aware comparison plots; missing baseline root: ${BASELINE_BIT_FLIP_ROOT}" >&2
fi
