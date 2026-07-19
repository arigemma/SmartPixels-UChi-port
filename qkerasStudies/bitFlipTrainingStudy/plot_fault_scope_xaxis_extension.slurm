#!/bin/bash
#SBATCH --job-name=qkerasFaultXPlot
#SBATCH --output=logs/qkerasFaultXPlot_%j.out
#SBATCH --error=logs/qkerasFaultXPlot_%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${QKERAS_SCOPE_ROOT:?Missing sweep output root}"
: "${QKERAS_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"
QKERAS_SCOPE_MODELS="${QKERAS_SCOPE_MODELS:-all}"
QKERAS_FINAL_OUTPUT_DIR="${QKERAS_FINAL_OUTPUT_DIR:-temp}"
S08_MAX_EVALUATION_PERCENTAGE="${S08_MAX_EVALUATION_PERCENTAGE:-7}"

python -m qkerasStudies.bitFlipTrainingStudy.final_fault_scope_plots \
  --root "$QKERAS_SCOPE_ROOT" \
  --models "$QKERAS_SCOPE_MODELS" \
  --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
  --output-dir "$QKERAS_FINAL_OUTPUT_DIR" \
  --max-inference-percentage "$S08_MAX_EVALUATION_PERCENTAGE"
