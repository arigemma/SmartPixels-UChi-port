#!/bin/bash
#SBATCH --job-name=qkerasFaultCombine
#SBATCH --output=logs/qkerasFaultCombine_%j.out
#SBATCH --error=logs/qkerasFaultCombine_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${QKERAS_SCOPE_ROOT:?Missing sweep output root}"
: "${QKERAS_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"
QKERAS_SCOPE_MODELS="${QKERAS_SCOPE_MODELS:-all}"

python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
  --models "$QKERAS_SCOPE_MODELS" \
  --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
  --combine \
  --root "$QKERAS_SCOPE_ROOT" \
  --fit-exponential-decay
