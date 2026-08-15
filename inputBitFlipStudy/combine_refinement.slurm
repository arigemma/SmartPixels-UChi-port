#!/bin/bash
#SBATCH --job-name=inputFaultRefinePlot
#SBATCH --output=logs/inputFaultRefinePlot_%j.out
#SBATCH --error=logs/inputFaultRefinePlot_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${INPUT_FAULT_ROOT:?Missing campaign output root}"

REFINEMENT_MODE="${INPUT_FAULT_REFINEMENT_MODE:-low}"
if [[ "$REFINEMENT_MODE" == "low" ]]; then
  COMBINE_ACTION="--combine-refinement"
elif [[ "$REFINEMENT_MODE" == "paper" ]]; then
  COMBINE_ACTION="--combine-paper-refinement"
else
  echo "Unknown refinement mode: ${REFINEMENT_MODE}" >&2
  exit 2
fi

python -m inputBitFlipStudy.campaign \
  "$COMBINE_ACTION" \
  --root "$INPUT_FAULT_ROOT"
