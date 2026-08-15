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

python -m inputBitFlipStudy.campaign \
  --combine-refinement \
  --root "$INPUT_FAULT_ROOT"
