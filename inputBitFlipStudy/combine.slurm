#!/bin/bash
#SBATCH --job-name=selectedInputFaultPlot
#SBATCH --output=logs/selectedInputFaultPlot_%j.out
#SBATCH --error=logs/selectedInputFaultPlot_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

: "${INPUT_FAULT_ROOT:?Missing campaign output root}"

python -m inputBitFlipStudy.campaign \
  --combine \
  --root "$INPUT_FAULT_ROOT"
