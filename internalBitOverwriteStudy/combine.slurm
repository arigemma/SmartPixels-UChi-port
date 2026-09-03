#!/bin/bash
#SBATCH --job-name=internalOverwritePlot
#SBATCH --output=logs/internalOverwritePlot_%j.out
#SBATCH --error=logs/internalOverwritePlot_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"
: "${INTERNAL_OVERWRITE_ROOT:?Missing campaign output root}"

python -m internalBitOverwriteStudy.campaign \
  --combine --root "$INTERNAL_OVERWRITE_ROOT"
