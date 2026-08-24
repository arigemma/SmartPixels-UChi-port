#!/bin/bash
#SBATCH --job-name=hgqM2CalSummary
#SBATCH --output=logs/hgqM2CalSummary_%j.out
#SBATCH --error=logs/hgqM2CalSummary_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

python -m hgqStudies.summarize_calibrated --result-root "$HGQ_RESULT_ROOT"
