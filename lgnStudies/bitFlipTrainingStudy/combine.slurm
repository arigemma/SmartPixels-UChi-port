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

python lgnStudies/bitFlipStudy/combine_results.py \
  --root "$BIT_FLIP_TRAINING_ROOT" \
  --title-context "Fault-aware training: ${BIT_FLIP_TRAINING_PERCENTAGE}% per activation site"
