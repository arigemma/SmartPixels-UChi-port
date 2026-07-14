#!/bin/bash
#SBATCH --job-name=s08FaultAwarePctCombine
#SBATCH --output=logs/s08FaultAwarePctCombine_%j.out
#SBATCH --error=logs/s08FaultAwarePctCombine_%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${S08_SWEEP_ROOT:?Missing sweep output root}"
: "${S08_SWEEP_TRAINING_PERCENTAGES:?Missing training percentages}"

python -m lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep \
  --training-percentages "$S08_SWEEP_TRAINING_PERCENTAGES" \
  --combine \
  --root "$S08_SWEEP_ROOT" \
  --fit-exponential-decay
