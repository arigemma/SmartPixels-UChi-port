#!/bin/bash
#SBATCH --job-name=s08FaultScopeCombine
#SBATCH --output=logs/s08FaultScopeCombine_%j.out
#SBATCH --error=logs/s08FaultScopeCombine_%j.err
#SBATCH --time=0:25:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${S08_SCOPE_ROOT:?Missing sweep output root}"
: "${S08_SCOPE_TRAINING_PERCENTAGES:?Missing training percentages}"
: "${S08_WHOLE_SCOPE_ROOT:?Missing reusable whole-model sweep root}"

python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
  --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
  --combine \
  --root "$S08_SCOPE_ROOT" \
  --whole-scope-root "$S08_WHOLE_SCOPE_ROOT" \
  --fit-exponential-decay
