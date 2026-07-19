#!/bin/bash
#SBATCH --job-name=s08FaultXPlot
#SBATCH --output=logs/s08FaultXPlot_%j.out
#SBATCH --error=logs/s08FaultXPlot_%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"

S08_MAX_EVALUATION_PERCENTAGE="${S08_MAX_EVALUATION_PERCENTAGE:-7}"
export S08_MAX_EVALUATION_PERCENTAGE
S08_FINAL_MAX_TRAINING_PERCENTAGE="${S08_FINAL_MAX_TRAINING_PERCENTAGE:-$S08_MAX_EVALUATION_PERCENTAGE}"
S08_FINAL_TRAINING_PERCENTAGES="${S08_FINAL_TRAINING_PERCENTAGES:-}"
S08_FINAL_OUTPUT_DIR="${S08_FINAL_OUTPUT_DIR:-temp}"
S08_FINAL_SEARCH_ROOTS="${S08_FINAL_SEARCH_ROOTS:-results/SLURM}"

read -r -a SEARCH_ROOTS <<< "$S08_FINAL_SEARCH_ROOTS"
SEARCH_ARGS=()
for root in "${SEARCH_ROOTS[@]}"; do
  SEARCH_ARGS+=(--search-root "$root")
done
TRAINING_ARGS=()
if [[ -n "$S08_FINAL_TRAINING_PERCENTAGES" ]]; then
  TRAINING_ARGS+=(--training-percentages "$S08_FINAL_TRAINING_PERCENTAGES")
fi

python -m lgnStudies.bitFlipTrainingStudy.final_s08_plots \
  "${SEARCH_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  --output-dir "$S08_FINAL_OUTPUT_DIR" \
  --max-training-percentage "$S08_FINAL_MAX_TRAINING_PERCENTAGE" \
  --max-inference-percentage "$S08_MAX_EVALUATION_PERCENTAGE" \
  --expected-chunks 7
