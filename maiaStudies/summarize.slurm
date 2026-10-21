#!/bin/bash
#SBATCH --job-name=maiaSummarize
#SBATCH --output=logs/maiaSummarize_%j.out
#SBATCH --error=logs/maiaSummarize_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${MAIA_RESULT_ROOT:?MAIA_RESULT_ROOT is required}"
bash maiaStudies/plot_summary.sh "$MAIA_RESULT_ROOT"

if [[ "${MAIA_CAMPAIGN:-main}" == "all" ]]; then
  python -m maiaStudies.training_dynamics \
    --result-root "$MAIA_RESULT_ROOT" \
    --output-dir "$MAIA_RESULT_ROOT/summary_plots" \
    --formats png pdf \
    --dpi "${MAIA_PLOT_DPI:-600}"
fi
