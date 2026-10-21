#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RESULT_ROOT="${1:-${MAIA_RESULT_ROOT:-$PWD/results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k}}"
CAMPAIGN="${MAIA_CAMPAIGN:-main}"

if [[ ! -d "$RESULT_ROOT" ]]; then
  echo "[ERROR] Campaign directory does not exist: $RESULT_ROOT" >&2
  echo "Usage: bash maiaStudies/plot_summary.sh /path/to/campaign" >&2
  exit 2
fi

ARGS=(
  --result-root "$RESULT_ROOT"
  --campaign "$CAMPAIGN"
  --output-dir "$RESULT_ROOT/summary_plots"
  --formats png pdf
  --dpi "${MAIA_PLOT_DPI:-600}"
  --font-scale "${MAIA_PLOT_FONT_SCALE:-1.1}"
)

if [[ "${MAIA_ALLOW_PARTIAL:-0}" == "1" ]]; then
  ARGS+=(--allow-partial)
fi

if [[ "${MAIA_ALLOW_COLLAPSED:-0}" == "1" ]]; then
  ARGS+=(--allow-collapsed)
fi

if [[ "${MAIA_PLOT_ANNOTATIONS:-1}" == "0" ]]; then
  ARGS+=(--no-annotations)
fi

python -m maiaStudies.summarize "${ARGS[@]}"

echo
echo "Main dashboard: $RESULT_ROOT/summary_plots/17_campaign_dashboard.png"
echo "All figures:    $RESULT_ROOT/summary_plots"
echo "All numbers:    $RESULT_ROOT/model_comparison_wide.tsv"
echo "Best models:    $RESULT_ROOT/best_models.tsv"
echo "Text summary:   $RESULT_ROOT/SUMMARY.md"
