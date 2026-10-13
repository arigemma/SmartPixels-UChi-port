#!/bin/bash
set -euo pipefail

ROOT="${1:-${MAIA_RESULT_ROOT:-$PWD/results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k}}"
EXPECTED=30
if [[ -s "$ROOT/submitted_jobs.txt" ]] && grep -q '^gate_job_id=' "$ROOT/submitted_jobs.txt"; then
  EXPECTED=2
fi
echo "Campaign: $ROOT"
echo "Training complete:   $(find "$ROOT" -mindepth 2 -maxdepth 2 -name training_complete.json -size +0c 2>/dev/null | wc -l) / $EXPECTED"
echo "Evaluation complete: $(find "$ROOT" -mindepth 2 -maxdepth 2 -name evaluation_complete.json -size +0c 2>/dev/null | wc -l) / $EXPECTED"
echo "Best checkpoints:    $(find "$ROOT" -mindepth 2 -maxdepth 2 -name best_model.pth -size +0c 2>/dev/null | wc -l) / $EXPECTED"
if [[ -s "$ROOT/submitted_jobs.txt" ]]; then
  echo
  cat "$ROOT/submitted_jobs.txt"
fi
if [[ -s "$ROOT/performance_summary.tsv" ]]; then
  echo
  column -s $'\t' -t < "$ROOT/performance_summary.tsv"
fi
