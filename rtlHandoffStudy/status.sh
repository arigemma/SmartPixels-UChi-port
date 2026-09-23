#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
echo "Campaign root: $ROOT"
echo "Training complete: $(find "$ROOT" -mindepth 2 -maxdepth 2 -name training_complete.json -size +0c 2>/dev/null | wc -l) / 20"
echo "Test metrics:      $(find "$ROOT" -mindepth 2 -maxdepth 2 -name test_metrics.json -size +0c 2>/dev/null | wc -l) / 20"
echo "Portable models:   $(find "$ROOT" -mindepth 2 -maxdepth 2 -name best_model.importable.pth -size +0c 2>/dev/null | wc -l) / 20"

if [[ -s "$ROOT/submitted_jobs.txt" ]]; then
  # This file is emitted by submit.sh and contains only shell-safe scalar values.
  source "$ROOT/submitted_jobs.txt"
  JOBS="$training_job_id,$evaluation_job_id,$package_job_id"
  echo
  sacct -X -j "$JOBS" \
    --format=JobID%24,JobName%24,State%22,ExitCode%12,Elapsed
fi

if [[ -s "$ROOT/balanced_accuracy_summary.tsv" ]]; then
  echo
  column -s $'\t' -t < "$ROOT/balanced_accuracy_summary.tsv"
else
  echo
  echo "Summary is not ready. Use rtlHandoffStudy/recover.sh after the active jobs finish if any task failed or timed out."
fi
