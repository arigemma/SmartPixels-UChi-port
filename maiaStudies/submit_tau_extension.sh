#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$(git branch --show-current)" != "maia-studies" ]]; then
  echo "[ERROR] Checkout maia-studies before submitting." >&2
  exit 2
fi

export MAIA_DATA_ROOT="${MAIA_DATA_ROOT:-/scratch/gpfs/IOJALVO/mb7126/SmartPixels/22016562}"
export MAIA_CACHE_ROOT="${MAIA_CACHE_ROOT:-$MAIA_DATA_ROOT/derived/maia_lnn_spatial_v2}"
export MAIA_RESULT_ROOT="${MAIA_RESULT_ROOT:-$PWD/results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k}"
export MAIA_TIMING_COLUMN="${MAIA_TIMING_COLUMN:-adjusted_hit_time}"

if [[ ! -s "$MAIA_CACHE_ROOT/preparation_complete.json" ]]; then
  echo "[ERROR] Prepared MAIA cache is missing: $MAIA_CACHE_ROOT" >&2
  exit 2
fi

python - "$MAIA_RESULT_ROOT" <<'PY'
import sys
from pathlib import Path
from maiaStudies.campaign import get_campaign_configs

root = Path(sys.argv[1])
missing = [
    config.alias
    for config in get_campaign_configs("main")
    if not (root / config.alias / "evaluation_complete.json").is_file()
]
if missing:
    raise SystemExit(
        "[ERROR] Original 30-model campaign is incomplete; missing:\n  "
        + "\n  ".join(missing)
    )
print("[MAIN CAMPAIGN] All 30 evaluations are present.")
PY

mkdir -p logs "$MAIA_RESULT_ROOT"
python -m unittest discover -s maiaStudies/tests -p 'test_*.py'
python -m maiaStudies.campaign --campaign tau_extension --validate
python -m maiaStudies.preflight

TRAIN_JOB="$(sbatch --parsable --array=0-19%6 --chdir="$PWD" \
  --export="ALL,MAIA_CAMPAIGN=tau_extension,MAIA_TRACK_RELAXED_VALIDATION=1,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/train.slurm)"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$(sbatch --parsable --array=0-19%8 --dependency="afterok:$TRAIN_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CAMPAIGN=tau_extension,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/evaluate.slurm)"
EVAL_JOB="${EVAL_JOB%%;*}"

SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CAMPAIGN=all,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/summarize.slurm)"
SUMMARY_JOB="${SUMMARY_JOB%%;*}"

cat > "$MAIA_RESULT_ROOT/submitted_tau_extension_jobs.txt" <<EOF
training_job_id=$TRAIN_JOB
evaluation_job_id=$EVAL_JOB
summary_job_id=$SUMMARY_JOB
git_commit=$(git rev-parse HEAD)
campaign=tau_extension
taus=1,3
cache_root=$MAIA_CACHE_ROOT
result_root=$MAIA_RESULT_ROOT
track_relaxed_validation=1
EOF

echo "Tau-extension training:   $TRAIN_JOB (20 tasks: 5 dense + 5 conv x tau=1,3)"
echo "Tau-extension evaluation: $EVAL_JOB"
echo "Combined 50-model plots:  $SUMMARY_JOB"
echo "Campaign root:            $MAIA_RESULT_ROOT"
echo
echo "Each new model writes training_history.tsv with paired relaxed and discrete validation metrics."
echo "Final dynamics plot: $MAIA_RESULT_ROOT/summary_plots/19_relaxed_vs_discrete_training.png"
