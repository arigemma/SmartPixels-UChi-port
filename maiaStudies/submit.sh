#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$(git branch --show-current)" != "maia-studies" ]]; then
  echo "[ERROR] Checkout maia-studies before submitting." >&2
  exit 2
fi

export MAIA_DATA_ROOT="${MAIA_DATA_ROOT:-/scratch/gpfs/IOJALVO/mb7126/SmartPixels/22016562}"
export MAIA_CACHE_ROOT="${MAIA_CACHE_ROOT:-$MAIA_DATA_ROOT/derived/maia_lnn_v1}"
export MAIA_RESULT_ROOT="${MAIA_RESULT_ROOT:-$PWD/results/SLURM/2026_10_07_maia_signal_vs_bib_lnn_5h_20k}"
export MAIA_TIMING_COLUMN="${MAIA_TIMING_COLUMN:-adjusted_hit_time}"

mkdir -p logs "$MAIA_RESULT_ROOT"
python -m unittest discover -s maiaStudies/tests -p 'test_*.py'
python -m maiaStudies.campaign --validate
python -m maiaStudies.preflight
python - <<'PY'
import os
from pathlib import Path
from maiaStudies.data import SOURCES, locate_source_files
root = Path(os.environ["MAIA_DATA_ROOT"])
for source in SOURCES:
    recon, labels = locate_source_files(root, source)
    print(f"[DATA] {source}: {recon} | {labels}")
PY

PREPARE_JOB="$(sbatch --parsable --chdir="$PWD" \
  --export="ALL,MAIA_DATA_ROOT=$MAIA_DATA_ROOT,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_TIMING_COLUMN=$MAIA_TIMING_COLUMN" \
  maiaStudies/prepare.slurm)"
PREPARE_JOB="${PREPARE_JOB%%;*}"

TRAIN_JOB="$(sbatch --parsable --dependency="afterok:$PREPARE_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/train.slurm)"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/evaluate.slurm)"
EVAL_JOB="${EVAL_JOB%%;*}"

SUMMARY_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" \
  maiaStudies/summarize.slurm)"
SUMMARY_JOB="${SUMMARY_JOB%%;*}"

cat > "$MAIA_RESULT_ROOT/submitted_jobs.txt" <<EOF
prepare_job_id=$PREPARE_JOB
training_job_id=$TRAIN_JOB
evaluation_job_id=$EVAL_JOB
summary_job_id=$SUMMARY_JOB
git_commit=$(git rev-parse HEAD)
data_root=$MAIA_DATA_ROOT
cache_root=$MAIA_CACHE_ROOT
timing_column=$MAIA_TIMING_COLUMN
result_root=$MAIA_RESULT_ROOT
EOF

echo "Preparation job: $PREPARE_JOB (streaming projection/cache; safely skips if complete)"
echo "Training array:  $TRAIN_JOB (30 tasks: 5 dense + 5 conv x 3 taus)"
echo "Evaluation:      $EVAL_JOB (full test and [-0.5,15] ns timing-cut test)"
echo "Summary:         $SUMMARY_JOB"
echo "Campaign root:   $MAIA_RESULT_ROOT"
echo
echo "Final products:"
echo "  $MAIA_RESULT_ROOT/performance_summary.tsv"
echo "  $MAIA_RESULT_ROOT/model_comparison_wide.tsv"
echo "  $MAIA_RESULT_ROOT/best_models.tsv"
echo "  $MAIA_RESULT_ROOT/SUMMARY.md"
echo "  $MAIA_RESULT_ROOT/summary_plots/ (18 figures, PNG and PDF)"
echo "  $MAIA_RESULT_ROOT/maia_lnn_screening.png"
echo "  $MAIA_RESULT_ROOT/maia_lnn_screening.pdf"
