#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$(git branch --show-current)" != "maia-studies" ]]; then
  echo "[ERROR] Checkout maia-studies before submitting." >&2
  exit 2
fi

export MAIA_DATA_ROOT="${MAIA_DATA_ROOT:-/scratch/gpfs/IOJALVO/mb7126/SmartPixels/22016562}"
export MAIA_CACHE_ROOT="${MAIA_CACHE_ROOT:-$MAIA_DATA_ROOT/derived/maia_lnn_yprofile_timing8_v3}"
export MAIA_RESULT_ROOT="${MAIA_PILOT_ROOT:-$PWD/results/SLURM/2026_10_21_maia_signal_vs_bib_canonical_inputs_pilot_v1}"
export MAIA_TIMING_COLUMN="${MAIA_TIMING_COLUMN:-adjusted_hit_time}"

mkdir -p logs "$MAIA_RESULT_ROOT"
python -m unittest discover -s maiaStudies/tests -p 'test_*.py'
python -m maiaStudies.campaign --validate
python -m maiaStudies.preflight

PREPARE_JOB="$(sbatch --parsable --chdir="$PWD" \
  --export="ALL,MAIA_DATA_ROOT=$MAIA_DATA_ROOT,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_TIMING_COLUMN=$MAIA_TIMING_COLUMN" \
  maiaStudies/prepare.slurm)"
PREPARE_JOB="${PREPARE_JOB%%;*}"

TRAIN_JOB="$(sbatch --parsable --array=7,22 --dependency="afterok:$PREPARE_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT,MAIA_MINIMUM_STEPS=5000,MAIA_MINIMUM_TRAINING_HOURS=0.5,MAIA_EVAL_FREQUENCY=1000,MAIA_VALIDATION_EXAMPLES=50000" \
  maiaStudies/train.slurm)"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$(sbatch --parsable --array=7,22 --dependency="afterok:$TRAIN_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_CACHE_ROOT=$MAIA_CACHE_ROOT,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT,MAIA_MAXIMUM_VALIDATION_EXAMPLES=100000,MAIA_MAXIMUM_TEST_EXAMPLES=100000" \
  maiaStudies/evaluate.slurm)"
EVAL_JOB="${EVAL_JOB%%;*}"

GATE_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --chdir="$PWD" \
  --export="ALL,MAIA_RESULT_ROOT=$MAIA_RESULT_ROOT" maiaStudies/pilot_gate.slurm)"
GATE_JOB="${GATE_JOB%%;*}"

cat > "$MAIA_RESULT_ROOT/submitted_jobs.txt" <<EOF
prepare_job_id=$PREPARE_JOB
training_job_id=$TRAIN_JOB
evaluation_job_id=$EVAL_JOB
gate_job_id=$GATE_JOB
git_commit=$(git rev-parse HEAD)
data_root=$MAIA_DATA_ROOT
cache_root=$MAIA_CACHE_ROOT
result_root=$MAIA_RESULT_ROOT
EOF

echo "Preparation: $PREPARE_JOB (canonical 14/105-input v3 cache)"
echo "Pilot train: $TRAIN_JOB (dense14_s03_tau20 and conv105_s03_tau20)"
echo "Pilot eval:  $EVAL_JOB (100k validation + 100k test examples)"
echo "Pilot gate:  $GATE_JOB (must exceed chance before full submission)"
echo "Pilot root:  $MAIA_RESULT_ROOT"
echo
echo "After it passes: bash maiaStudies/submit.sh"
