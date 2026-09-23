#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$(git branch --show-current)" != "wip-mila-rtl-handoff" ]]; then
  echo "[ERROR] Checkout wip-mila-rtl-handoff before submitting." >&2
  exit 1
fi

mkdir -p logs
python -m unittest rtlHandoffStudy.test_campaign
python -m rtlHandoffStudy.campaign --validate
python -m rtlHandoffStudy.smoke

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
mkdir -p "$ROOT"

TRAIN_JOB="$(sbatch --parsable --chdir="$PWD" --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/train.slurm)"
TRAIN_JOB="${TRAIN_JOB%%;*}"
EVAL_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --chdir="$PWD" --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/evaluate.slurm)"
EVAL_JOB="${EVAL_JOB%%;*}"
PACKAGE_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --chdir="$PWD" --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/package.slurm)"
PACKAGE_JOB="${PACKAGE_JOB%%;*}"

cat > "$ROOT/submitted_jobs.txt" <<EOF
training_job_id=$TRAIN_JOB
evaluation_job_id=$EVAL_JOB
package_job_id=$PACKAGE_JOB
git_commit=$(git rev-parse HEAD)
result_root=$ROOT
EOF

echo "Training array:   $TRAIN_JOB (20 tasks: 10 dense + 10 conv)"
echo "Evaluation array: $EVAL_JOB (waits for all training tasks)"
echo "Packaging job:    $PACKAGE_JOB (waits for all evaluations)"
echo "Campaign root:    $ROOT"
echo
python -m rtlHandoffStudy.campaign --print-table
echo
echo "Final summary:    $ROOT/balanced_accuracy_summary.tsv"
echo "Transfer bundle:  $ROOT/transfer_ready/"
