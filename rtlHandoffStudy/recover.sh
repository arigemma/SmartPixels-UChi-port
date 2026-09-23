#!/bin/bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ "$(git branch --show-current)" != "wip-mila-rtl-handoff" ]]; then
  echo "[ERROR] Checkout wip-mila-rtl-handoff before recovering." >&2
  exit 1
fi

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
mkdir -p logs "$ROOT"

mapfile -t ALIASES < <(python -c 'from models.rtl_handoff import RTL_HANDOFF_CONFIGS; print("\n".join(str(c["alias"]) for c in RTL_HANDOFF_CONFIGS))')
MISSING_TRAIN=()
MISSING_EVAL=()
for TASK in "${!ALIASES[@]}"; do
  ALIAS="${ALIASES[$TASK]}"
  if [[ ! -s "$ROOT/$ALIAS/training_complete.json" ]]; then
    MISSING_TRAIN+=("$TASK")
  elif [[ ! -s "$ROOT/$ALIAS/test_metrics.json" ]]; then
    MISSING_EVAL+=("$TASK")
  fi
done

join_by_comma() {
  local IFS=,
  echo "$*"
}

TRAIN_JOB="none"
EVAL_JOB="none"
if (( ${#MISSING_TRAIN[@]} > 0 )); then
  TRAIN_SPEC="$(join_by_comma "${MISSING_TRAIN[@]}")%6"
  TRAIN_JOB="$(sbatch --parsable --array="$TRAIN_SPEC" --chdir="$PWD" \
    --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/train.slurm)"
  TRAIN_JOB="${TRAIN_JOB%%;*}"
  # Evaluate the full array after recovery; already-complete evaluations skip quickly.
  EVAL_JOB="$(sbatch --parsable --array=0-19%10 --dependency="afterok:$TRAIN_JOB" \
    --chdir="$PWD" --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" \
    rtlHandoffStudy/evaluate.slurm)"
  EVAL_JOB="${EVAL_JOB%%;*}"
elif (( ${#MISSING_EVAL[@]} > 0 )); then
  EVAL_SPEC="$(join_by_comma "${MISSING_EVAL[@]}")%10"
  EVAL_JOB="$(sbatch --parsable --array="$EVAL_SPEC" --chdir="$PWD" \
    --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/evaluate.slurm)"
  EVAL_JOB="${EVAL_JOB%%;*}"
fi

if [[ "$EVAL_JOB" == "none" ]]; then
  PACKAGE_JOB="$(sbatch --parsable --chdir="$PWD" \
    --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/package.slurm)"
else
  PACKAGE_JOB="$(sbatch --parsable --dependency="afterok:$EVAL_JOB" --chdir="$PWD" \
    --export="ALL,RTL_HANDOFF_RESULT_ROOT=$ROOT" rtlHandoffStudy/package.slurm)"
fi
PACKAGE_JOB="${PACKAGE_JOB%%;*}"

cat > "$ROOT/recovery_jobs.txt" <<EOF
training_recovery_job_id=$TRAIN_JOB
evaluation_recovery_job_id=$EVAL_JOB
package_recovery_job_id=$PACKAGE_JOB
git_commit=$(git rev-parse HEAD)
result_root=$ROOT
EOF

echo "Missing training tasks:   ${MISSING_TRAIN[*]:-none}"
echo "Missing evaluation tasks: ${MISSING_EVAL[*]:-none}"
echo "Training recovery:        $TRAIN_JOB"
echo "Evaluation recovery:      $EVAL_JOB"
echo "Packaging recovery:       $PACKAGE_JOB"
