#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"

RESULT_ROOT="${1:-results/SLURM/2026_08_24_model3lgnConv_frontier_extension}"
EVAL_ROOT="${2:-results/evaluation/2026_08_24_model3Frontier_discrete_balacc}"
TAUS=(10 20 40)
SIZES=(
  s01_p003M s02_p007M s03_p012M s04_p018M s05_p026M
  s06_p039M s07_p069M s08_p110M s09_p213M s10_p385M
)

if [[ ! -d "$RESULT_ROOT" ]]; then
  echo "[ERROR] Training root does not exist: ${RESULT_ROOT}" >&2
  exit 2
fi

TASK_IDS=()
RECOVERY_SIZES=()
for SIZE_INDEX in "${!SIZES[@]}"; do
  SIZE="${SIZES[$SIZE_INDEX]}"
  SIZE_INCOMPLETE=0
  for TAU in "${TAUS[@]}"; do
    if [[ ! -s "$RESULT_ROOT/$SIZE/tau_$TAU/best_model.pth" ]]; then
      SIZE_INCOMPLETE=1
    fi
  done
  if [[ "$SIZE_INCOMPLETE" -eq 1 ]]; then
    RECOVERY_SIZES+=("$SIZE")
    for TAU_INDEX in "${!TAUS[@]}"; do
      TASK_IDS+=("$((SIZE_INDEX * 3 + TAU_INDEX))")
    done
  fi
done

ABSOLUTE_RESULT_ROOT="$(realpath "$RESULT_ROOT")"
ABSOLUTE_EVAL_ROOT="$(realpath -m "$EVAL_ROOT")"
mkdir -p logs

if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
  echo "All 30 checkpoints are present; submitting strict evaluation only."
  exec sbatch \
    --chdir="$REPOSITORY_ROOT" \
    --export="ALL,MODEL3_FRONTIER_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT},MODEL3_FRONTIER_EVAL_ROOT=${ABSOLUTE_EVAL_ROOT}" \
    "$REPOSITORY_ROOT/eval_model3_frontier.slurm"
fi

ARRAY_SPEC="$(IFS=,; echo "${TASK_IDS[*]}")"
TRAIN_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --array="$ARRAY_SPEC" \
    --export="ALL,MODEL3_FRONTIER_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT}" \
    "$REPOSITORY_ROOT/train_model3_frontier.slurm"
})"
TRAIN_JOB="${TRAIN_JOB%%;*}"

EVAL_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --dependency="afterany:${TRAIN_JOB}" \
    --export="ALL,MODEL3_FRONTIER_RESULT_ROOT=${ABSOLUTE_RESULT_ROOT},MODEL3_FRONTIER_EVAL_ROOT=${ABSOLUTE_EVAL_ROOT}" \
    "$REPOSITORY_ROOT/eval_model3_frontier.slurm"
})"
EVAL_JOB="${EVAL_JOB%%;*}"

echo "Recovering complete tau triplets for: ${RECOVERY_SIZES[*]}"
echo "Submitted recovery array ${TRAIN_JOB}: ${ARRAY_SPEC}"
echo "Submitted strict dependent evaluation ${EVAL_JOB}"
