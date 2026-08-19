#!/bin/bash
set -euo pipefail

DENSE_ROOT="${1:-results/SLURM/2026_08_14_model2lgnFull_left_extension_fixed}"
CONV_ROOT="${2:-results/SLURM/2026_08_14_model3lgnConv_curve_extension_fixed}"
EVAL_ROOT="${3:-results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc}"

TAUS=(10 20 40)
CONV_SIZES=(
  s01_p046M s02_p074M s03_p109M s04_p152M s05_p201M
  s06_p1325M s07_p1608M s08_p1919M s09_p2257M s10_p2623M
)

count_checkpoints() {
  local root="$1"
  find "$root" -mindepth 3 -maxdepth 3 \
    -path '*/tau_*/best_model.pth' -size +0c | wc -l
}

DENSE_COUNT="$(count_checkpoints "$DENSE_ROOT")"
if [[ "$DENSE_COUNT" -ne 21 ]]; then
  echo "Refusing convolutional recovery because the dense grid is partial: ${DENSE_COUNT}/21" >&2
  exit 2
fi

TASK_IDS=()
RECOVERY_SIZES=()
for SIZE_INDEX in "${!CONV_SIZES[@]}"; do
  SIZE="${CONV_SIZES[$SIZE_INDEX]}"
  SIZE_INCOMPLETE=0
  for TAU in "${TAUS[@]}"; do
    if [[ ! -s "$CONV_ROOT/$SIZE/tau_$TAU/best_model.pth" ]]; then
      SIZE_INCOMPLETE=1
    fi
  done

  if [[ "$SIZE_INCOMPLETE" -eq 1 ]]; then
    RECOVERY_SIZES+=("$SIZE")
    # The first convolutional size begins at global architecture index 7.
    # Rerun the complete tau triplet for a fair within-size comparison.
    for TAU_INDEX in "${!TAUS[@]}"; do
      TASK_IDS+=("$(( (SIZE_INDEX + 7) * 3 + TAU_INDEX ))")
    done
  fi
done

if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
  CONV_COUNT="$(count_checkpoints "$CONV_ROOT")"
  if [[ "$CONV_COUNT" -ne 30 ]]; then
    echo "No incomplete known size was found, but the convolutional grid is ${CONV_COUNT}/30." >&2
    exit 2
  fi
  echo "The training grid is already complete; submitting evaluation only."
  exec bash submit_pareto_extension_evaluation.sh \
    "$DENSE_ROOT" "$CONV_ROOT" "$EVAL_ROOT"
fi

ARRAY_SPEC="$(IFS=,; echo "${TASK_IDS[*]}")"
mkdir -p logs

echo "Recovering complete tau triplets for: ${RECOVERY_SIZES[*]}"
echo "Slurm array tasks: ${ARRAY_SPEC}"
echo "Host memory: 48G"
echo "Large-convolution batch size: 512"

TRAIN_JOB_ID="$(
  sbatch \
    --parsable \
    --array="$ARRAY_SPEC" \
    --mem=48G \
    --export="ALL,PARETO_DENSE_OUTPUT_BASE=${DENSE_ROOT},PARETO_CONV_OUTPUT_BASE=${CONV_ROOT},PARETO_FRESH_RETRY=1" \
    run.slurm
)"
TRAIN_JOB_ID="${TRAIN_JOB_ID%%;*}"

# Use afterany because the strict evaluator itself verifies all 51 checkpoints.
# This also permits evaluation after a training task reaches its Slurm time
# limit but has already handled the advance signal and published its model.
EVAL_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterany:${TRAIN_JOB_ID}" \
    --export="ALL,PARETO_DENSE_RESULT_ROOT=${DENSE_ROOT},PARETO_CONV_RESULT_ROOT=${CONV_ROOT},PARETO_EXTENSION_EVAL_ROOT=${EVAL_ROOT}" \
    eval_lgn_pareto_extensions.slurm
)"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"

echo "Submitted recovery array ${TRAIN_JOB_ID} (${ARRAY_SPEC})"
echo "Submitted strict dependent evaluation ${EVAL_JOB_ID} (tasks 0-1)"
echo "Evaluation root: ${EVAL_ROOT}"
