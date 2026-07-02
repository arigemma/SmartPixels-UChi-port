#!/bin/bash
set -euo pipefail

RESOURCE_JOB_ID="${1:-}"
RESOURCE_ROOT="results/SLURM/2026_07_02_model2_resource_precision"
MODELS=(
  "s06_p760M_tau40"
  "s08_p1202M_tau40"
  "s09_p1460M_tau40"
)

SBATCH_DEPENDENCY=()
if [[ -n "$RESOURCE_JOB_ID" ]]; then
  RESOURCE_JOB_ID="${RESOURCE_JOB_ID%%;*}"
  if [[ ! "$RESOURCE_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Invalid resource job ID: ${RESOURCE_JOB_ID}" >&2
    exit 2
  fi
  SBATCH_DEPENDENCY=(--dependency="afterok:${RESOURCE_JOB_ID}")
  echo "Bit-flip jobs will wait for resource job ${RESOURCE_JOB_ID}"
else
  MISSING=()
  for MODEL in "${MODELS[@]}"; do
    CHECKPOINT="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
    if [[ ! -f "$CHECKPOINT" ]]; then
      MISSING+=("$CHECKPOINT")
    fi
  done
  if (( ${#MISSING[@]} > 0 )); then
    echo "Cannot submit bit-flip jobs: clean checkpoints are missing:" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    echo >&2
    echo "Wait for the resource array to finish, or pass its Slurm job ID:" >&2
    echo "  bash lgnStudies/bitFlipStudy/submit.sh RESOURCE_JOB_ID" >&2
    exit 2
  fi
fi

ARRAY_JOB_ID="$(
  sbatch \
    --parsable \
    "${SBATCH_DEPENDENCY[@]}" \
    lgnStudies/bitFlipStudy/run.slurm
)"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"
echo "Submitted bit-flip array job ${ARRAY_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    lgnStudies/bitFlipStudy/combine.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent plot job ${COMBINE_JOB_ID}"
