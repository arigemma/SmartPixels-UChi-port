#!/bin/bash
set -euo pipefail

RESOURCE_JOB_ID="${1:-}"
RESOURCE_ROOT="results/SLURM/2026_07_03_model2_resource_precision_5h_20k"
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
fi

MISSING=()
for MODEL in "${MODELS[@]}"; do
  CHECKPOINT="${RESOURCE_ROOT}/${MODEL}/final_model.pth"
  COMPLETION="${RESOURCE_ROOT}/${MODEL}/training_complete.json"
  if [[ ! -f "$CHECKPOINT" || ! -f "$COMPLETION" ]] || \
     ! python -m lgnStudies.training_completion \
       --checkpoint "$CHECKPOINT" \
       --status "$COMPLETION" \
       --minimum-hours 5.0 \
       --minimum-steps 20000 >/dev/null; then
    MISSING+=("$CHECKPOINT + $COMPLETION")
  fi
done

RESOURCE_IS_ACTIVE=false
if [[ -n "$RESOURCE_JOB_ID" ]]; then
  RESOURCE_QUEUE_STATE="$(squeue -h -j "$RESOURCE_JOB_ID" -o '%T' 2>/dev/null || true)"
  if [[ -n "$RESOURCE_QUEUE_STATE" ]]; then
    RESOURCE_IS_ACTIVE=true
  fi
fi

if [[ "$RESOURCE_IS_ACTIVE" == true ]]; then
  SBATCH_DEPENDENCY=(--dependency="afterok:${RESOURCE_JOB_ID}")
  echo "Bit-flip jobs will wait for active resource job ${RESOURCE_JOB_ID}"
elif (( ${#MISSING[@]} == 0 )); then
  if [[ -n "$RESOURCE_JOB_ID" ]]; then
    echo "Resource job ${RESOURCE_JOB_ID} is no longer active; all checkpoints exist."
  fi
  echo "Submitting bit-flip jobs without a resource dependency."
else
  if [[ -n "$RESOURCE_JOB_ID" ]]; then
    echo "Resource job ${RESOURCE_JOB_ID} is no longer active, but checkpoints are missing:" >&2
  else
    echo "Cannot submit bit-flip jobs: clean checkpoints are missing:" >&2
  fi
  printf '  %s\n' "${MISSING[@]}" >&2
  echo >&2
  if [[ -n "$RESOURCE_JOB_ID" ]]; then
    echo "Inspect the resource array:" >&2
    echo "  sacct -j ${RESOURCE_JOB_ID} --format=JobID,State,ExitCode" >&2
  else
    echo "Wait for the resource array, or pass its active Slurm job ID." >&2
  fi
  exit 2
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
