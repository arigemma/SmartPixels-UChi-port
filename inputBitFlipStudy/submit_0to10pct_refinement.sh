#!/bin/bash
set -euo pipefail

INPUT_FAULT_ROOT="${1:-results/SLURM/2026_08_13_selected_input_bit_flip_1to10pct_eval60_linear}"
SUMMARY="${INPUT_FAULT_ROOT}/input_bit_flip_campaign_summary.json"
REFINED_SUMMARY="${INPUT_FAULT_ROOT}/input_bit_flip_campaign_refined_0to10pct_summary.json"

if [[ ! -f "$SUMMARY" ]]; then
  echo "Completed base campaign summary does not exist: ${SUMMARY}" >&2
  exit 2
fi
if [[ -e "$REFINED_SUMMARY" ]]; then
  echo "Refusing to overwrite completed refinement: ${REFINED_SUMMARY}" >&2
  exit 2
fi

EXPECTED_TASKS="$(
  python -m inputBitFlipStudy.campaign --print-refinement-task-count
)"
COMPLETED_TRAINING="$(
  find "$INPUT_FAULT_ROOT" -name training_complete.json -type f | wc -l
)"
COMBINED_BASE_RESULTS="$(
  find "$INPUT_FAULT_ROOT" -name bit_flip_results.csv -type f \
    -not -path '*/chunks/*' -not -path '*/refinement_0to10pct/*' | wc -l
)"
if [[ "$COMPLETED_TRAINING" -ne 22 ]]; then
  echo "Refusing refinement: found ${COMPLETED_TRAINING}/22 completed training tasks." >&2
  exit 2
fi
if [[ "$COMBINED_BASE_RESULTS" -ne 22 ]]; then
  echo "Refusing refinement: found ${COMBINED_BASE_RESULTS}/22 combined base results." >&2
  exit 2
fi

mkdir -p logs
export INPUT_FAULT_ROOT
export PYTHONUNBUFFERED=1

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((EXPECTED_TASKS - 1))" \
    --export=ALL \
    inputBitFlipStudy/run_refinement_evaluation.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted low-percentage refinement array ${EVALUATION_JOB_ID} (22 tasks)"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    inputBitFlipStudy/combine_refinement.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent refined combine/plot job ${COMBINE_JOB_ID}"
echo "Campaign root: ${INPUT_FAULT_ROOT}"
echo "Additional target percentages: 2.5% and 7.5%"
