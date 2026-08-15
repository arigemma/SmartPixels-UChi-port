#!/bin/bash
set -euo pipefail

INPUT_FAULT_ROOT="${1:-results/SLURM/2026_08_13_selected_input_bit_flip_1to10pct_eval60_linear}"
PREREQUISITE="${INPUT_FAULT_ROOT}/input_bit_flip_campaign_refined_0to10pct_summary.json"
PAPER_SUMMARY="${INPUT_FAULT_ROOT}/input_bit_flip_campaign_paper_0to50pct_summary.json"

if [[ ! -f "$PREREQUISITE" ]]; then
  echo "Completed 0--10% refinement does not exist: ${PREREQUISITE}" >&2
  exit 2
fi
if [[ -e "$PAPER_SUMMARY" ]]; then
  echo "Refusing to overwrite completed paper refinement: ${PAPER_SUMMARY}" >&2
  exit 2
fi

EXPECTED_TASKS="$(
  python -m inputBitFlipStudy.campaign \
    --print-paper-refinement-task-count
)"
REFINED_BASE_RESULTS="$(
  find "$INPUT_FAULT_ROOT" -name bit_flip_results_refined_0to10pct.csv \
    -type f | wc -l
)"
if [[ "$EXPECTED_TASKS" -ne 22 ]]; then
  echo "Unexpected paper-refinement task count: ${EXPECTED_TASKS}" >&2
  exit 2
fi
if [[ "$REFINED_BASE_RESULTS" -ne 22 ]]; then
  echo "Refusing paper refinement: found ${REFINED_BASE_RESULTS}/22 refined base results." >&2
  exit 2
fi

mkdir -p logs
export INPUT_FAULT_ROOT
export INPUT_FAULT_REFINEMENT_MODE=paper
export PYTHONUNBUFFERED=1

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((EXPECTED_TASKS - 1))" \
    --export=ALL \
    inputBitFlipStudy/run_refinement_evaluation.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted five-point paper refinement ${EVALUATION_JOB_ID} (22 tasks)"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    inputBitFlipStudy/combine_refinement.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent paper combine/plot job ${COMBINE_JOB_ID}"
echo "Campaign root: ${INPUT_FAULT_ROOT}"
echo "New target percentages: 1.25 3.75 12.5 17.5 22.5%"
echo "Paper plot x-axis: 0--50%"
