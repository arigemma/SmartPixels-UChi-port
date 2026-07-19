#!/bin/bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: bash lgnStudies/bitFlipTrainingStudy/submit_s08_xaxis_extension.sh SCOPE ROOT MAX_EVALUATION_PERCENT [--skip-baseline] TRAINING_PERCENTAGE [TRAINING_PERCENTAGE ...]" >&2
  echo "SCOPE must be 'whole' or 'input'." >&2
  echo "Example whole: bash lgnStudies/bitFlipTrainingStudy/submit_s08_xaxis_extension.sh whole results/SLURM/2026_07_17_s08_fault_aware_training_pct_sweep_10pct_5h_20k 7 0.344827586207 5.1724137931 5.51724137931 5.86206896552 6.20689655172 6.55172413793 6.89655172414" >&2
  echo "Example input extension without baseline: bash lgnStudies/bitFlipTrainingStudy/submit_s08_xaxis_extension.sh input results/SLURM/2026_07_18_s08_input_vs_whole_fault_scope_5to7pct_5h_20k 7 --skip-baseline 5.1724137931 5.51724137931 5.86206896552 6.20689655172 6.55172413793 6.89655172414" >&2
  exit 2
fi

S08_EXTENSION_SCOPE="$1"
S08_EXTENSION_ROOT="$2"
S08_MAX_EVALUATION_PERCENTAGE="$3"
shift 3
S08_EXTENSION_SKIP_BASELINE=0
if [[ "${1:-}" == "--skip-baseline" ]]; then
  S08_EXTENSION_SKIP_BASELINE=1
  shift
fi

if (( $# < 1 )); then
  echo "At least one training percentage is required." >&2
  exit 2
fi

case "$S08_EXTENSION_SCOPE" in
  whole|input) ;;
  *)
    echo "SCOPE must be 'whole' or 'input', got '${S08_EXTENSION_SCOPE}'." >&2
    exit 2
    ;;
esac

S08_EXTENSION_TRAINING_PERCENTAGES="$(
  python - "$@" <<'PY'
import math
import sys

values = []
for raw in sys.argv[1:]:
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"Training flip percentage must be numeric: {raw!r}")
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise SystemExit("Training flip percentages must lie in (0, 100].")
    values.append(value)
if len(set(values)) != len(values):
    raise SystemExit("Duplicate training flip percentages are not allowed.")
print(" ".join(f"{value:.12g}" for value in sorted(values)))
PY
)"

if [[ ! -d "$S08_EXTENSION_ROOT" ]]; then
  echo "Missing existing S08 sweep root: ${S08_EXTENSION_ROOT}" >&2
  exit 2
fi

MAX_TAG="${S08_MAX_EVALUATION_PERCENTAGE//./p}"
REPORT_PATH="${S08_EXTENSION_ROOT}/s08_${S08_EXTENSION_SCOPE}_xaxis_extension_${MAX_TAG}pct_task_report.json"
SKIP_BASELINE_ARGS=()
if [[ "$S08_EXTENSION_SKIP_BASELINE" == "1" ]]; then
  SKIP_BASELINE_ARGS+=(--skip-baseline)
fi
TASK_IDS="$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_xaxis_extension \
    --scope "$S08_EXTENSION_SCOPE" \
    --root "$S08_EXTENSION_ROOT" \
    --training-percentages "$S08_EXTENSION_TRAINING_PERCENTAGES" \
    --max-evaluation-percentage "$S08_MAX_EVALUATION_PERCENTAGE" \
    --write-report "$REPORT_PATH" \
    "${SKIP_BASELINE_ARGS[@]}" \
    --print-task-ids
)"

echo "S08 ${S08_EXTENSION_SCOPE} x-axis extension report: ${REPORT_PATH}"
if [[ -z "$TASK_IDS" ]]; then
  echo "No missing final chunks for ${S08_EXTENSION_SCOPE} scope at ${S08_MAX_EVALUATION_PERCENTAGE}%."
  exit 0
fi

mkdir -p logs
export S08_MAX_EVALUATION_PERCENTAGE
export PYTHONUNBUFFERED=1

if [[ "$S08_EXTENSION_SCOPE" == "whole" ]]; then
  export S08_SWEEP_ROOT="$S08_EXTENSION_ROOT"
  export S08_SWEEP_TRAINING_PERCENTAGES="$S08_EXTENSION_TRAINING_PERCENTAGES"
  SLURM_SCRIPT="lgnStudies/bitFlipTrainingStudy/evaluate_s08_percentage_sweep.slurm"
else
  export S08_SCOPE_ROOT="$S08_EXTENSION_ROOT"
  export S08_SCOPE_TRAINING_PERCENTAGES="$S08_EXTENSION_TRAINING_PERCENTAGES"
  SLURM_SCRIPT="lgnStudies/bitFlipTrainingStudy/evaluate_s08_fault_scope_sweep.slurm"
fi

EVAL_JOB_ID="$(
  sbatch \
    --parsable \
    --array="$TASK_IDS" \
    --export=ALL \
    "$SLURM_SCRIPT"
)"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"

echo "Submitted S08 ${S08_EXTENSION_SCOPE} x-axis extension evaluation job ${EVAL_JOB_ID}"
echo "Only missing final chunk task IDs: ${TASK_IDS}"
echo "Root: ${S08_EXTENSION_ROOT}"
echo "Training percentages: ${S08_EXTENSION_TRAINING_PERCENTAGES}"
echo "Skip baseline: ${S08_EXTENSION_SKIP_BASELINE}"
echo "Maximum evaluation percentage: ${S08_MAX_EVALUATION_PERCENTAGE}"
