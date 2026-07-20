#!/bin/bash
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: bash lgnStudies/bitFlipTrainingStudy/submit_s08_custom_xaxis_points.sh SCOPE ROOT X_PERCENTAGES [--skip-baseline] TRAINING_PERCENTAGE [TRAINING_PERCENTAGE ...]" >&2
  echo "Example: bash lgnStudies/bitFlipTrainingStudy/submit_s08_custom_xaxis_points.sh whole results/SLURM/2026_07_17_s08_fault_aware_training_pct_sweep_10pct_5h_20k '10 12 18 22 26 30' --skip-baseline 5.1724137931 5.51724137931 5.86206896552 6.20689655172 6.55172413793 6.89655172414" >&2
  exit 2
fi

S08_CUSTOM_SCOPE="$1"
S08_CUSTOM_ROOT="$2"
S08_CUSTOM_X_PERCENTAGES="$3"
shift 3
S08_CUSTOM_SKIP_BASELINE=0
if [[ "${1:-}" == "--skip-baseline" ]]; then
  S08_CUSTOM_SKIP_BASELINE=1
  shift
fi

if (( $# < 1 )); then
  echo "At least one training percentage is required." >&2
  exit 2
fi
if [[ ! -d "$S08_CUSTOM_ROOT" ]]; then
  echo "Missing existing S08 sweep root: ${S08_CUSTOM_ROOT}" >&2
  exit 2
fi
case "$S08_CUSTOM_SCOPE" in
  whole|input) ;;
  *)
    echo "SCOPE must be 'whole' or 'input', got '${S08_CUSTOM_SCOPE}'." >&2
    exit 2
    ;;
esac

S08_CUSTOM_TRAINING_PERCENTAGES="$(
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

S08_CUSTOM_CHUNK_OFFSET="${S08_CUSTOM_CHUNK_OFFSET:-7}"
REPORT_PATH="${S08_CUSTOM_ROOT}/s08_${S08_CUSTOM_SCOPE}_custom_xaxis_task_report.json"
SKIP_BASELINE_ARGS=()
if [[ "$S08_CUSTOM_SKIP_BASELINE" == "1" ]]; then
  SKIP_BASELINE_ARGS+=(--skip-baseline)
fi
TASK_IDS="$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_custom_xaxis_points \
    --scope "$S08_CUSTOM_SCOPE" \
    --root "$S08_CUSTOM_ROOT" \
    --training-percentages "$S08_CUSTOM_TRAINING_PERCENTAGES" \
    --x-percentages "$S08_CUSTOM_X_PERCENTAGES" \
    --chunk-offset "$S08_CUSTOM_CHUNK_OFFSET" \
    --write-report "$REPORT_PATH" \
    "${SKIP_BASELINE_ARGS[@]}" \
    --print-task-ids
)"

echo "S08 ${S08_CUSTOM_SCOPE} custom x-axis report: ${REPORT_PATH}"
if [[ -z "$TASK_IDS" ]]; then
  echo "No missing custom chunks for ${S08_CUSTOM_SCOPE} scope."
  exit 0
fi

mkdir -p logs
export S08_CUSTOM_SCOPE
export S08_CUSTOM_ROOT
export S08_CUSTOM_TRAINING_PERCENTAGES
export S08_CUSTOM_X_PERCENTAGES
export S08_CUSTOM_SKIP_BASELINE
export S08_CUSTOM_CHUNK_OFFSET
export PYTHONUNBUFFERED=1

EVAL_JOB_ID="$(
  sbatch \
    --parsable \
    --array="$TASK_IDS" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/evaluate_s08_custom_xaxis_points.slurm
)"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"

echo "Submitted S08 ${S08_CUSTOM_SCOPE} custom x-axis evaluation job ${EVAL_JOB_ID}"
echo "Missing custom task IDs: ${TASK_IDS}"
echo "Root: ${S08_CUSTOM_ROOT}"
echo "Training percentages: ${S08_CUSTOM_TRAINING_PERCENTAGES}"
echo "X-axis percentages: ${S08_CUSTOM_X_PERCENTAGES}"
echo "Skip baseline: ${S08_CUSTOM_SKIP_BASELINE}"
echo "Custom chunk offset: ${S08_CUSTOM_CHUNK_OFFSET}"
