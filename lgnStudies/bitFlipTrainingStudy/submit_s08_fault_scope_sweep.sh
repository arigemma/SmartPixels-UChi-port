#!/bin/bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash lgnStudies/bitFlipTrainingStudy/submit_s08_fault_scope_sweep.sh OUTPUT_ROOT TRAINING_PERCENTAGE [TRAINING_PERCENTAGE ...]" >&2
  echo "Example: bash lgnStudies/bitFlipTrainingStudy/submit_s08_fault_scope_sweep.sh results/SLURM/$(date +%Y_%m_%d)_s08_input_vs_whole_fault_scope_5h_20k 0.1 0.3 1.0 3.0 5.0" >&2
  echo "Override the reused whole-model root with S08_WHOLE_SCOPE_ROOT=/path/to/2026_07_14_s08_fault_aware_training_pct_sweep_5h_20k." >&2
  exit 2
fi

S08_SCOPE_ROOT="$1"
shift
S08_MAX_EVALUATION_PERCENTAGE="${S08_MAX_EVALUATION_PERCENTAGE:-5.3}"
export S08_MAX_EVALUATION_PERCENTAGE
S08_WHOLE_SCOPE_ROOT="${S08_WHOLE_SCOPE_ROOT:-results/SLURM/2026_07_14_s08_fault_aware_training_pct_sweep_5h_20k}"
S08_SCOPE_TRAINING_PERCENTAGES="$(
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

if [[ -e "$S08_SCOPE_ROOT" ]]; then
  echo "Refusing to reuse an existing sweep root: ${S08_SCOPE_ROOT}" >&2
  exit 2
fi
if [[ ! -d "$S08_WHOLE_SCOPE_ROOT" ]]; then
  echo "Missing reusable whole-model s08 sweep root: ${S08_WHOLE_SCOPE_ROOT}" >&2
  exit 2
fi

TRAINING_TASKS="$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
    --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
    --print-training-task-count
)"
EVALUATION_TASKS="$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
    --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
    --print-evaluation-task-count
)"
INPUT_EVALUATION_COUNTS="$(
  python -m lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep \
    --training-percentages "$S08_SCOPE_TRAINING_PERCENTAGES" \
    --print-input-evaluation-counts
)"

mkdir -p logs "$S08_SCOPE_ROOT"
GIT_COMMIT="$(git rev-parse HEAD)"
python - "$S08_SCOPE_ROOT/campaign_config.json" \
  "$S08_SCOPE_TRAINING_PERCENTAGES" "$INPUT_EVALUATION_COUNTS" \
  "$GIT_COMMIT" "$S08_WHOLE_SCOPE_ROOT" \
  "$S08_MAX_EVALUATION_PERCENTAGE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
percentages = [float(value) for value in sys.argv[2].split()]
counts = [int(value) for value in sys.argv[3].split()]
git_commit = sys.argv[4]
whole_scope_root = sys.argv[5]
max_evaluation_percentage = float(sys.argv[6])
path.write_text(
    json.dumps(
        {
            "study": "s08_input_vs_whole_fault_scope_training_percentage_sweep",
            "git_commit": git_commit,
            "model": "s08_p1202M_tau40",
            "training_fault_percentages": percentages,
            "minimum_training_hours": 5.0,
            "minimum_training_steps": 20_000,
            "new_jobs": "input-only fault scope training and evaluation",
            "reused_whole_model_scope_root": whole_scope_root,
            "input_scope_evaluation_flip_counts_per_event": counts,
            "maximum_evaluation_percentage": max_evaluation_percentage,
            "evaluation": (
                "Exact-count inference-time fault sweep. Percentages are "
                "computed relative to each scope's eligible fault-site count."
            ),
        },
        indent=2,
    )
    + "\n"
)
PY

export S08_SCOPE_ROOT
export S08_SCOPE_TRAINING_PERCENTAGES
export S08_WHOLE_SCOPE_ROOT
export PYTHONUNBUFFERED=1

TRAINING_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((TRAINING_TASKS - 1))" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/run_s08_fault_scope_sweep.slurm
)"
TRAINING_JOB_ID="${TRAINING_JOB_ID%%;*}"
echo "Submitted s08 input-scope fault-aware training job ${TRAINING_JOB_ID}"

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((EVALUATION_TASKS - 1))" \
    --dependency="afterok:${TRAINING_JOB_ID}" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/evaluate_s08_fault_scope_sweep.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted s08 input-scope evaluation job ${EVALUATION_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/combine_s08_fault_scope_sweep.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted s08 fault-scope combine/plot job ${COMBINE_JOB_ID}"
echo "New sweep root: ${S08_SCOPE_ROOT}"
echo "Reused whole-model root: ${S08_WHOLE_SCOPE_ROOT}"
echo "Training percentages: ${S08_SCOPE_TRAINING_PERCENTAGES}"
echo "Input-scope evaluation counts: ${INPUT_EVALUATION_COUNTS}"
