#!/bin/bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash qkerasStudies/bitFlipTrainingStudy/submit_fault_scope_sweep.sh OUTPUT_ROOT TRAINING_PERCENTAGE [TRAINING_PERCENTAGE ...]" >&2
  echo "Example: bash qkerasStudies/bitFlipTrainingStudy/submit_fault_scope_sweep.sh results/SLURM/\$(date +%Y_%m_%d)_qkeras_input_vs_whole_fault_scope_3h_6k 0.1 0.3 1.0 3.0 5.0" >&2
  exit 2
fi

QKERAS_SCOPE_ROOT="$1"
shift
QKERAS_SCOPE_TRAINING_PERCENTAGES="$(
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

if [[ -e "$QKERAS_SCOPE_ROOT" ]]; then
  echo "Refusing to reuse an existing sweep root: ${QKERAS_SCOPE_ROOT}" >&2
  exit 2
fi

TRAINING_TASKS="$(
  python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
    --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
    --print-training-task-count
)"
EVALUATION_TASKS="$(
  python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
    --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
    --print-evaluation-task-count
)"
MODEL_COUNTS="$(
  python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
    --training-percentages "$QKERAS_SCOPE_TRAINING_PERCENTAGES" \
    --print-model-counts
)"

mkdir -p logs "$QKERAS_SCOPE_ROOT"
GIT_COMMIT="$(git rev-parse HEAD)"
python - "$QKERAS_SCOPE_ROOT/campaign_config.json" \
  "$QKERAS_SCOPE_TRAINING_PERCENTAGES" "$MODEL_COUNTS" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
percentages = [float(value) for value in sys.argv[2].split()]
counts_text = sys.argv[3]
git_commit = sys.argv[4]
models = {}
for line in counts_text.splitlines():
    alias, scope_key, total_sites, counts = line.split("\t")
    model = models.setdefault(alias, {"model_alias": alias, "scopes": {}})
    model["scopes"][scope_key] = {
        "total_fault_sites_per_event": int(total_sites),
        "evaluation_flip_counts": [int(value) for value in counts.split()],
    }
path.write_text(
    json.dumps(
        {
            "study": "qkeras_input_vs_whole_activation_bit_fault_scope_training_percentage_sweep",
            "git_commit": git_commit,
            "models": list(models.values()),
            "training_fault_percentages": percentages,
            "minimum_training_hours": 3.0,
            "minimum_training_steps": 6000,
            "training_jobs": (
                "Clean baseline once per model plus fault-aware training for "
                "input-only and whole-model activation-bit scopes."
            ),
            "evaluation": (
                "Exact-count inference-time activation-bit fault sweep. "
                "Percentages are computed relative to each model/scope eligible "
                "activation-bit count."
            ),
            "maximum_evaluation_percentage": 5.3,
            "input_scope": (
                "The 14 y-profile input features quantized to the model activation "
                "width using the training split calibration range."
            ),
            "whole_scope": (
                "Input activation bits plus every hidden QActivation output bit; "
                "logits, weights, biases, and batch-normalization state are not faulted."
            ),
        },
        indent=2,
    )
    + "\n"
)
PY

export QKERAS_SCOPE_ROOT
export QKERAS_SCOPE_TRAINING_PERCENTAGES
export PYTHONUNBUFFERED=1

TRAINING_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((TRAINING_TASKS - 1))" \
    --export=ALL \
    qkerasStudies/bitFlipTrainingStudy/run_fault_scope_sweep.slurm
)"
TRAINING_JOB_ID="${TRAINING_JOB_ID%%;*}"
echo "Submitted QKeras training job ${TRAINING_JOB_ID}"

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --array="0-$((EVALUATION_TASKS - 1))" \
    --dependency="afterok:${TRAINING_JOB_ID}" \
    --export=ALL \
    qkerasStudies/bitFlipTrainingStudy/evaluate_fault_scope_sweep.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted QKeras evaluation job ${EVALUATION_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    qkerasStudies/bitFlipTrainingStudy/combine_fault_scope_sweep.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted QKeras combine/plot job ${COMBINE_JOB_ID}"
echo "New sweep root: ${QKERAS_SCOPE_ROOT}"
echo "Training percentages: ${QKERAS_SCOPE_TRAINING_PERCENTAGES}"
echo "Training tasks: ${TRAINING_TASKS}"
echo "Evaluation tasks: ${EVALUATION_TASKS}"
echo "Model/scope evaluation grids:"
echo "${MODEL_COUNTS}"
