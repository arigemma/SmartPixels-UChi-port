#!/bin/bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash qkerasStudies/bitFlipTrainingStudy/submit_fault_scope_xaxis_extension.sh ROOT MAX_EVALUATION_PERCENT TRAINING_PERCENTAGE [TRAINING_PERCENTAGE ...]" >&2
  echo "Example: QKERAS_SCOPE_MODELS=qkeras_y256_64_w4a8 bash qkerasStudies/bitFlipTrainingStudy/submit_fault_scope_xaxis_extension.sh results/SLURM/2026_07_16_qkeras_y256_64_input_vs_whole_fault_scope_3h_6k 7 3 5" >&2
  exit 2
fi

QKERAS_SCOPE_ROOT="$1"
S08_MAX_EVALUATION_PERCENTAGE="$2"
shift 2
QKERAS_SCOPE_MODELS="${QKERAS_SCOPE_MODELS:-all}"
QKERAS_FINAL_OUTPUT_DIR="${QKERAS_FINAL_OUTPUT_DIR:-temp}"
QKERAS_SCOPE_TRAINING_PERCENTAGES="$(
  python - "$@" <<'PY'
import math
import sys

values = []
for raw in sys.argv[1:]:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise SystemExit("Training percentages must lie in (0, 100].")
    values.append(value)
if len(set(values)) != len(values):
    raise SystemExit("Duplicate training percentages are not allowed.")
print(" ".join(f"{value:.12g}" for value in sorted(values)))
PY
)"
TASK_IDS="$(
  S08_MAX_EVALUATION_PERCENTAGE="$S08_MAX_EVALUATION_PERCENTAGE" \
    python - "$QKERAS_SCOPE_MODELS" "$QKERAS_SCOPE_TRAINING_PERCENTAGES" <<'PY'
import sys

from qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep import (
    evaluation_flip_count_chunks,
    evaluation_tasks,
    parse_model_selection,
    parse_training_percentages,
)

models = sys.argv[1]
percentages = parse_training_percentages(sys.argv[2])
configs = parse_model_selection(models)
configs_by_alias = {config.alias: config for config in configs}
task_ids = []
for task_id, task in enumerate(evaluation_tasks(percentages, configs)):
    config = configs_by_alias[task["model_alias"]]
    last_chunk = len(evaluation_flip_count_chunks(config, task["fault_scope"])) - 1
    if task["chunk_index"] == last_chunk:
        task_ids.append(str(task_id))
print(",".join(task_ids))
PY
)"

if [[ -z "$TASK_IDS" ]]; then
  echo "No final evaluation chunks selected." >&2
  exit 2
fi

export QKERAS_SCOPE_ROOT
export QKERAS_SCOPE_MODELS
export QKERAS_SCOPE_TRAINING_PERCENTAGES
export QKERAS_FINAL_OUTPUT_DIR
export S08_MAX_EVALUATION_PERCENTAGE
export PYTHONUNBUFFERED=1

mkdir -p logs "$QKERAS_FINAL_OUTPUT_DIR"

EVAL_JOB_ID="$(
  sbatch \
    --parsable \
    --array="$TASK_IDS" \
    --export=ALL \
    qkerasStudies/bitFlipTrainingStudy/evaluate_fault_scope_sweep.slurm
)"
EVAL_JOB_ID="${EVAL_JOB_ID%%;*}"
echo "Submitted QKeras x-axis extension evaluation job ${EVAL_JOB_ID}"
echo "Only final chunk task IDs: ${TASK_IDS}"

PLOT_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVAL_JOB_ID}" \
    --export=ALL \
    qkerasStudies/bitFlipTrainingStudy/plot_fault_scope_xaxis_extension.slurm
)"
PLOT_JOB_ID="${PLOT_JOB_ID%%;*}"
echo "Submitted QKeras final plot job ${PLOT_JOB_ID}"
echo "Plots will be written under: ${QKERAS_FINAL_OUTPUT_DIR}"
