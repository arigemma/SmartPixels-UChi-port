#!/bin/bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash lgnStudies/bitFlipTrainingStudy/submit.sh TRAINING_FLIP_PERCENTAGE [OUTPUT_ROOT]" >&2
  echo "Example: bash lgnStudies/bitFlipTrainingStudy/submit.sh 1.0" >&2
  exit 2
fi

BIT_FLIP_TRAINING_PERCENTAGE="$(
  python -c '
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("Training flip percentage must be numeric.")
if not math.isfinite(value) or not 0.0 < value <= 100.0:
    raise SystemExit("Training flip percentage must lie in (0, 100].")
print(f"{value:.12g}")
' "$1"
)"

PERCENTAGE_TAG="$(
  python -c '
import sys
print(sys.argv[1].replace(".", "p"))
' "$BIT_FLIP_TRAINING_PERCENTAGE"
)"
DEFAULT_ROOT="results/SLURM/$(date +%Y_%m_%d)_model2_fault_aware_training_${PERCENTAGE_TAG}pct_5h_20k"
BIT_FLIP_TRAINING_ROOT="${2:-$DEFAULT_ROOT}"

if [[ -e "$BIT_FLIP_TRAINING_ROOT" ]]; then
  echo "Refusing to reuse an existing campaign root: ${BIT_FLIP_TRAINING_ROOT}" >&2
  echo "Pass a new OUTPUT_ROOT as the second argument." >&2
  exit 2
fi

mkdir -p logs "$BIT_FLIP_TRAINING_ROOT"
GIT_COMMIT="$(git rev-parse HEAD)"
python - "$BIT_FLIP_TRAINING_ROOT/campaign_config.json" \
  "$BIT_FLIP_TRAINING_PERCENTAGE" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
percentage = float(sys.argv[2])
git_commit = sys.argv[3]
path.write_text(
    json.dumps(
        {
            "study": "fault_aware_activation_bit_flip_training",
            "git_commit": git_commit,
            "training_fault_percentage": percentage,
            "training_fault_probability": percentage / 100.0,
            "minimum_training_hours": 5.0,
            "minimum_training_steps": 20_000,
            "models": [
                "s06_p760M_tau40",
                "s08_p1202M_tau40",
                "s09_p1460M_tau40",
            ],
            "evaluation": (
                "Existing exact-count inference-time fault sweep with five "
                "complete-test realizations per nonzero K."
            ),
        },
        indent=2,
    )
    + "\n"
)
PY

export BIT_FLIP_TRAINING_PERCENTAGE
export BIT_FLIP_TRAINING_ROOT
export PYTHONUNBUFFERED=1

TRAINING_JOB_ID="$(
  sbatch \
    --parsable \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/run.slurm
)"
TRAINING_JOB_ID="${TRAINING_JOB_ID%%;*}"
echo "Submitted fault-aware training array job ${TRAINING_JOB_ID}"

EVALUATION_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${TRAINING_JOB_ID}" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/evaluate.slurm
)"
EVALUATION_JOB_ID="${EVALUATION_JOB_ID%%;*}"
echo "Submitted dependent inference robustness array job ${EVALUATION_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${EVALUATION_JOB_ID}" \
    --export=ALL \
    lgnStudies/bitFlipTrainingStudy/combine.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent validation/plot job ${COMBINE_JOB_ID}"
echo "Campaign root: ${BIT_FLIP_TRAINING_ROOT}"
