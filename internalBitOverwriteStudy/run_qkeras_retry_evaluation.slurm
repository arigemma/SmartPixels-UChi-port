#!/bin/bash
#SBATCH --job-name=internalOverwriteQRetry
#SBATCH --output=logs/internalOverwriteQRetry_%A_%a.out
#SBATCH --error=logs/internalOverwriteQRetry_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-23%12
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartpixels-matplotlib}"
: "${INTERNAL_OVERWRITE_ROOT:?Missing campaign output root}"
source internalBitOverwriteStudy/qkeras_runtime.sh

IFS=$'\t' read -r FAMILY TAG TRAINING_PERCENTAGE MODEL REGISTRY_NAME COUNT MINIMUM_HOURS MINIMUM_STEPS <<< "$(
  python -m internalBitOverwriteStudy.campaign \
    --root "$INTERNAL_OVERWRITE_ROOT" \
    --qkeras-retry-task-id "$SLURM_ARRAY_TASK_ID"
)"
if [[ "$FAMILY" != "qkeras" ]]; then
  echo "Retry manifest contains a non-QKeras task: $FAMILY" >&2
  exit 2
fi

TRAINING_DIR="${INTERNAL_OVERWRITE_ROOT}/${FAMILY}/${TAG}/${MODEL}/training"
OUTPUT_DIR="${INTERNAL_OVERWRITE_ROOT}/${FAMILY}/${TAG}/${MODEL}/chunks/chunk_retry_${SLURM_ARRAY_TASK_ID}_count_${COUNT}"
CHECKPOINT="${TRAINING_DIR}/final_model.keras"
STATUS="${TRAINING_DIR}/training_complete.json"
mkdir -p logs "$OUTPUT_DIR"

echo "Retrying ${FAMILY}/${MODEL}, trained at ${TRAINING_PERCENTAGE}%"
echo "One exact selected internal-site count: ${COUNT}"

python -m qkerasStudies.training_completion \
  --checkpoint "$CHECKPOINT" --status "$STATUS" \
  --minimum-hours "$MINIMUM_HOURS" --minimum-steps "$MINIMUM_STEPS"
python -m qkerasStudies.bitFlipStudy.study \
  -c config/10783560/v3.yml --device cuda -bs 1024 \
  --flip-counts "$COUNT" --repeats 5 --seed 42 --no-plots \
  --fault-scope internal --fault-operation random_set \
  --model-alias "$MODEL" --registry-name "$REGISTRY_NAME" \
  --clean-model "$CHECKPOINT" --training-status "$STATUS" \
  -o "$OUTPUT_DIR"
