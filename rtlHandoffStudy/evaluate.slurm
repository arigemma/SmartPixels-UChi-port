#!/bin/bash
#SBATCH --job-name=rtlHandoffEval
#SBATCH --output=logs/rtlHandoffEval_%A_%a.out
#SBATCH --error=logs/rtlHandoffEval_%A_%a.err
#SBATCH --time=03:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-19%10
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f rtlHandoffStudy/evaluate.py || ! -f config/10783560/v3.yml ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
ALIAS="$(python -c 'from models.rtl_handoff import RTL_HANDOFF_CONFIGS; import os; print(RTL_HANDOFF_CONFIGS[int(os.environ["SLURM_ARRAY_TASK_ID"])]["alias"])')"

if [[ -s "$ROOT/$ALIAS/test_metrics.json" ]]; then
  echo "[SKIP] Complete test evaluation: $ROOT/$ALIAS/test_metrics.json"
  exit 0
fi

exec python -m rtlHandoffStudy.evaluate \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --result-root "$ROOT" \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 256
