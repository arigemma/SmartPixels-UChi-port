#!/bin/bash
#SBATCH --job-name=model3FrontierEval
#SBATCH --output=logs/model3FrontierEval_%j.out
#SBATCH --error=logs/model3FrontierEval_%j.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f eval_acceptance.py || ! -f lgnStudies/select_best_tau_checkpoints.py ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi

TAUS=(10 20 40)
SIZES=(
  s01_p003M s02_p007M s03_p012M s04_p018M s05_p026M
  s06_p039M s07_p069M s08_p110M s09_p213M s10_p385M
)
RESULT_ROOT="${MODEL3_FRONTIER_RESULT_ROOT:-results/SLURM/2026_08_24_model3lgnConv_frontier_extension}"
EVAL_ROOT="${MODEL3_FRONTIER_EVAL_ROOT:-results/evaluation/2026_08_24_model3Frontier_discrete_balacc}"
STAGE_DIR="${EVAL_ROOT}/transfer_ready"

mkdir -p logs "$EVAL_ROOT"

CANDIDATES=()
for SIZE in "${SIZES[@]}"; do
  for TAU in "${TAUS[@]}"; do
    CHECKPOINT="${RESULT_ROOT}/${SIZE}/tau_${TAU}/best_model.pth"
    if [[ ! -s "$CHECKPOINT" ]]; then
      echo "[ERROR] Missing or empty checkpoint: ${CHECKPOINT}" >&2
      exit 1
    fi
    CANDIDATES+=("$CHECKPOINT")
  done
done

if [[ "${#CANDIDATES[@]}" -ne 30 ]]; then
  echo "[ERROR] Expected 30 Model 3 frontier checkpoints, found ${#CANDIDATES[@]}." >&2
  exit 1
fi

echo "Evaluating all 30 Model 3 frontier checkpoints in discrete mode."
printf '  %s\n' "${CANDIDATES[@]}"

python eval_acceptance.py \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 256 \
  --candidate-forward discrete \
  --stream-metrics \
  --outdir "$EVAL_ROOT" \
  --candidates "${CANDIDATES[@]}"

SUMMARY="${EVAL_ROOT}/balanced_accuracy_summary.tsv"
if [[ ! -s "$SUMMARY" ]]; then
  echo "[ERROR] Evaluation did not produce ${SUMMARY}." >&2
  exit 1
fi

SELECTION_ARGS=()
for SIZE in "${SIZES[@]}"; do
  SELECTION_ARGS+=(--expected-size "$SIZE")
done
for TAU in "${TAUS[@]}"; do
  SELECTION_ARGS+=(--expected-tau "$TAU")
done

python lgnStudies/select_best_tau_checkpoints.py \
  --summary "$SUMMARY" \
  --repository-root "$PWD" \
  --stage-dir "$STAGE_DIR" \
  --best-table "${EVAL_ROOT}/best_by_size.tsv" \
  "${SELECTION_ARGS[@]}" \
  --overwrite-stage

echo "[DONE] All scores: ${SUMMARY}"
echo "[DONE] Best tau per size: ${EVAL_ROOT}/best_by_size.tsv"
echo "[DONE] Transfer-ready checkpoints: ${STAGE_DIR}"
