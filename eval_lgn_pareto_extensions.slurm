#!/bin/bash
#SBATCH --job-name=paretoExtBalAcc
#SBATCH --output=logs/paretoExtBalAcc_%A_%a.out
#SBATCH --error=logs/paretoExtBalAcc_%A_%a.err
#SBATCH --time=5:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f "eval_acceptance.py" || ! -f "config/10783560/v3.yml" ]]; then
  echo "[ERROR] Submit from the SmartPixels repository root." >&2
  exit 1
fi

TAUS=(10 20 40)
EVAL_ROOT="${PARETO_EXTENSION_EVAL_ROOT:-results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc}"

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  FAMILY="dense"
  RESULT_ROOT="${PARETO_DENSE_RESULT_ROOT:-results/SLURM/2026_08_14_model2lgnFull_left_extension_fixed}"
  SIZES=(
    s01_p005M s02_p008M s03_p010M s04_p013M
    s05_p015M s06_p018M s07_p020M
  )
elif [[ "$SLURM_ARRAY_TASK_ID" -eq 1 ]]; then
  FAMILY="conv"
  RESULT_ROOT="${PARETO_CONV_RESULT_ROOT:-results/SLURM/2026_08_14_model3lgnConv_curve_extension_fixed}"
  SIZES=(
    s01_p046M s02_p074M s03_p109M s04_p152M s05_p201M
    s06_p1325M s07_p1608M s08_p1919M s09_p2257M s10_p2623M
  )
else
  echo "[ERROR] Array task must be 0 (dense) or 1 (conv)." >&2
  exit 2
fi

OUTDIR="${EVAL_ROOT}/${FAMILY}"
STAGE_DIR="${OUTDIR}/transfer_ready"
mkdir -p logs "$OUTDIR"

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

EXPECTED_COUNT=$(( ${#SIZES[@]} * ${#TAUS[@]} ))
if [[ "${#CANDIDATES[@]}" -ne "$EXPECTED_COUNT" ]]; then
  echo "[ERROR] Expected ${EXPECTED_COUNT} ${FAMILY} checkpoints, found ${#CANDIDATES[@]}." >&2
  exit 1
fi

echo "Evaluating ${#CANDIDATES[@]} ${FAMILY} Pareto-extension checkpoints in discrete mode."
printf '  %s\n' "${CANDIDATES[@]}"

python eval_acceptance.py \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 256 \
  --candidate-forward discrete \
  --stream-metrics \
  --outdir "$OUTDIR" \
  --candidates "${CANDIDATES[@]}"

SUMMARY="${OUTDIR}/balanced_accuracy_summary.tsv"
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
  --best-table "${OUTDIR}/best_by_size.tsv" \
  "${SELECTION_ARGS[@]}" \
  --overwrite-stage

echo "[DONE] ${FAMILY} scores: ${SUMMARY}"
echo "[DONE] ${FAMILY} best taus: ${OUTDIR}/best_by_size.tsv"
echo "[DONE] ${FAMILY} transfer-ready checkpoints: ${STAGE_DIR}"
