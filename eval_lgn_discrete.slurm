#!/bin/bash
#SBATCH --job-name=model3FullBalAcc
#SBATCH --output=logs/model3FullBalAcc_%j.out
#SBATCH --error=logs/model3FullBalAcc_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

# Submit this file from the SmartPixels repository root. Slurm evaluates the
# paths below relative to the submission directory.
if [[ ! -f "eval_acceptance.py" || ! -f "config/10783560/v3.yml" ]]; then
  echo "[ERROR] Submit eval_lgn_discrete.slurm from the SmartPixels repository root." >&2
  exit 1
fi

RESULT_ROOT="${MODEL3_FULL_RESULT_ROOT:-results/SLURM/2026_06_25_model3lgnConv_expanded}"
OUTDIR="${MODEL3_FULL_EVAL_OUTDIR:-results/evaluation/2026_08_13_model3Full_discrete_balacc}"
STAGE_DIR="$OUTDIR/transfer_ready"
EXPECTED_SIZES=(
  "s01_p319M"
  "s02_p465M"
  "s03_p639M"
  "s04_p840M"
  "s05_p1069M"
)
EXPECTED_TAUS=(20 40 60)

mkdir -p logs "$OUTDIR"

CANDIDATES=()
for SIZE in "${EXPECTED_SIZES[@]}"; do
  for TAU in "${EXPECTED_TAUS[@]}"; do
    CKPT="$RESULT_ROOT/$SIZE/tau_$TAU/best_model.pth"
    if [[ ! -f "$CKPT" ]]; then
      echo "[ERROR] Missing expected checkpoint: $CKPT" >&2
      exit 1
    fi
    CANDIDATES+=("$CKPT")
  done
done

if [[ "${#CANDIDATES[@]}" -ne 15 ]]; then
  echo "[ERROR] Expected 15 model3Full checkpoints, found ${#CANDIDATES[@]}." >&2
  exit 1
fi

echo "Evaluating ${#CANDIDATES[@]} model3Full checkpoints in discrete mode."
printf '  %s\n' "${CANDIDATES[@]}"

python eval_acceptance.py \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 256 \
  --candidate-forward discrete \
  --stream-metrics \
  --outdir "$OUTDIR" \
  --candidates "${CANDIDATES[@]}"

SUMMARY="$OUTDIR/balanced_accuracy_summary.tsv"
if [[ ! -s "$SUMMARY" ]]; then
  echo "[ERROR] Evaluation did not produce $SUMMARY" >&2
  exit 1
fi

python lgnStudies/select_best_tau_checkpoints.py \
  --summary "$SUMMARY" \
  --repository-root "$PWD" \
  --stage-dir "$STAGE_DIR" \
  --best-table "$OUTDIR/best_by_size.tsv" \
  --expected-size "${EXPECTED_SIZES[0]}" \
  --expected-size "${EXPECTED_SIZES[1]}" \
  --expected-size "${EXPECTED_SIZES[2]}" \
  --expected-size "${EXPECTED_SIZES[3]}" \
  --expected-size "${EXPECTED_SIZES[4]}" \
  --expected-tau "${EXPECTED_TAUS[0]}" \
  --expected-tau "${EXPECTED_TAUS[1]}" \
  --expected-tau "${EXPECTED_TAUS[2]}" \
  --overwrite-stage

echo "[DONE] Five best-tau checkpoints are ready for SCP from:"
echo "  $STAGE_DIR"
