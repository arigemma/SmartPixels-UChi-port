#!/bin/bash
#SBATCH --job-name=model3DeepBalAcc
#SBATCH --output=logs/model3DeepBalAcc_%j.out
#SBATCH --error=logs/model3DeepBalAcc_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

RESULT_ROOTS=(
  "results/SLURM/2026_06_27_model3lgnConv_deeper"
)

OUTDIR="results/evaluation/2026_06_28_model3Deep_discrete_balacc"

mkdir -p logs "$OUTDIR"

CANDIDATES=()
for ROOT in "${RESULT_ROOTS[@]}"; do
  if [[ ! -d "$ROOT" ]]; then
    echo "[WARN] Missing result root: $ROOT"
    continue
  fi

  while IFS= read -r -d '' CKPT; do
    CANDIDATES+=("$CKPT")
  done < <(find "$ROOT" -mindepth 3 -maxdepth 3 -path "*/tau_*/best_model.pth" -print0 | sort -z)
done

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "[ERROR] No tau_*/best_model.pth checkpoints found under configured result roots."
  exit 1
fi

if [[ "${#CANDIDATES[@]}" -ne 12 ]]; then
  echo "[WARN] Expected 12 model3Deep checkpoints, found ${#CANDIDATES[@]}."
fi

echo "Evaluating ${#CANDIDATES[@]} model3Deep checkpoints in discrete mode."
printf '  %s\n' "${CANDIDATES[@]}"

python eval_acceptance.py \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 256 \
  --candidate-forward discrete \
  --stream-metrics \
  --outdir "$OUTDIR" \
  --candidates "${CANDIDATES[@]}"
