#!/bin/bash
#SBATCH --job-name=model2lgnEval
#SBATCH --output=logs/model2lgnEval_%j.out
#SBATCH --error=logs/model2lgnEval_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

RESULT_ROOTS=(
  "results/SLURM/2026_06_01_model2lgnFull"
  "results/SLURM/2026_06_01_model2lgnFull_expanded"
  "results/SLURM/2026_06_22_model2lgnFull_expanded"
)

OUTDIR="results/evaluation/2026_06_23_model2lgnFull_all_discrete"

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

echo "Evaluating ${#CANDIDATES[@]} LGN checkpoints in discrete mode."
printf '  %s\n' "${CANDIDATES[@]}"

python eval_acceptance.py \
  --data-config config/10783560/v3.yml \
  --device cuda \
  --batch-size 1024 \
  --candidate-forward discrete \
  --stream-metrics \
  --outdir "$OUTDIR" \
  --candidates "${CANDIDATES[@]}"
