#!/bin/bash
#SBATCH --job-name=hgqM2Summary
#SBATCH --output=logs/hgqM2Summary_%j.out
#SBATCH --error=logs/hgqM2Summary_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ -z "${HGQ_RESULT_ROOT:-}" ]]; then
  echo "[ERROR] HGQ_RESULT_ROOT was not exported by hgqStudies/submit.sh." >&2
  exit 1
fi

python -m hgqStudies.summarize \
  --result-root "$HGQ_RESULT_ROOT" \
  --output-dir "$HGQ_RESULT_ROOT/summary"
