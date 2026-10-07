#!/bin/bash
#SBATCH --job-name=maiaPrepare
#SBATCH --output=logs/maiaPrepare_%j.out
#SBATCH --error=logs/maiaPrepare_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

[[ -f maiaStudies/prepare.py ]] || { echo "Run from the SmartPixels root." >&2; exit 2; }
: "${MAIA_DATA_ROOT:?MAIA_DATA_ROOT is required}"
: "${MAIA_CACHE_ROOT:?MAIA_CACHE_ROOT is required}"

mkdir -p logs "$MAIA_CACHE_ROOT"
python -m maiaStudies.prepare \
  --data-root "$MAIA_DATA_ROOT" \
  --cache-dir "$MAIA_CACHE_ROOT" \
  --timing-column "${MAIA_TIMING_COLUMN:-adjusted_hit_time}" \
  --batch-size 4096 \
  --thresholds 100 \
  --threshold-sample-rows 50000 \
  --seed 42
