#!/bin/bash
set -euo pipefail

ROOT="${1:-results/SLURM/2026_08_13_selected_input_bit_flip_1to10pct_eval60_linear}"

python -m inputBitFlipStudy.campaign \
    --combine-paper-refinement \
    --root "$ROOT"

echo "[DONE] Replotted the paper figures and wrote point-location JSON files under:"
echo "  $ROOT"
