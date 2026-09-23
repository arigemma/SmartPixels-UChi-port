#!/bin/bash
#SBATCH --job-name=rtlHandoffPackage
#SBATCH --output=logs/rtlHandoffPackage_%j.out
#SBATCH --error=logs/rtlHandoffPackage_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

if [[ ! -f rtlHandoffStudy/package.py ]]; then
  echo "[ERROR] Run from the SmartPixels repository root." >&2
  exit 1
fi

ROOT="${RTL_HANDOFF_RESULT_ROOT:-$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k}"
exec python -m rtlHandoffStudy.package --result-root "$ROOT" --overwrite
