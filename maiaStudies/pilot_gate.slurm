#!/bin/bash
#SBATCH --job-name=maiaPilotGate
#SBATCH --output=logs/maiaPilotGate_%j.out
#SBATCH --error=logs/maiaPilotGate_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

set -euo pipefail

: "${MAIA_RESULT_ROOT:?MAIA_RESULT_ROOT is required}"
python -m maiaStudies.pilot_gate --result-root "$MAIA_RESULT_ROOT"
