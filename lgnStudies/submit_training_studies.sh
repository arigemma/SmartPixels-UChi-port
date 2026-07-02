#!/bin/bash
set -euo pipefail

RESOURCE_JOB_ID="$(sbatch --parsable lgnStudies/run.slurm)"
RESOURCE_JOB_ID="${RESOURCE_JOB_ID%%;*}"
echo "Submitted resource/accuracy array job ${RESOURCE_JOB_ID}"

bash lgnStudies/bitFlipStudy/submit.sh "$RESOURCE_JOB_ID"
