#!/bin/bash
set -euo pipefail

ARRAY_JOB_ID="$(sbatch --parsable lgnStudies/bitFlipStudy/run.slurm)"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"
echo "Submitted bit-flip array job ${ARRAY_JOB_ID}"

COMBINE_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    lgnStudies/bitFlipStudy/combine.slurm
)"
COMBINE_JOB_ID="${COMBINE_JOB_ID%%;*}"
echo "Submitted dependent plot job ${COMBINE_JOB_ID}"
