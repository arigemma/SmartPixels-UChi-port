#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RESULTS_ROOT" >&2
  exit 2
fi

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"
RESULT_ROOT="$(realpath "$1")"
if [[ ! -f "$RESULT_ROOT/campaign_manifest.json" ]]; then
  echo "[ERROR] Not an HGQ2 campaign root: $RESULT_ROOT" >&2
  exit 1
fi

TASK_SPEC="${HGQ_CALIBRATION_TASKS:-0-20%4}"
CALIBRATION_JOB="$({
  sbatch \
    --parsable \
    --chdir="$REPOSITORY_ROOT" \
    --array="$TASK_SPEC" \
    --export="ALL,HGQ_RESULT_ROOT=${RESULT_ROOT}" \
    "$REPOSITORY_ROOT/hgqStudies/calibrate.slurm"
})"
CALIBRATION_JOB="${CALIBRATION_JOB%%;*}"

echo "Submitted HGQ2 calibration: ${CALIBRATION_JOB} tasks=${TASK_SPEC}"
echo "Campaign root: ${RESULT_ROOT}"

if [[ -z "${HGQ_CALIBRATION_TASKS:-}" ]]; then
  SUMMARY_JOB="$({
    sbatch \
      --parsable \
      --chdir="$REPOSITORY_ROOT" \
      --dependency="afterany:${CALIBRATION_JOB}" \
      --export="ALL,HGQ_RESULT_ROOT=${RESULT_ROOT}" \
      "$REPOSITORY_ROOT/hgqStudies/summarize_calibrated.slurm"
  })"
  SUMMARY_JOB="${SUMMARY_JOB%%;*}"
  echo "Submitted strict calibrated summary: ${SUMMARY_JOB}"
fi
