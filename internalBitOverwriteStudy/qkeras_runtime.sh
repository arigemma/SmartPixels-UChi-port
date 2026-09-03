#!/bin/bash

# Source this file. Do not execute it: it exports the isolated Google QKeras
# package path for the current process and for subsequently submitted jobs.
SMARTPIXELS_QKERAS_RUNTIME="${SMARTPIXELS_QKERAS_RUNTIME:-$PWD/.runtime/google-qkeras-0.9.0}"
if [[ ! -f "${SMARTPIXELS_QKERAS_RUNTIME}/qkeras/qlayers.py" ]]; then
  echo "Missing isolated Google QKeras runtime at ${SMARTPIXELS_QKERAS_RUNTIME}." >&2
  echo "Run: bash internalBitOverwriteStudy/prepare_qkeras_runtime.sh" >&2
  return 2 2>/dev/null || exit 2
fi

export SMARTPIXELS_QKERAS_RUNTIME
export PYTHONPATH="${SMARTPIXELS_QKERAS_RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
