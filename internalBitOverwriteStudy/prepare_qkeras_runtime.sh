#!/bin/bash
set -euo pipefail

if [[ ! -f internalBitOverwriteStudy/prepare_qkeras_runtime.sh ]]; then
  echo "Run this command from the SmartPixels repository root." >&2
  exit 2
fi

TARGET="${SMARTPIXELS_QKERAS_RUNTIME:-$PWD/.runtime/google-qkeras-0.9.0}"
PARENT="$(dirname "$TARGET")"
case "$TARGET" in
  ""|/|"$HOME"|"$PWD")
    echo "Refusing unsafe QKeras runtime target: $TARGET" >&2
    exit 2
    ;;
esac
mkdir -p "$PARENT"

if [[ ! -f "$TARGET/qkeras/qlayers.py" ]]; then
  if [[ -e "$TARGET" ]]; then
    echo "Existing runtime target is incomplete: $TARGET" >&2
    echo "Move that exact directory aside and rerun this script." >&2
    exit 2
  fi
  TEMPORARY="${TARGET}.installing.$$"
  trap 'rm -rf "$TEMPORARY"' EXIT
  rm -rf "$TEMPORARY"
  python -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "$TEMPORARY" \
    'QKeras==0.9.0'
  test -f "$TEMPORARY/qkeras/qlayers.py"
  mv "$TEMPORARY" "$TARGET"
  trap - EXIT
fi

export SMARTPIXELS_QKERAS_RUNTIME="$TARGET"
export PYTHONPATH="${TARGET}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from pathlib import Path

import qkeras
from qkeras import QActivation, QBatchNormalization, QDense, quantized_bits

origin = Path(qkeras.__file__).resolve()
qlayers = origin.parent / "qlayers.py"
if getattr(qkeras, "__version__", None) != "0.9.0":
    raise SystemExit(f"Expected Google QKeras 0.9.0, found {qkeras.__version__!r}.")
if not qlayers.is_file():
    raise SystemExit(f"Expected the Google QKeras qlayers.py module, found {qlayers}.")
print(f"[QKERAS RUNTIME] version={qkeras.__version__} package={origin}")
print(
    "[QKERAS API] "
    f"{QActivation.__name__}, {QBatchNormalization.__name__}, "
    f"{QDense.__name__}, {quantized_bits.__name__}"
)
PY

python -m internalBitOverwriteStudy.campaign --qkeras-preflight

echo "Prepared isolated Google QKeras runtime: $TARGET"
