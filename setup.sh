#!/usr/bin/env bash
set -euo pipefail

module load anaconda3/2025.12
module load cudatoolkit/13.0
conda activate smart_pixels

python - <<'PY'
import numpy, pyarrow, sklearn, torch
import torchlogix
print("MAIA SmartPixels environment is ready.")
print("PyTorch:", torch.__version__)
print("TorchLogix:", torchlogix.__file__)
PY
