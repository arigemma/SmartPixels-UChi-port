# Actual LUT synthesis for training-stage checkpoints

The immediate resource plot made by `resource_precision_study.py` is an
analytical diagnostic. Publication LUT counts should come from the actual
Vivado flow in `synthesis-SmartPixels`.

Every recorded training stage now writes this transfer-ready layout:

```text
synthesis_export/
  model2/LGN/
    s09_p1460M_tau40_step000000/
      best_model.importable.pth
      model.json
      SHA256SUMS
    s09_p1460M_tau40_step000200/
      ...
```

`best_model.importable.pth` is a full `torch.nn.Sequential` module assembled
from the existing model in `models/classification.py`. It contains the actual
TorchLogix binarizer, every `LogicDense` layer, GroupSum, trained LUT weights,
thresholds, and fixed connection tensors. Do not replace it with a plain
`state_dict`: current TorchLogix connection indices are not registered buffers.

## 1. Copy checkpoints to the synthesis cluster

Create an isolated campaign root in `synthesis-SmartPixels`, then copy the
contents from all three training jobs:

```bash
cd ~/synthesis-SmartPixels
mkdir -p staged_models/model2/LGN

scp -r TRAIN_HOST:/path/to/results/SLURM/2026_07_03_model2_resource_precision_5h/s06_p760M_tau40/synthesis_export/model2/LGN/* staged_models/model2/LGN/
scp -r TRAIN_HOST:/path/to/results/SLURM/2026_07_03_model2_resource_precision_5h/s08_p1202M_tau40/synthesis_export/model2/LGN/* staged_models/model2/LGN/
scp -r TRAIN_HOST:/path/to/results/SLURM/2026_07_03_model2_resource_precision_5h/s09_p1460M_tau40/synthesis_export/model2/LGN/* staged_models/model2/LGN/
```

Optionally verify transfer integrity:

```bash
find staged_models/model2/LGN -name SHA256SUMS -execdir sha256sum -c SHA256SUMS \;
```

## 2. Convert and synthesize

Use isolated converted/result directories so this campaign does not mix with
existing model scans:

```bash
source scripts/env.sh
export SMARTPIXELS_MODELS_ROOT="$PWD/staged_models"
export SMARTPIXELS_CONVERTED_ROOT="$PWD/converted_stage_models"
export SMARTPIXELS_RESULTS_ROOT="$PWD/results_stage_models"

scripts/run_da4ml_safe.py --model-set model2 --flow lgn --overwrite
scripts/run_vivado_synth.py --model-set model2 --flow lgn
scripts/scan_results.py --format csv --output stage_synthesis_results.csv
```

The safe runner will use `best_model.importable.pth` directly. The fast Vivado
step reports actual post-synthesis LUT and FF counts. Full Vivado can be used
instead when final timing-optimized publication numbers are required.

## 3. Copy the small result table back

```bash
scp stage_synthesis_results.csv TRAIN_HOST:/path/to/model/result/root/
```

For each model, join the Vivado LUTs to its exact balanced accuracies:

```bash
python lgnStudies/import_synthesis_results.py \
  --training-csv results/SLURM/2026_07_03_model2_resource_precision_5h/s09_p1460M_tau40/resource_precision.csv \
  --synthesis-csv results/SLURM/2026_07_03_model2_resource_precision_5h/stage_synthesis_results.csv \
  --model-alias s09_p1460M_tau40 \
  --output-dir results/SLURM/2026_07_03_model2_resource_precision_5h/s09_p1460M_tau40
```

This produces `vivado_luts_vs_balanced_accuracy.png` and `.pdf`. Use
`--allow-partial` when intentionally synthesizing only selected stages.
