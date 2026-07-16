# QKeras Activation-Bit Fault-Scope Studies

This folder is the QKeras counterpart to the S08 input-vs-whole LGN
fault-scope sweep in `lgnStudies/bitFlipTrainingStudy`. It is intentionally
self-contained so the `wip-mila-lgn-analysis` branch can run QKeras robustness
campaigns without rewriting the existing PyTorch/LGN model registry.

The main campaign trains and evaluates three y-profile QKeras models copied
from `wip-mila-qkeras`:

- `qkeras_y192_w5a10`: `Y192`, `W5/A10`
- `qkeras_y256_64_w4a8`: `Y256/64`, `W4/A8`
- `qkeras_y768_256_128_w5a10`: `Y768/256/128`, `W5/A10`

All three use the same 14-feature `y-profile` input and three-class
classification target as the LGN model-2 studies.

## Launch

Run from the SmartPixels repository root on the `wip-mila-lgn-analysis`
branch:

```bash
bash qkerasStudies/bitFlipTrainingStudy/submit_fault_scope_sweep.sh \
  results/SLURM/$(date +%Y_%m_%d)_qkeras_input_vs_whole_fault_scope_3h_6k \
  0.1 0.3 1.0 3.0 5.0
```

This is the one-command launcher. It submits:

1. clean baseline training once per model;
2. fault-aware training for both fault scopes and every requested training
   percentage;
3. exact-count inference fault evaluation for all clean and fault-aware
   checkpoints;
4. one dependent combine/plot job.

With the five percentages above, the launcher creates 33 training tasks and
132 evaluation tasks. Training jobs request one GPU for `3:59:59` and require
both at least 6,000 optimizer steps and at least 3.0 training-loop hours
before publishing `final_model.keras`.

The QKeras environment must provide TensorFlow/Keras plus QKeras. The local
dependency hint is:

```bash
pip install -r qkerasStudies/requirements.txt
```

On clusters, use the site module or environment that already supports the
`wip-mila-qkeras` branch if that is preferred.

## Fault Model

The LGN studies flip Boolean activation sites. QKeras activations are
multi-bit quantized values, so this study defines one fault site as one
activation bit.

Two fault scopes are evaluated:

- `input`: the 14 y-profile input features quantized to the model activation
  width using a range calibrated from the training split;
- `all`: the input activation bits plus every hidden `QActivation` output bit.

Logits, weights, biases, and batch-normalization state are never faulted. For
example, `qkeras_y192_w5a10` has `(14 + 192) * 10 = 2060` whole-model
activation-bit sites and `14 * 10 = 140` input-only sites.

Training faults use an independent Bernoulli probability per eligible bit on
every forward pass. A launch percentage `q` means `p = q / 100`. The operation
quantizes the activation, flips selected bits, dequantizes, and uses a
straight-through estimator so gradients flow through the faulted activation.

Inference faults use exactly `K` distinct activation-bit sites per event,
sampled uniformly without replacement. This mirrors the LGN exact-count
evaluation and keeps the x-axis controlled. The plotted percentage is always:

```text
100 * K / total_fault_sites_per_event
```

The evaluation K grid maps the LGN S08 percentage grid onto each QKeras
model/scope and ends at 5.3%. Small input-only fault spaces naturally have
fewer unique K values after rounding.

## Output Layout

A campaign root contains:

```text
campaign_config.json
clean_training/
input_scope/
whole_scope/
qkeras_fault_scope_sweep_summary.json
qkeras_y192_w5a10_*_training_percentage_sweep_accuracy*.png/pdf
qkeras_y256_64_w4a8_*_training_percentage_sweep_accuracy*.png/pdf
qkeras_y768_256_128_w5a10_*_training_percentage_sweep_accuracy*.png/pdf
```

Clean baseline checkpoints live under:

```text
clean_training/MODEL/training/
```

Fault-aware training checkpoints live under:

```text
input_scope/train_1pct/MODEL/training/
whole_scope/train_1pct/MODEL/training/
```

Each training directory contains:

- `final_model.keras`
- `model_config.json`
- `training_complete.json`
- `fault_aware_training_history.csv`
- `clean_accuracy_during_fault_aware_training.png`
- `clean_accuracy_during_fault_aware_training.pdf`
- `latest_model.keras` at the latest recorded stage

Chunked exact-count evaluations remain auditable under:

```text
input_scope/TAG/MODEL/chunks/chunk_00/
whole_scope/TAG/MODEL/chunks/chunk_00/
```

After the combine job, each `TAG/MODEL` directory also contains:

- `bit_flip_results.csv`
- `bit_flip_metadata.json`
- `clean_predictions.npy`

The root-level plots are produced per model:

- `MODEL_input_scope_training_percentage_sweep_accuracy.png/pdf`
- `MODEL_input_scope_training_percentage_sweep_accuracy_3d.png/pdf`
- `MODEL_input_scope_training_percentage_sweep_accuracy_fit.png/pdf`
- `MODEL_input_scope_training_percentage_sweep_accuracy_fit_3d.png/pdf`
- `MODEL_whole_scope_training_percentage_sweep_accuracy...`
- `MODEL_fault_scope_training_percentage_sweep_accuracy...`

These correspond to the LGN S08 plot families, with the model alias prefixed
because this campaign runs three QKeras architectures at once.

## Useful Dry Runs

Before submitting, inspect the task map:

```bash
python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
  --training-percentages "0.1 0.3 1 3 5" \
  --print-training-task-count

python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
  --training-percentages "0.1 0.3 1 3 5" \
  --print-evaluation-task-count

python -m qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep \
  --training-percentages "0.1 0.3 1 3 5" \
  --print-model-counts
```

The expected counts for the standard five-percentage launch are:

```text
training tasks: 33
evaluation tasks: 132
```

## Interpretation

Use the `inference_only` rows as the answer to "what happens when a clean
QKeras model only sees faults at inference time?" Use the `train_*pct` rows as
the answer to "what happens when the model is trained with this fault rate in
the same fault scope?"

The clean `K=0` points show whether fault-aware training changed nominal
classification. The nonzero K curves show robustness. The input-vs-whole
plots separate sensitivity to corrupted detector/input representation from
sensitivity to internal activation-bit corruption.
