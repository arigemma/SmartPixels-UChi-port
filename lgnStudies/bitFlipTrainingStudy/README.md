# Fault-aware bit-flip training study

This study covers the second bit-flip scenario: transient activation faults
are injected **during training** so the optimizer can learn a model that is
robust to them. It is separate from
[`../bitFlipStudy`](../bitFlipStudy/README.md), which leaves an already-trained
model frozen and injects faults only during inference.

The campaign trains the same three tau-40, model-2 y-profile LGNs:

- `s06_p760M_tau40`
- `s08_p1202M_tau40`
- `s09_p1460M_tau40`

After all three fault-aware training jobs finish, the campaign automatically
evaluates every checkpoint with the existing exact-count inference fault
sweep. Consequently, its final per-model plots have exactly the same layout
as the inference-only study: balanced accuracy at the top, all three
true-class recalls in the middle, and changed/repaired/damaged predictions at
the bottom.

## Training fault definition

The launcher takes a percentage `q`; the corresponding per-site probability
is `p = q / 100`. On every training forward pass, every eligible activation is
selected independently with probability `p`. Selected values are complemented
with `x -> 1-x`.

The default eligible activation space is unchanged from the inference study:

1. every output of `FixedBinarization`;
2. every output of every `LogicDense` layer.

`GroupSum`, learned LUT parameters, and fixed connections are not faulted. The
training and inference scripts also accept `--fault-scope input`, which limits
the eligible space to the `FixedBinarization` output only. Masks are newly
sampled on every forward pass and every event. If a scope has `N` eligible
sites, the number of training flips per event follows
`Binomial(N, p)`, with:

```text
expected flips = N p
standard deviation = sqrt(N p (1-p))
```

This is intentionally different from the inference evaluation. Training uses
an independent probability because this is a stochastic regularization
strategy. Evaluation uses exactly `K` distinct faults per event so robustness
curves have a controlled and directly interpretable x-axis. The evaluation
also provides the equivalent percentage `100 K / N`.

`FixedBinarization` produces 0/1 values, while TorchLogix `LogicDense` outputs
are relaxed during training and discrete during evaluation. The
differentiable training analogue of a Boolean bit flip is therefore the
out-of-place operation `x -> 1-x`. Its derivative is `-1` at a selected site,
allowing gradients to continue through every faulted logic layer. Clean stage
evaluations disable all faults and use the discrete model.

## Launch

Run from the SmartPixels repository root. The percentage is mandatory so a
scientific operating point is never selected implicitly:

```bash
git pull
bash lgnStudies/bitFlipTrainingStudy/submit.sh 1.0
```

Here, `1.0` means an independent 1% training flip probability per eligible
site. For `s06`, this is approximately 773 expected flips per event and is
close to the existing `K=768` inference evaluation point.

To use a specific isolated output root:

```bash
bash lgnStudies/bitFlipTrainingStudy/submit.sh 1.0 \
  results/SLURM/my_model2_fault_aware_training_p1pct
```

The submitter refuses to reuse an existing campaign root. To study several
training probabilities, launch each into a separate root:

```bash
bash lgnStudies/bitFlipTrainingStudy/submit.sh 0.1
bash lgnStudies/bitFlipTrainingStudy/submit.sh 1.0
bash lgnStudies/bitFlipTrainingStudy/submit.sh 5.0
```

Each invocation submits:

1. a three-task GPU array, one five-hour fault-aware training job per model;
2. a dependent 21-task GPU array for the exact-count inference sweep;
3. a dependent CPU job that validates, combines, and plots all results.

Every training task uses the same dataset, initialization seed, class-weighted
cross-entropy, learning rate, batch size, and Torch compilation setting as the
existing model-2 training study. It must complete both at least 20,000
optimizer steps and at least five training-loop hours. A stop request or
failure to meet either floor produces `interrupted_model.pth`, never
`final_model.pth`; dependent evaluation then cannot start.

## Outputs

The default root is:

```text
results/SLURM/YYYY_MM_DD_model2_fault_aware_training_PCTpct_5h_20k/
```

It contains `campaign_config.json` and one directory per model. Training
artifacts are isolated below `MODEL/training/`:

- `final_model.pth`: published only after both completion requirements;
- `training_complete.json`: exact probability, fault-space, duration, and
  final clean metric provenance;
- `fault_aware_training_history.csv`: clean balanced accuracy and all three
  clean class recalls at every recorded stage;
- `clean_accuracy_during_fault_aware_training.png` and `.pdf`;
- `latest_model.pth`, or `interrupted_model.pth` after an incomplete run.

The final inference-robustness artifacts are written at `MODEL/`:

- `bit_flips_per_event.png` and `.pdf`;
- `bit_flip_percentage.png` and `.pdf`;
- `bit_flip_results.csv`;
- `bit_flip_metadata.json`;
- `clean_predictions.npy`.

The campaign root also receives the combined three-model plots. Intermediate
evaluation chunks remain under `MODEL/chunks/` for auditing. If the
inference-only campaign root exists at
`results/SLURM/2026_07_04_model2_bit_flip_5h_20k_dense_sweep/`, the combine
job also writes one comparison graph per model:

- `bit_flip_accuracy_comparison_s06_p760M_tau40.png` and `.pdf`;
- `bit_flip_accuracy_comparison_s08_p1202M_tau40.png` and `.pdf`;
- `bit_flip_accuracy_comparison_s09_p1460M_tau40.png` and `.pdf`.

To redraw only those comparison plots from saved CSVs, without training or
evaluation, run:

```bash
python -m lgnStudies.bitFlipStudy.compare_training \
  --inference-root results/SLURM/2026_07_04_model2_bit_flip_5h_20k_dense_sweep \
  --fault-aware-root results/SLURM/2026_07_05_model2_fault_aware_training_1pct_5h_20k
```

## S08 Training-Percentage Sweep

To vary the training-time fault percentage without launching all three
models, use the s08-only sweep:

```bash
bash lgnStudies/bitFlipTrainingStudy/submit_s08_percentage_sweep.sh \
  results/SLURM/$(date +%Y_%m_%d)_s08_fault_aware_training_pct_sweep_5h_20k \
  0.1 0.3 1.0 3.0 5.0
```

This submits one training job per percentage for `s08_p1202M_tau40`, plus an
s08 inference-only baseline evaluated on the same percentage grid. The
inference-time x-axis is always stored as equivalent flipped activation sites
per event [%], and the grid ends at 5.3% for s08 rather than stopping at the
old 4096-count point.

The final sweep root receives:

- `s08_training_percentage_sweep_accuracy.png` and `.pdf`: regular 2D lines;
- `s08_training_percentage_sweep_accuracy_3d.png` and `.pdf`: 3D-style view;
- `s08_training_percentage_sweep_accuracy_fit.png` and `.pdf`: 2D lines with
  exponential-decay fits;
- `s08_training_percentage_sweep_accuracy_fit_3d.png` and `.pdf`: 3D-style
  view with the same fits.

## S08 Input-vs-Whole Fault-Scope Sweep

After the whole-model s08 percentage sweep is complete, reuse it and launch
only the missing input-only scope:

```bash
bash lgnStudies/bitFlipTrainingStudy/submit_s08_fault_scope_sweep.sh \
  results/SLURM/$(date +%Y_%m_%d)_s08_input_vs_whole_fault_scope_5h_20k \
  0.1 0.3 1.0 3.0 5.0
```

By default this reuses:

```text
results/SLURM/2026_07_14_s08_fault_aware_training_pct_sweep_5h_20k
```

as the completed whole-model scope. Override it only if needed:

```bash
S08_WHOLE_SCOPE_ROOT=results/SLURM/my_completed_s08_whole_scope \
  bash lgnStudies/bitFlipTrainingStudy/submit_s08_fault_scope_sweep.sh \
  results/SLURM/my_s08_input_vs_whole_scope 0.1 0.3 1.0 3.0 5.0
```

The new Slurm jobs train and evaluate `--fault-scope input` only. The input
scope has 1,400 eligible sites for `s08_p1202M_tau40`, so the exact-count
evaluation grid ends at `K=74`, about 5.3% of that scope. The combine job
loads the existing whole-model CSVs/chunks and writes comparison plots into
the new root:

- `s08_input_scope_training_percentage_sweep_accuracy*`: input-only standalone
  2D, 3D-style, and fitted plots;
- `s08_whole_scope_training_percentage_sweep_accuracy*`: redrawn whole-model
  plots in the same output root for direct visual comparison;
- `s08_fault_scope_training_percentage_sweep_accuracy*`: side-by-side
  whole-model vs input-only plots, with inference-only training as a solid
  line and fault-aware training percentages as dashed lines.

## Interpretation

The clean-trained checkpoints and plots from `bitFlipStudy` are the baseline.
Compare them with the fault-aware campaign at:

- `K=0`, to test whether noise-based training changed clean generalization;
- every `K>0`, to test whether fault-aware training improved robustness;
- the per-class recall panel, to expose class-specific tradeoffs;
- the repair/damage panel, to explain non-monotonic balanced accuracy.

More x-axis points improve curve resolution; they do not reduce statistical
uncertainty. The evaluation therefore retains five independent full-test
fault realizations and two-sided 95% Student-t confidence intervals at every
nonzero `K`.
