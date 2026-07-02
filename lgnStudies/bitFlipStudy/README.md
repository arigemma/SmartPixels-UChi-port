# Exact-count bit-flip robustness study

This is an inference-only study of the unchanged final checkpoints for the
three selected tau-40, model-2 y-profile LGNs:

- `s06_p760M_tau40`
- `s08_p1202M_tau40`
- `s09_p1460M_tau40`

There is no fault-aware fine-tuning and no random-class baseline.

## Fault definition

For every test event, exactly `K` distinct fault sites are sampled uniformly
without replacement from one global signal space. That space is the
concatenation of the Boolean outputs of:

1. the existing TorchLogix `FixedBinarization`;
2. each of the seven existing TorchLogix `LogicDense` layers.

Each selected signal is changed by `x -> 1-x`. Faults are transient, are
resampled for each forward pass, and do not modify model parameters,
thresholds, learned logic functions, or fixed connection indices. `GroupSum`
is excluded because it is an arithmetic class accumulator rather than a
Boolean activation stage.

The primary sweep is:

```text
K = 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

The percentage plot is retained as a secondary normalization. It uses
`100 * K / N`, where `N` is the number of eligible activation sites in that
model. It is derived from the exact same runs; it is not an independent
Bernoulli-per-site experiment.

This distinction matters. An exact-count x-axis directly answers how many
faults the model tolerates. An independent probability only controls an
expected count, which varies event by event and differs between model sizes.

## Statistics and metrics

`K=0` is evaluated once. Every nonzero K is evaluated with five independent
fault seeds. Every evaluation covers the complete configured test split.

Balanced accuracy is computed once from the aggregate 3-by-3 confusion
matrix:

```text
balanced accuracy = (recall_0 + recall_1 + recall_2) / 3
```

It is not a mean of batch accuracies and there is no scale factor. The plot
shows the mean and a two-sided 95% Student-t confidence interval across the
five complete-test fault realizations. A second panel shows the fraction of
events whose predicted class differs from that same event at `K=0`.

A small upward movement in balanced accuracy is possible: a random fault can
correct an event the clean model classified incorrectly. It is only evidence
of an improvement if it is reproducible and exceeds the uncertainty interval.
The prediction-change panel exposes how much behavior changed even when
corrected and damaged events happen to cancel in aggregate accuracy.

## Cluster workflow

The study requires the three `final_model.pth` files produced by the resource
study. If resource array job `10570179` is still running:

```bash
git pull
scancel 10570203 10570204 2>/dev/null || true
bash lgnStudies/bitFlipStudy/submit.sh 10570179
```

The new bit-flip array waits for the resource array to finish successfully.
If all three final checkpoints already exist, submit without a dependency:

```bash
git pull
bash lgnStudies/bitFlipStudy/submit.sh
```

For a completely fresh resource and bit-flip campaign:

```bash
bash lgnStudies/submit_training_studies.sh
```

The clean checkpoints were trained by the resource study with the model-2
learning rate, batch size, and `--compile-model`. This inference study does
not train or compile a replacement model; it calls the existing model's
`bin`, `layers`, and `group_sum` modules directly so faults can be inserted at
the defined stage boundaries.

## Outputs

The result root is:

```text
results/SLURM/2026_07_02_model2_bit_flip_inference_exact_count/
```

Each model directory contains:

- `bit_flips_per_event.png` and `.pdf`: primary exact-K paper plot;
- `bit_flip_percentage.png` and `.pdf`: secondary normalized plot;
- `bit_flip_results.csv`: seeds, recalls, class totals, confusion matrices,
  balanced accuracy, and prediction-change fractions;
- `bit_flip_metadata.json`: exact fault-space and evaluation definitions;
- `clean_predictions.npy`: event-aligned K=0 predictions used for comparison.

The dependent CPU job creates the corresponding combined three-model plots in
the common result root. FPGA synthesis is not needed for this transient
inference-fault study.
