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

The dense primary sweep is:

```text
K = 0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256,
    384, 512, 768, 1024, 1536, 2048, 3072, 4096
```

This gives 22 points and extends the study to approximately 3.8--5.3% of
the raw activation fault space, depending on model size.

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
five complete-test fault realizations. The other panels show all three
recalls, the fraction of predictions changed from `K=0`, and the balanced
repair and damage rates.

A small upward movement in balanced accuracy is possible: a random fault can
correct an event the clean model classified incorrectly. It is only evidence
of an improvement if it is reproducible and exceeds the uncertainty interval.
The prediction-change panel exposes how much behavior changed even when
corrected and damaged events happen to cancel in aggregate accuracy.

The CSV and per-model plots now separate:

- clean-wrong to fault-correct events ("repairs");
- clean-correct to fault-wrong events ("damages");
- all changed predictions;
- recall for every true class.

The balanced repair gain and balanced damage loss use the same equal-class
weighting as balanced accuracy. Consequently,
`balanced_accuracy(K) - balanced_accuracy(0) = repair_gain - damage_loss`
apart from floating-point rounding. This directly explains any upward or
downward movement.

## Parallel sweep layout

One model can no longer consume the entire six-hour job while evaluating the
full grid. The campaign is divided into seven chunks of three nonzero counts
per model. The Slurm array therefore contains 21 tasks:

```text
3 models x 7 count chunks = 21 GPU jobs
```

Each task evaluates `K=0` plus its three assigned counts, with five complete
test-set fault realizations at every nonzero count. Chunk outputs are isolated
under `MODEL/chunks/chunk_XX`, so array tasks never write the same files.

The dependent combine job verifies every expected chunk, repeat, registry
name, fault-space size, and clean prediction array. It refuses to plot partial
or inconsistent results. After validation it writes the merged CSV, metadata,
clean predictions, and plots at the model directory level.

## Cluster workflow

The study requires the three `final_model.pth` files produced by the resource
study. If a resource array is still running, pass its job ID:

```bash
git pull
bash lgnStudies/bitFlipStudy/submit.sh RESOURCE_JOB_ID
```

The new bit-flip array waits for the resource array to finish successfully.
If all three final checkpoints already exist, do not retrain them; submit
without a dependency:

```bash
git pull
bash lgnStudies/bitFlipStudy/submit.sh
```

For a completely fresh resource and bit-flip campaign:

```bash
bash lgnStudies/submit_training_studies.sh
```

The clean checkpoints are accepted only when `training_complete.json` proves
that the resource job reached both 20,000 optimizer steps and five training
hours without a stop request. Evaluation/export time is excluded from the
five-hour counter. Old, interrupted, or short checkpoints are rejected even
if a file named `final_model.pth` exists.

The resource jobs use the model-2 learning rate, batch size, class-weighted
loss, and `--compile-model`. This inference study does not train or compile a
replacement model; it calls the existing model's `bin`, `layers`, and
`group_sum` modules directly so faults can be inserted at the defined stage
boundaries.

## Outputs

The result root is:

```text
results/SLURM/2026_07_03_model2_bit_flip_inference_dense_sweep/
```

Each model directory contains:

- `bit_flips_per_event.png` and `.pdf`: primary exact-K paper plot;
- `bit_flip_percentage.png` and `.pdf`: secondary normalized plot;
- `bit_flip_results.csv`: seeds, recalls, class totals, confusion matrices,
  balanced accuracy, prediction changes, and repair/damage diagnostics;
- `bit_flip_metadata.json`: exact fault-space and evaluation definitions;
- `clean_predictions.npy`: event-aligned K=0 predictions used for comparison.

Intermediate chunk files remain below `chunks/` for auditing. The dependent
CPU job creates the corresponding combined three-model plots in the common
result root. FPGA synthesis is not needed for this transient inference-fault
study.
