# Selected-model input bit-flip study

This campaign recreates the input-only bit-flip training study for the two
points selected from the `fit_10_triplet_architecture_slide_title` resource
plot:

- second Dense LGN point from the left:
  `s02_p163M_tau20`, balanced accuracy `0.807266`, LUT+FF `11578`;
- third Dense QKeras point from the right:
  `qkeras_y256_64_w4a8`, balanced accuracy `0.798000`, LUT+FF `275815`.

The LGN plot point establishes tau `20`; the campaign therefore uses the
registered `s02_p163M_tau20` architecture rather than selecting a new tau.
This study is independent of the newly launched Pareto-extension models.

## What is trained

For each architecture the campaign trains from the same fixed seed:

1. one clean baseline with no training-time faults;
2. ten fault-aware models with independent input-bit flip probabilities of
   `1, 2, 3, ..., 10%` per training event.

Only input bits are eligible. LGN faults complement the 1,400 Boolean outputs
of `FixedBinarization`. QKeras faults flip the 112 bits representing 14 input
features at eight activation bits each. Internal activations, parameters,
connections, `GroupSum`, and logits are not faulted.

The existing training recipes are retained:

- LGN: class-weighted cross-entropy, AdamW, learning rate `0.01`, batch size
  `1024`, Torch compilation, at least 20,000 steps and five training hours;
- QKeras: the existing class-weighted QKeras loop, Adam, learning rate `0.001`,
  batch size `1024`, at least 6,000 steps and three training hours.

## Evaluation

Every one of the 22 checkpoints is evaluated at target input-fault
percentages `0, 5, 10, ..., 60%`. Each nonzero point uses five independent
full-test realizations. Evaluation samples exactly `K` distinct random input
bits per event without replacement, so it is controlled rather than
Bernoulli-distributed. QKeras counts are rounded to the nearest representable
percentage of its 112-bit input space and the exact realized percentages are
saved in campaign metadata.

The final x-axis is ordinary linear scale, from 0 to 60%. No logarithmic or
symlogarithmic scaling and no fitted curves are used.

## One-command launch

Run from the repository root:

```bash
bash inputBitFlipStudy/submit.sh
```

An explicit new output root can be supplied:

```bash
bash inputBitFlipStudy/submit.sh \
  results/SLURM/my_selected_input_bit_flip_campaign
```

The launcher refuses to reuse an existing root and submits:

1. a 22-task GPU training array;
2. a dependent 88-task GPU exact-count evaluation array;
3. a dependent CPU validation/combine/plot job.

The default output root is:

```text
results/SLURM/YYYY_MM_DD_selected_input_bit_flip_1to10pct_eval60_linear/
```

## Outputs

Each condition retains its checkpoint, training completion metadata, training
history, raw evaluation chunks, combined CSV, bit-flip metadata, and clean
predictions. The campaign root receives:

```text
campaign_config.json
input_bit_flip_campaign_summary.json
s02_p163M_tau20_input_bit_flip_robustness_linear.png
s02_p163M_tau20_input_bit_flip_robustness_linear.pdf
qkeras_y256_64_w4a8_input_bit_flip_robustness_linear.png
qkeras_y256_64_w4a8_input_bit_flip_robustness_linear.pdf
```

The plots show the clean-trained baseline as a solid line and all ten
fault-aware training conditions as dashed lines, with balanced accuracy on
the y-axis and random input-bit faults per event [%] on a linear x-axis.

## Resuming failed evaluation tasks

The trained checkpoints and completed evaluation chunks can be reused. After
fixing the cause of a failed task, provide the existing campaign root and the
Slurm array indices to rerun. The launcher submits only those evaluation tasks
and a dependent combine/plot job:

```bash
bash inputBitFlipStudy/resume_failed_evaluation.sh \
  results/SLURM/YYYY_MM_DD_selected_input_bit_flip_1to10pct_eval60_linear \
  0-43,84-87
```

For the 2026-08-13 campaign, indices `0-43` are the LGN chunks and `84-87`
are the QKeras 10%-trained chunks. The launcher refuses to run unless all 22
training completion records exist.

## Refining the 0--10% evaluation region

The completed 22 checkpoints can be evaluated at two additional target
percentages, 2.5% and 7.5%, without retraining:

```bash
bash inputBitFlipStudy/submit_0to10pct_refinement.sh
```

This submits one two-point evaluation task per checkpoint and a dependent
validation/combine/plot job. It retains every original result and plot. LGN
uses the exact counts 35 and 105 of 1,400 input sites. QKeras uses the nearest
representable exact counts, 3 and 8 of 112 input bits (2.68% and 7.14%).

The new plots are:

```text
s02_p163M_tau20_input_bit_flip_robustness_linear_refined_0to10pct.png
s02_p163M_tau20_input_bit_flip_robustness_linear_refined_0to10pct.pdf
qkeras_y256_64_w4a8_input_bit_flip_robustness_linear_refined_0to10pct.png
qkeras_y256_64_w4a8_input_bit_flip_robustness_linear_refined_0to10pct.pdf
```

Per-condition merged data are written as
`bit_flip_results_refined_0to10pct.csv`; the original
`bit_flip_results.csv` files are not modified.
