# Internal activation random-overwrite study

This campaign trains and evaluates the two matched-clean-accuracy models:

- LNN `s02_p163M_tau20` (registry
  `lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_20`);
- QKeras `qkeras_y256_64_w4a8` (registry
  `qkeras-model-2-yprofile-d256-d64-w4a8`).

It trains every model from scratch at internal overwrite-selection rates
`0, 1, 3, 10%`. The 0% condition is a newly trained clean baseline. The
existing recipes are retained: LNN uses at least 20,000 steps and five hours;
QKeras uses at least 6,000 steps and three hours.

## Fault protocol

Only hidden activation outputs are eligible:

- LNN: the outputs of all seven `LogicDense` layers, 34,650 Boolean sites;
- QKeras: the outputs of the 256- and 64-unit `QActivation` layers at eight
  bits per activation, 2,560 bit sites.

Inputs, model parameters, connections, biases, batch-normalization state,
output logits, and LNN `GroupSum` are excluded. During training, sites are
independently selected with probability `p`. During evaluation, exactly `K`
distinct sites are selected uniformly without replacement for each event.
Every selected site is independently assigned a uniformly random 0 or 1.
Thus, the x-axis is the selected overwrite-site rate; only half of selected
writes change the existing value in expectation. Faults are transient and
resampled for every event/forward pass.

Every checkpoint is evaluated at 16 targets from 0 to 10%, with extra
resolution below 3%. This covers and extends the 5.3--7% range of the earlier
internal-activation studies without spending GPU time on extremely large
fault plans. Nonzero points use five independent full-test realizations. The
actual exactly representable percentages are retained in the output CSV.

## Launch everything

From the repository root:

```bash
module load anaconda3/2025.12
conda activate smart_pixels
bash internalBitOverwriteStudy/submit.sh
```

The launcher installs the pinned Google `QKeras==0.9.0` wheel without
dependencies into `.runtime/google-qkeras-0.9.0` and prepends only that small
directory to `PYTHONPATH`. This process-level overlay is necessary when HGQ2
is installed in the same Conda environment because HGQ2 also provides an
incompatible package named `qkeras`. It does not uninstall or modify HGQ2.

You may supply a new campaign root as the first argument. The launcher refuses
to overwrite an existing root. It submits eight training jobs, 40 dependent
evaluation jobs, and one final validation/combine/plot job. The final job runs
only if every upstream task succeeds.

The three submitted SLURM job IDs are also written to
`submitted_jobs.txt` inside the campaign root, so they do not need to be
copied from the terminal output.

If only QKeras training tasks 4--7 from an existing campaign failed before
training began, retain the original LNN jobs and run:

```bash
bash internalBitOverwriteStudy/recover_qkeras.sh CAMPAIGN_ROOT ORIGINAL_TRAINING_JOB_ID
```

The recovery launcher runs a local QKeras import/model-build preflight,
resubmits only tasks 4--7, and makes its new evaluation array wait for both
the four original LNN tasks and the replacement QKeras array.

If completed QKeras evaluations exist but high-corruption chunks reached the
six-hour limit, do not repeat successful points. Run:

```bash
bash internalBitOverwriteStudy/retry_qkeras_evaluations.sh \
  CAMPAIGN_ROOT PRIOR_EVALUATION_ARRAY_JOB_ID
```

The launcher validates completed artifacts, writes a frozen retry manifest,
and submits one missing corruption point per job with all five repetitions.
The final combiner ignores incomplete timeout artifacts but requires exactly
the complete configured point/repetition grid before producing a plot.

The training grid is configuration-driven. A future fresh campaign can use a
larger grid without code changes, for example:

```bash
INTERNAL_OVERWRITE_TRAINING_PERCENTAGES="0 1 2 3 4 5 6 7 8 9 10" \
  bash internalBitOverwriteStudy/submit.sh results/SLURM/my_extended_campaign
```

## Outputs and status

```bash
ROOT=results/SLURM/YYYY_MM_DD_selected_internal_random_overwrite_0_1_3_10pct
python -m internalBitOverwriteStudy.campaign --status --root "$ROOT"
```

The final artifacts are:

- `lnn_vs_qkeras_internal_random_overwrite.png` and `.pdf`: paper-ready
  side-by-side comparison;
- `internal_random_overwrite_plot_points.csv`: aggregated plotted x/y values,
  95% confidence intervals, exact counts, and styles;
- `internal_random_overwrite_raw_results.csv`: every repeat-level result;
- `internal_random_overwrite_campaign_summary.json`: complete provenance and
  protocol summary.

Each model/training-condition directory also contains its checkpoint,
training metadata/history, raw evaluation chunks, combined result table, and
clean reference predictions.
