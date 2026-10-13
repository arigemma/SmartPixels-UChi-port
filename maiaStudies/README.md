# MAIA campaign details

## Data preparation

`prepare.slurm` streams the compressed reconstruction Parquet files in small
batches. It writes memory-mapped Model-2 inputs (21-bin x profile, 13-bin y
profile, and three position values) and Model-3 inputs (the full 13x21 spatial
cluster plus the same position values), split-index arrays, timing metadata,
and training-only fixed thresholds. It never calls `pandas.read_parquet` on a
complete reconstruction file.

The charge cluster is summed over the 20 time slices. `n_module` and `x_local`
are deterministically derived from `z-global` using the published
13-mm/module barrel geometry; `y-local` is read directly. Timing, momentum,
and direction labels remain excluded. `audit_cache.py` verifies schemas,
threshold ordering and finite values, then requires a lightweight diagnostic
classifier to obtain validation ROC AUC above 0.60 before GPU jobs are allowed.

The cache is finalized by `preparation_complete.json`; without this marker,
training refuses to run. Re-running the normal submit script reuses a complete
cache. Deliberate regeneration requires running `python -m maiaStudies.prepare
... --overwrite-cache` directly.

## Meaning of rejection metrics

`signal` is the positive class. Thus:

- signal efficiency = accepted signal / all signal;
- BIB acceptance = BIB misclassified as signal / all BIB;
- BIB rejection factor = 1 / BIB acceptance;
- BIB rejection fraction = 1 - BIB acceptance.

At fixed target signal efficiencies, score thresholds are selected on the
validation split and then applied unchanged to the test split. This avoids
choosing the deployment threshold on test data. The same procedure is repeated
inside the timing-cut validation/test subsets.

## Job chain and recovery

The mandatory pilot chain is:

```text
prepare + cache audit -> 2 pilot training tasks -> 2 pilot evaluations -> gate
```

Only after that gate passes is the full chain enabled:

```text
30 training tasks -> 30 evaluation tasks -> summary/plots
```

Every downstream dependency is `afterok`, so a missing model cannot silently
produce a partial final plot. Job IDs and exact roots are recorded in
`submitted_jobs.txt`.

Check failures with:

```bash
ROOT=results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k
source "$ROOT/submitted_jobs.txt"
sacct -X -j "$prepare_job_id,$training_job_id,$evaluation_job_id,$summary_job_id" \
  --format=JobID%24,JobName%24,State%22,ExitCode%12,Elapsed
```

The Slurm files are idempotent at the task level: completed preparation,
training, and evaluation outputs are skipped. Failed array elements can be
resubmitted using the same exported `MAIA_CACHE_ROOT` and `MAIA_RESULT_ROOT`.

## Comprehensive summary plots

Once all 30 evaluations are complete, regenerate the complete paper-ready
summary package without rerunning training or inference:

```bash
ROOT="$PWD/results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k"
bash maiaStudies/plot_summary.sh "$ROOT"
```

Dense Model 2 is blue and convolutional Model 3 is orange in every figure.
Marker shape and line style identify `tau=10`, `20`, and `40`. Exact model
aliases are included as annotations and in the ranking figures. Set
`MAIA_PLOT_ANNOTATIONS=0` for less crowded scatter plots. Plot resolution and
font scale can be overridden, for example:

```bash
MAIA_PLOT_DPI=400 MAIA_PLOT_FONT_SCALE=1.25 \
  bash maiaStudies/plot_summary.sh "$ROOT"
```

For a preliminary plot while only some evaluations exist, explicitly allow a
partial summary:

```bash
MAIA_ALLOW_PARTIAL=1 bash maiaStudies/plot_summary.sh "$ROOT"
```

Chance-level campaigns are rejected before presentation plots are produced.
For forensic plotting only, add `MAIA_ALLOW_COLLAPSED=1`; such output must not
be used as a result figure.

The final package contains:

- `summary_plots/`: 18 PNG/PDF figures covering balanced accuracy, ROC AUC,
  signal efficiency, BIB rejection, parameter and logic-node proxies,
  architecture/tau heatmaps, rankings, full rejection curves, and the timing
  selection comparison;
- `performance_summary.tsv`, `.csv`, and `.json`: long-form plotting points,
  with separate full-test and timing-cut rows;
- `model_comparison_wide.tsv`: one row per model with both evaluations;
- `best_models.tsv`: top-five Model 2 and Model 3 configurations under the
  principal performance criteria;
- `SUMMARY.md`: a human-readable result digest;
- `summary_manifest.json`: the exact products and visual style mapping.

The reported parameter count is the exact trainable PyTorch parameter count.
The logic-node count remains an architectural proxy and must not be described
as synthesized LUT+FF utilization.
