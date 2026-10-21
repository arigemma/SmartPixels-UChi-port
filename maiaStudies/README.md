# MAIA campaign details

This campaign applies the established 14-input SmartPixels LNN Model 2 and
105-input LNN Model 3 to the MAIA binary target. These names differ from the
MAIA reference paper's own Model 2 (37 inputs) and Model 3 (276 inputs). Never
combine their results without recording the input schema.

## Data preparation

`prepare.slurm` streams the compressed reconstruction Parquet files in small
batches. It writes memory-mapped canonical Model-2 inputs (`y0` plus the
final-frame 13-bin y profile) and Model-3 inputs (`y0` plus the 13x8 y/time
profile from slices 0--7), split-index arrays, timing metadata, and
training-only fixed thresholds. It never calls `pandas.read_parquet` on a
complete reconstruction file.

The symmetric-log transform is applied per pixel before averaging over the 21
x positions. `y-local` supplies `y0`; no other truth coordinates are model
inputs. Truth timing, momentum, and direction remain excluded. `audit_cache.py` verifies schemas,
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

Each training task performs an exact checkpoint round-trip test: the selected
checkpoint is loaded into a freshly constructed model and must reproduce the
stored validation metrics before `training_complete.json` is written.
The checkpoint stores fixed TorchLogix connection indices explicitly because
the wiring is part of the trained model and cannot be randomized again when
the learned LUT weights are loaded.

Only after that gate passes is the full chain enabled:

```text
30 training tasks -> 30 evaluation tasks -> summary/plots
```

Every downstream dependency is `afterok`, so a missing model cannot silently
produce a partial final plot. Job IDs and exact roots are recorded in
`submitted_jobs.txt`.

Check failures with:

```bash
ROOT=results/SLURM/2026_10_21_maia_signal_vs_bib_canonical_inputs_5h_20k
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
ROOT="$PWD/results/SLURM/2026_10_21_maia_signal_vs_bib_canonical_inputs_5h_20k"
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

## Tau=1,3 and relaxed/discrete extension

The original 30 plotted test results are all discrete circuit evaluations.
Training uses TorchLogix's relaxed soft mixture of truth tables, but the old
campaign retained only discrete validation metrics, so its relaxed trajectory
cannot be reconstructed after the fact. To test whether the uniform ordering
`tau=10 > tau=20 > tau=40` is caused by GroupSum optimization rather than
model capacity, run the extension after all 30 original evaluations exist:

```bash
bash maiaStudies/submit_tau_extension.sh
```

This trains the same five dense and five convolutional architectures at
`tau=1` and `tau=3` for the same five-hour/20,000-step minimum. At each
validation checkpoint it evaluates the identical validation subset once in
relaxed mode and once in discrete mode; checkpoint selection uses discrete BIB
rejection fraction at the ≥99% signal-efficiency operating point. It then
evaluates the 20 new checkpoints and regenerates the summary over all 50
models. New diagnostics are:

- one `training_history.tsv` per tau-extension model;
- `relaxed_discrete_training_history.tsv` at the campaign root;
- `summary_plots/19_relaxed_vs_discrete_training.{png,pdf}`.

The extension jobs and dependencies are recorded separately in
`submitted_tau_extension_jobs.txt`, so the original submission record is not
overwritten.
