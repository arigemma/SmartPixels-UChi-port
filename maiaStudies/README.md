# MAIA campaign details

## Data preparation

`prepare.slurm` streams the compressed reconstruction Parquet files in small
batches. It writes memory-mapped 13-bin dense profiles and 13x20 temporal
profiles, split-index arrays, timing metadata, and training-only fixed
thresholds. It never calls `pandas.read_parquet` on a complete reconstruction
file.

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

The standard chain is:

```text
prepare -> 30 training tasks -> 30 evaluation tasks -> summary/plots
```

Every downstream dependency is `afterok`, so a missing model cannot silently
produce a partial final plot. Job IDs and exact roots are recorded in
`submitted_jobs.txt`.

Check failures with:

```bash
ROOT=results/SLURM/2026_10_07_maia_signal_vs_bib_lnn_5h_20k
source "$ROOT/submitted_jobs.txt"
sacct -X -j "$prepare_job_id,$training_job_id,$evaluation_job_id,$summary_job_id" \
  --format=JobID%24,JobName%24,State%22,ExitCode%12,Elapsed
```

The Slurm files are idempotent at the task level: completed preparation,
training, and evaluation outputs are skipped. Failed array elements can be
resubmitted using the same exported `MAIA_CACHE_ROOT` and `MAIA_RESULT_ROOT`.
