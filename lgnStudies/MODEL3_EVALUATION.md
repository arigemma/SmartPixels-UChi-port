# Model-3 convolutional LGN tau selection

The June-25 full-width convolutional scan contains five model sizes and three
tau values per size. Tau is treated as a hyperparameter: evaluate all 15 exact
checkpoints in discrete mode, then retain only the highest-balanced-accuracy
tau for each size.

From the SmartPixels repository root on Della, the complete Della-side
workflow is one command:

```bash
sbatch eval_lgn_discrete.slurm
```

The job strictly requires this grid:

```text
results/SLURM/2026_06_25_model3lgnConv_expanded/
  s01_p319M/tau_{20,40,60}/best_model.pth
  s02_p465M/tau_{20,40,60}/best_model.pth
  s03_p639M/tau_{20,40,60}/best_model.pth
  s04_p840M/tau_{20,40,60}/best_model.pth
  s05_p1069M/tau_{20,40,60}/best_model.pth
```

It evaluates the complete test split through `eval_acceptance.py` with
`--candidate-forward discrete`. It fails instead of publishing a partial
selection if any checkpoint or evaluation row is missing. On success it
writes:

```text
results/evaluation/2026_08_13_model3Full_discrete_balacc/
  balanced_accuracy_summary.tsv
  best_by_size.tsv
  transfer_ready/
    balanced_accuracy_summary.tsv
    best_by_size.tsv
    selection_manifest.json
    s01_p319M_tauXX/
      best_model.pth
      selection.json
      SHA256SUMS
    ...one directory for each of the five selected sizes...
```

`transfer_ready/` contains independent copies of exactly five full-module
checkpoints. It never stages plain state dicts or all 15 tau candidates. Copy
that directory to the synthesis host after the job succeeds, for example:

```bash
scp -r \
  results/evaluation/2026_08_13_model3Full_discrete_balacc/transfer_ready/* \
  mbileska@login.hep.wisc.edu:/path/to/synthesis-SmartPixels/models/model3/LGN/
```

After transfer, verify the checkpoint copies on the synthesis host:

```bash
find models/model3/LGN -name SHA256SUMS -execdir sha256sum -c SHA256SUMS \;
```

The earlier June-27 deeper-model evaluator remains available as
`eval_model3_deep_discrete.slurm`; the primary `eval_lgn_discrete.slurm` is the
June-25 evaluate-select-stage entry point.
