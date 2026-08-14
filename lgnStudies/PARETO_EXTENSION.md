# Dense and convolutional LGN Pareto extension

The fixed campaign replaces the 2026-08-13 run whose long-lived loss tensor
retained autograd graphs and whose largest convolutional architecture exceeded
40 GB at batch size 1024. The fixed run accumulates detached scalar losses,
uses batch size 512 consistently for all three `s10_p2623M` taus, and catches
the five-minute Slurm warning so it can publish `final_model.pth` cleanly.

## Train

From the SmartPixels repository root on Della:

```bash
bash submit_pareto_extension_training.sh
```

This submits 51 jobs and writes new results without modifying the original
campaign:

```text
results/SLURM/2026_08_14_model2lgnFull_left_extension_fixed/
results/SLURM/2026_08_14_model3lgnConv_curve_extension_fixed/
```

## Evaluate and stage

After the training array has finished, run:

```bash
bash submit_pareto_extension_evaluation.sh
```

The launcher refuses a partial grid. It submits one dense and one
convolutional GPU evaluation task, evaluates every `best_model.pth` in exact
discrete mode, and selects the maximum-balanced-accuracy tau for every size.
Outputs are written under:

```text
results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc/
  dense/
    balanced_accuracy_summary.tsv
    best_by_size.tsv
    transfer_ready/
  conv/
    balanced_accuracy_summary.tsv
    best_by_size.tsv
    transfer_ready/
```

The two `balanced_accuracy_summary.tsv` files contain all 51 scores. The two
`best_by_size.tsv` files identify the 17 checkpoints to synthesize.

## Transfer

Copy only the selected checkpoints to the synthesis host:

```bash
scp -r \
  results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc/dense/transfer_ready/* \
  mbileska@login.hep.wisc.edu:/path/to/synthesis-SmartPixels/models/model2/LGN/

scp -r \
  results/evaluation/2026_08_14_lgn_pareto_extensions_discrete_balacc/conv/transfer_ready/* \
  mbileska@login.hep.wisc.edu:/path/to/synthesis-SmartPixels/models/model3/LGN/
```

On the synthesis host, verify the independent checkpoint copies before
conversion:

```bash
find models/model2/LGN models/model3/LGN \
  -name SHA256SUMS -execdir sha256sum -c SHA256SUMS \;
```
