# Model 3 compact/depth-assisted frontier extension

This campaign adds ten Model 3 convolutional LGN architectures specifically
for the left and upper-left portions of the FPGA-resource versus balanced-
accuracy plot. It does not extend the Pareto-dominated giant-width
`model3Curve` branch.

All points retain the established Model 3 recipe: 100-bin fixed input
binarization, four logic convolutions with the existing OR pool and strides,
the `y0` bypass, logic-dense classification, class-weighted cross entropy,
AdamW at `0.01`, batch size `1024`, compiled training, and taus 10, 20, and 40.

The first five points reduce convolutional and dense widths to reach below the
current `p046M` point. The last five progressively add useful dense logic depth
while widening moderately, connecting toward the high-accuracy `model3Deep`
cluster. Connectivity proxy parameters are conventional dense/conv connection
counts and are architecture labels, not FPGA resource measurements.

| Size | k1/k2/k3/k4 | fc_base | Extra h1→h1 layers | Proxy parameters |
|---|---:|---:|---:|---:|
| `s01_p003M` | 16/32/64/60 | 300 | 0 | 2.964M |
| `s02_p007M` | 24/48/96/90 | 450 | 0 | 6.568M |
| `s03_p012M` | 32/64/128/120 | 600 | 0 | 11.587M |
| `s04_p018M` | 40/80/160/150 | 750 | 0 | 18.021M |
| `s05_p026M` | 48/96/192/180 | 900 | 0 | 25.870M |
| `s06_p039M` | 48/96/192/180 | 900 | 1 | 38.830M |
| `s07_p069M` | 64/128/256/240 | 1200 | 1 | 68.851M |
| `s08_p110M` | 96/192/384/360 | 1500 | 1 | 110.220M |
| `s09_p213M` | 128/256/512/480 | 1800 | 2 | 213.178M |
| `s10_p385M` | 192/384/768/720 | 2400 | 2 | 384.979M |

## One-command launch

From the repository root on Della:

```bash
bash submit_model3_frontier.sh
```

This submits a 30-task training array (ten architectures times three taus,
maximum six concurrent) and one strict dependent discrete-evaluation job. The
training jobs receive an advance signal 15 minutes before their six-hour
limit, allowing `train.py` to finish the current batch and publish the final
checkpoint. The evaluator requires all 30 nonempty best checkpoints before it
runs the complete test split.

Default outputs are:

```text
results/SLURM/2026_08_24_model3lgnConv_frontier_extension/
  sXX_pXXXM/tau_{10,20,40}/best_model.pth

results/evaluation/2026_08_24_model3Frontier_discrete_balacc/
  balanced_accuracy_summary.tsv
  best_by_size.tsv
  transfer_ready/
```

Print the selected tau and score for every architecture with:

```bash
EVAL=results/evaluation/2026_08_24_model3Frontier_discrete_balacc
column -s $'\t' -t < "$EVAL/best_by_size.tsv"
```

If any training task fails or does not publish a checkpoint, reuse completed
work and rerun complete tau triplets for only the affected sizes:

```bash
bash resume_model3_frontier.sh
```

After synthesis, plot this `model3Frontier` family separately from the old
giant-width `model3Curve` and fixed-width `model3Deep` families. Whether the
ten points form a genuine Pareto curve must be decided from measured Vivado
LUT+FF results, not from the proxy-parameter labels.
