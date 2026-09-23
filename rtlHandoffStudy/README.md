# EFLX RTL handoff campaign

This campaign trains and evaluates 20 deliberately compact LGNs for the
EFLX4K tape-out screen: ten dense Model-2 candidates and ten convolutional
Model-3 candidates.  It is a screening campaign, not a claim that a model
already fits the target.  The final fit decision requires synthesis against
the Flex Logix primitive library, including the input decoder and GroupSum.

The scan varies three costs that matter at handoff:

- 15, 31, 63, or 100 ordered input thresholds per continuous feature;
- fixed quantile thresholds versus consistently ordered learned thresholds;
- the number of Boolean outputs accumulated per class by `GroupSum`.

The compact external interface transmits an unsigned ordinal code per
feature rather than the full thermometer vector.  A model with `T`
thresholds uses `ceil(log2(T+1))` pins per feature.  Thus dense T63 uses 84
input bits, convolutional T15 uses 420, T31 uses 525, and convolutional T63
uses 630.  The EFLX4K has 632 accessible input pins, so the T63 convolutional
case is retained only as an I/O-bound control.  T127 is omitted because it
uses the same external seven bits per feature as T100 while making the logic
network larger.

## Launch the full campaign on Della

From a clean SmartPixels checkout:

```bash
git fetch origin
git switch wip-mila-rtl-handoff
git pull --ff-only

module load anaconda3/2025.12
conda activate smart_pixels

python -m unittest rtlHandoffStudy.test_campaign
python -m rtlHandoffStudy.smoke
bash rtlHandoffStudy/submit.sh
```

`submit.sh` launches a 20-task training array, a dependent 20-task exact-test
evaluation array, and a final packaging job.  Every model must satisfy both
20,000 optimizer steps and five hours of training.  Check progress with:

```bash
bash rtlHandoffStudy/status.sh
```

After all jobs leave the queue, if any task failed or hit the wall-time, run:

```bash
bash rtlHandoffStudy/recover.sh
```

The recovery script submits only incomplete training/evaluation work and then
re-runs packaging.  It is safe to run after inspecting the failed logs.

The default result root is:

```text
results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k
```

The important final products are `balanced_accuracy_summary.tsv` and
`transfer_ready/`.  To copy the bundle to Wisconsin:

```bash
ROOT="$PWD/results/SLURM/2026_09_23_eflx_rtl_handoff_5h_20k"
REMOTE=mbileska@login.hep.wisc.edu
REMOTE_ROOT=/afs/hep.wisc.edu/home/mbileska/synthesis-SmartPixels/models/rtl_handoff

ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT'"
scp -r "$ROOT/transfer_ready/." "$REMOTE:$REMOTE_ROOT/"
```

## Per-model artifact contract

Each staged model is under `model2/LGN/<alias>/` or
`model3/LGN/<alias>/` and contains:

- `best_model.importable.pth`: CPU, fixed-threshold, registry-independent
  PyTorch/TorchLogix module for the existing da4ml flow;
- `discrete_netlist.pth`: integer LUT identifiers, connection tensors,
  thresholds, GroupSum contract, and metadata without a campaign model-class
  dependency;
- `thresholds.pth`: final ordered per-feature thresholds;
- `model.json`: architecture, input/output contract, compact interface width,
  and the explicit warning that EFLX fit is not yet established;
- `test_metrics.json`: exact balanced accuracy on the complete unbalanced test
  split, including an ordinal-decoder equivalence check;
- training/completion metadata and SHA-256 checksums.

Learned thresholds are frozen before deployment artifacts are written.  The
portable checkpoint and discrete netlist therefore always expose a
`FixedBinarization`, even when the training configuration says `learnable`.
See `SYNTHESIS_HANDOFF.md` for the required implementation on the synthesis
side.
