# Visualizing a discrete LGN circuit

This folder turns an **existing, clean** dense TorchLogix checkpoint into two
readable circuit-overview PNGs. It does not train or change the model.

- `regular_network.png` shows the full discrete network. Every learned gate is
  included in the per-layer totals and colored by its selected one of 16
  rank-2 Boolean truth tables.
- `compacted_network.png` shows only reachable physical logic after constant
  propagation, identity/passthrough removal, inverter folding, and dead-logic
  removal. These are the same compacting rules used in
  `lgnStudies/resource_accounting.py` before RTL synthesis.
- `circuit_summary.json` records the checkpoint SHA-256, exact counts,
  connectivity/fan-out, compaction statistics, and equivalence-test result.
- `regular_netlist.npz` and `compacted_netlist.npz` retain the exact
  machine-readable circuits behind the overview graphics.

The overview is deliberately aggregated by layer and gate type. The requested
s08 model has 1,400 Boolean threshold inputs and 95,700 learned gates; laying
out every gate and roughly twice as many wires in one PNG would produce an
unusable image and can exhaust a graph layout engine. No gates are sampled or
omitted from the counts or saved netlists.

The implementation uses NumPy and Matplotlib, both open source and already
used by this repository. The `.npz` netlists can also be converted to DOT for
specialized subsets later. Graphviz is designed to turn DOT descriptions into
PNG/SVG, while Yosys `show` generates Graphviz schematics from selected RTL;
both are best used on a selected module or gate subset rather than the whole
s08 flat netlist:

- https://graphviz.org/documentation/
- https://yosyshq.readthedocs.io/projects/yosys/en/latest/cmd/show.html

## Generate the s08 images on the training server

Use the mature **clean** s08 checkpoint from the successful resource study,
not a `bitFlipTrainingStudy/.../training/final_model.pth` fault-aware
checkpoint. From the SmartPixels repository root:

```bash
git switch wip-mila-lgn-analysis
git pull --ff-only origin wip-mila-lgn-analysis

python -m visualizingLGN.visualize_lgn \
  results/SLURM/2026_07_03_model2_resource_precision_5h_20k/s08_p1202M_tau40/final_model.pth \
  --require-s08 \
  --output-dir results/LGN_visualization/s08_clean
```

If your clean run is stored elsewhere, replace only the checkpoint path. A
normal `best_model.pth` produced by `train.py` also works, but the resource
study's `final_model.pth` is preferred because its completion metadata proves
the five-hour/20k-step requirement. A portable
`best_model.importable.pth` from `synthesis_export` is accepted too.

Expected terminal milestones are:

```text
[MODEL] verified s08_p1202M_tau40 architecture
[EQUIVALENCE PASS] ... edge/structured/random cases
[DONE] .../regular_network.png
[DONE] .../compacted_network.png
```

The default proof uses all-zero/all-one, one-hot/one-cold, alternating,
valid thermometer-style, and 32 seeded random Boolean cases. It compares the
regular and compact circuits using exact integer GroupSum counts, then compares
the extracted regular circuit with the checkpoint's own discrete TorchLogix
forward pass. Any mismatch stops the command before outputs are published.
Increase ordinary random coverage without retraining:

```bash
python -m visualizingLGN.visualize_lgn /path/to/clean/s08/final_model.pth \
  --require-s08 \
  --random-cases 256 \
  --output-dir results/LGN_visualization/s08_clean_256cases
```

## Dependencies and tests

Use the same Python environment that trained the LGN; it already needs
PyTorch, TorchLogix, NumPy, and Matplotlib. No SLURM allocation or GPU is
required. On a login node, keep the default 32 random cases. Run the small
synthetic unit tests with:

```bash
python -m pytest visualizingLGN/test_circuit.py -q
```

The unit tests exhaust all inputs of a deliberately mixed small circuit and
also exercise every TorchLogix truth-table ID. The checkpoint command is the
actual s08 acceptance test; preserve `circuit_summary.json` with the figures so
the equivalence result and source hash remain auditable.
