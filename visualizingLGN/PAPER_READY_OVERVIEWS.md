# Paper-ready non-atlas LGN overview figures

This is an additional presentation mode for the original aggregated
`regular_network` and `compacted_network` visualizations. It does **not** render
or modify either similarity atlas.

Instead of combining several small panels in two PNGs, it writes every overview
panel as a separate large-text figure. Both PNG and PDF are produced by default.

## Generate the figures

From the SmartPixels repository root, in the same environment used for LGN
training:

```bash
python -m visualizingLGN.visualize_lgn_paper \
  results/SLURM/2026_07_03_model2_resource_precision_5h_20k/s08_p1202M_tau40/final_model.pth \
  --output-dir results/LGN_visualization/s08_clean/paper_overviews
```

No training, radiation model, GPU, or SLURM allocation is used. The launcher
accepts sequential dense LGN checkpoints and runs the same
regular-versus-compacted equivalence checks before publishing figures. Use
`--require-s08` or `--require-s02` to verify one of the two named architectures
before any figures are written.

## Generate the selected input-study s02 figures

The selected LGN--QKeras input-bit-flip study uses the clean
`s02_p163M_tau20` baseline, not s08. From the repository root:

```bash
ROOT=results/SLURM/2026_08_13_selected_input_bit_flip_1to10pct_eval60_linear
S02="$ROOT/lgn/baseline_clean/s02_p163M_tau20/training/final_model.pth"

python -m visualizingLGN.visualize_lgn_paper \
  "$S02" \
  --require-s02 \
  --output-dir results/LGN_visualization/s02_input_study_clean/paper_overviews \
  --font-scale 1.35 \
  --dpi 600
```

The strict check expects seven `LogicDense` widths
`6300/6300/6300/4200/4200/4200/3150`, totaling 34,650 regular logic gates,
with 1,400 threshold inputs, three output classes, and tau 20.

## Output files

```text
paper_overviews/
├── regular/
│   ├── regular_pipeline.{png,pdf}
│   ├── regular_gate_counts.{png,pdf}
│   ├── regular_connectivity.{png,pdf}
│   └── regular_lut_legend.{png,pdf}
├── compacted/
│   ├── compacted_pipeline.{png,pdf}
│   ├── compacted_gate_counts.{png,pdf}
│   ├── compacted_reachable_connections.{png,pdf}
│   └── compacted_lut_legend.{png,pdf}
└── paper_overview_manifest.json
```

PDF is recommended for paper assembly because its text remains sharp after
resizing. The default PNG resolution is 300 DPI.

For even larger text and 600-DPI PNGs:

```bash
python -m visualizingLGN.visualize_lgn_paper \
  results/SLURM/2026_07_03_model2_resource_precision_5h_20k/s08_p1202M_tau40/final_model.pth \
  --output-dir results/LGN_visualization/s08_clean/paper_overviews_large_text \
  --font-scale 1.35 \
  --dpi 600
```

To produce only PDF files:

```bash
python -m visualizingLGN.visualize_lgn_paper \
  results/SLURM/2026_07_03_model2_resource_precision_5h_20k/s08_p1202M_tau40/final_model.pth \
  --output-dir results/LGN_visualization/s08_clean/paper_overviews_pdf \
  --formats pdf
```

The original `regular_network.png`, `compacted_network.png`, atlas code, and
atlas outputs remain unchanged.
