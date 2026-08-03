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
accepts only the requested `s08_p1202M_tau40` architecture and runs the same
regular-versus-compacted equivalence checks before publishing figures.

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
