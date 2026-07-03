# Model-2 LGN resource/precision study

This study produces the PI-requested per-model trajectory: total estimated
hardware resource on the x-axis and balanced accuracy on the y-axis, with one
point for each recorded training stage.

Submit all three tau-40, y-profile models from the repository root:

```bash
sbatch lgnStudies/run.slurm
```

To submit this resource study and the dependent bit-flip study together, use:

```bash
bash lgnStudies/submit_training_studies.sh
```

The array runs `s06_p760M_tau40`, `s08_p1202M_tau40`, and
`s09_p1460M_tau40`. The training settings retain the LGN configuration used by
the prior sweep: batch size 1024, learning rate 0.01, class-weighted
cross-entropy, and `--compile-model`. A checkpoint is complete only after both
20,000 optimizer steps and five accumulated hours in the optimizer loop.
Full-test evaluation and synthesis-export time do not count toward those five
hours. The job continues until both requirements are satisfied.

Each output directory is updated after every stage and contains:

- `resource_vs_balanced_accuracy.png`: the primary total-resource plot;
- `resource_vs_balanced_accuracy_logic_only.png`: the architecture-discovery
  component without the fixed GroupSum offset;
- `resource_precision.csv`: exact metrics and the complete resource breakdown;
- `study_metadata.json`: class, dataset, and resource definitions;
- `synthesis_export/`: full-module staged checkpoints ready for transfer to
  `synthesis-SmartPixels`;
- `final_model.pth`: checkpoint published only after all training requirements;
- `training_complete.json`: proof that the minimum steps and five-hour
  requirement were both met.

Interrupted runs write `interrupted_model.pth` and
`training_incomplete.json`; they do not publish a valid new
`final_model.pth`. The dependent bit-flip campaign validates completion
metadata before loading any checkpoint.

Balanced accuracy is the mean of the three recalls from the discrete
(`model.eval()`) LGN. Counts are accumulated over the entire test split in
`config/10783560/v3.yml`; batches are never averaged and evaluation is never
truncated. The classes match training:
positive low-pT (`0 <= pT <= 0.2`), negative low-pT
(`-0.2 <= pT < 0`), and high-pT (`|pT| > 0.2`).

Resource is an analytical FPGA-LUT estimate, not a synthesis report. The
discrete dense network is simplified by constant propagation, passthrough
elimination, inverter folding, and dead-code elimination. The GroupSum
adder-tree estimate is fixed throughout training and is included in the
primary x-axis. Fixed binarization/input ADC and routing are intentionally
excluded.

For the publication plot using actual post-synthesis Vivado LUT counts, follow
[`SYNTHESIS.md`](SYNTHESIS.md). The immediate analytical plot is retained as a
cluster-side diagnostic.

The exact-count, inference-only bit-flip robustness study is documented in
[`bitFlipStudy/README.md`](bitFlipStudy/README.md).
