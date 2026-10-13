# SmartPixels MAIA signal-versus-BIB LNN study

This branch is intentionally scoped to the Muon Collider dataset published as
[Zenodo record 22016562](https://doi.org/10.5281/zenodo.22016562). It does not
train the former CMS signed-\(p_T\) task.

## Classification task

The binary target is defined by dataset provenance:

| Class | Index | Source files |
|---|---:|---|
| Beam-induced background (BIB) | 0 | `bib_mp`, `bib_mm` |
| PV/collision signal muon | 1 | `signal` |

All BIB rows are retained. Before splitting, the signal collection is
deterministically undersampled to match the combined number of `bib_mp` and
`bib_mm` rows. Each source is then split 80%/10%/10% into train, validation,
and test sets with seed 42.

The class label is determined only by source provenance. Detector position is
a deployed input, as in the MAIA model definitions; direction, momentum, and
timing truth are not model inputs. The collaborator-provided \([-0.5,15]\) ns
cut on `adjusted_hit_time` is applied only as a second, post-training evaluation
selection. Every model is reported both before and after that cut.

## Inputs and models

The 5,460 reconstruction columns are reshaped to `(20,13,21)` =
`(time,y,x)`.

- Dense/Model 2: integrate charge over time, form the 21-bin x profile and
  13-bin y profile, then concatenate `n_module`, `x_local`, and `y_local` (37
  inputs total).
- Convolutional/Model 3: integrate charge over time and convolve the full
  `13 x 21` spatial cluster. Thermometer-encoded `n_module`, `x_local`, and
  `y_local` are concatenated after the convolutional trunk.

Both families use 100 fixed empirical-quantile thresholds derived exclusively
from training data, `lut_rank=2`, residual logic initialization, and a binary
`GroupSum`. The final logic width is even in every model. Five increasing
architectures per family are trained at `tau = 10, 20, 40`, giving 30 jobs.

The campaign requires at least 20,000 optimizer steps **and** five hours of
training per model. The best checkpoint is selected by discrete validation
balanced accuracy. Final output additionally reports signal efficiency, BIB
acceptance/rejection, ROC AUC, average precision, and BIB rejection at
validation-derived 90%, 95%, and 99% signal-efficiency operating points.

The reported `logic_node_proxy` is an architecture count, not a synthesized
LUT+FF result.

## Launch on Della

```bash
cd ~/SmartPixels
git fetch origin
git switch maia-studies 2>/dev/null || git switch --track origin/maia-studies
git pull --ff-only origin maia-studies

module load anaconda3/2025.12
conda activate smart_pixels

export MAIA_DATA_ROOT=/scratch/gpfs/IOJALVO/mb7126/SmartPixels/22016562
bash maiaStudies/submit_pilot.sh
```

The one-time projected cache defaults to
`$MAIA_DATA_ROOT/derived/maia_lnn_spatial_v2`. This is deliberately distinct
from the invalid v1 cache. A completed v2 cache is detected and safely
reused. Override `MAIA_CACHE_ROOT` or `MAIA_RESULT_ROOT` before submission if
needed. The timing selection defaults to `adjusted_hit_time`. To study one of
the smeared timing definitions instead, set `MAIA_TIMING_COLUMN` and use a
distinct `MAIA_CACHE_ROOT` before launching.

The pilot prepares and audits the cache, then trains/evaluates one middle-size
Model 2 and one middle-size Model 3. It writes `pilot_gate_complete.json` only
if both learn above chance. After that marker exists, launch the 30-model run:

```bash
bash maiaStudies/submit.sh
```

Monitor with:

```bash
squeue -u "$USER"
bash maiaStudies/status.sh
```

Expected final products:

- `performance_summary.tsv`, `.csv`, and `.json`: one full-test and one
  timing-cut row per model;
- `model_comparison_wide.tsv`: one row per model with all performance and
  architecture fields;
- `best_models.tsv` and `SUMMARY.md`: ranked configurations and a concise
  digest;
- `summary_plots/`: 18 PNG/PDF comparison figures, with Model 2 in blue,
  Model 3 in orange, and marker shape encoding `tau`;
- `maia_lnn_screening.png` and `.pdf`: compatibility copies of the campaign
  dashboard;
- one `evaluation_complete.json`, `roc_full.csv`, and `roc_timing_cut.csv` per
  model;
- `best_model.pth` state-dict checkpoints with embedded architecture and fixed
  thresholds.

The complete plot package can be regenerated at any time, without retraining:

```bash
ROOT="$PWD/results/SLURM/2026_10_13_maia_signal_vs_bib_corrected_inputs_5h_20k"
bash maiaStudies/plot_summary.sh "$ROOT"
```

See [maiaStudies/README.md](maiaStudies/README.md) for recovery and metric
details.
