# HGQ2 Model-2 Pareto comparison

This is a separate comparison campaign for Lindsey. It does not replace the
LGN/QKeras paper figure. It trains 21 HGQ2 classifiers matched one-for-one to
the 21 QKeras points currently used by the Model-2 balanced-accuracy/resource
plot:

- five ADC2A models with one 128-unit hidden layer;
- five y-profile models with one 128-unit hidden layer;
- five y-profile models with one 192-unit hidden layer;
- five y-profile models with 256- and 64-unit hidden layers;
- one y-profile model with 768-, 256-, and 128-unit hidden layers.

Within each five-point family, HGQ uses beta values `2e-5`, `1e-5`, `5e-6`,
`2e-6`, and `1e-6` to span stronger to weaker bit-cost pressure. The QKeras
W/A names stored in the manifest are comparison labels, not constraints on
the learned HGQ precisions. Training-time EBOPs are diagnostic only. The final
resource coordinate must come from synthesizing every selected `.keras` model
under the same clock, device, and flow used for LGN and QKeras.

## One-time Della installation

HGQ2 requires a newer standalone Keras 3. The older QKeras code explicitly
uses legacy `tf_keras`, so install the HGQ additions into the existing
`smart_pixels` environment:

```bash
module load anaconda3/2025.12
conda activate smart_pixels
python -m pip install -r hgqStudies/requirements.txt
```

The campaign deliberately selects Keras' PyTorch backend, while the QKeras
studies retain their existing `tf_keras` path.

## Launch all training and evaluation

From the repository root on `wip-mila-lgn-analysis`:

```bash
module load anaconda3/2025.12
conda activate smart_pixels
bash hgqStudies/submit.sh
```

That one command submits a 21-task GPU array (four concurrent jobs by default)
and a dependent strict summary job. Each task trains for 6000 optimizer steps
with batch size 1024, Adam at `1e-3`, a 1% validation split, balanced training
sampling, selects the checkpoint with the lowest validation cross-entropy (as
in the QKeras campaign), and then evaluates balanced accuracy on the complete
held-out test set. Override concurrency with, for example,
`HGQ_MAX_PARALLEL=2`.

The default result root is
`results/SLURM/YYYY_MM_DD_hgq2_model2_pareto_qkeras_matched_6k`. The summary
job succeeds only if all 21 standalone `best_model.keras` checkpoints and
their test metrics pass consistency checks. Print the result with:

```bash
ROOT=results/SLURM/YYYY_MM_DD_hgq2_model2_pareto_qkeras_matched_6k
column -s $'\t' -t < "$ROOT/summary/balanced_accuracy_summary.tsv"
```

The TSV and JSON contain the absolute checkpoint paths, balanced accuracy,
class recalls, learned EBOP proxy, architecture, beta, and corresponding
QKeras point. They are the transfer inventory for the later Wisc synthesis
step; the campaign does not duplicate checkpoints into a staging directory.

## Required WRAP-range calibration

HGQ's `WRAP` activation quantizers must be calibrated with `trace_minmax`
before discrete inference or synthesis. Calibration is non-destructive: it
keeps the original checkpoints, calibrates both the saved best and final
models, chooses the lower-cross-entropy model using validation data, and only
then evaluates the selected model on the test set.

First calibrate one pilot (`task 9`, the y-profile 128-unit model with the
weakest beta):

```bash
ROOT=results/SLURM/YYYY_MM_DD_hgq2_model2_pareto_qkeras_matched_6k
HGQ_CALIBRATION_TASKS=9 bash hgqStudies/submit_calibration.sh "$ROOT"
```

The calibration task fails if held-out balanced accuracy remains below 60%.
If the pilot passes, launch the full calibration array; the completed pilot is
detected and skipped:

```bash
bash hgqStudies/submit_calibration.sh "$ROOT"
```

Transfer and synthesize only the checkpoints listed in
`$ROOT/calibrated_summary/balanced_accuracy_summary.tsv`.
