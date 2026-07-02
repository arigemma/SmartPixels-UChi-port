# Random bit-flip robustness study

This study measures robustness to random bit flips for the same three tau-40
model-2 y-profile LGNs used by the resource/precision study.

The fault model independently replaces `x` with `1-x` after
`FixedBinarization` and after every `LogicDense` layer. At discrete inference
this is an exact activation-bit flip. During relaxed training it mirrors the
soft Boolean activation around 0.5, preserving gradients. GroupSum is not
flipped because it contains integer class accumulators rather than Boolean
activations. This study does not model configuration-memory corruption.

## Cluster workflow

First complete the parent resource study, because its three clean
`final_model.pth` checkpoints are the controlled starting points:

```bash
sbatch lgnStudies/run.slurm
```

Then submit the three-model fault-aware array and its dependent combined plot:

```bash
bash lgnStudies/bitFlipStudy/submit.sh
```

Each model is fine-tuned for 500 steps with `p_train=0.001`, retaining the LGN
batch size 1024, learning rate 0.01, class-weighted cross-entropy, and
`--compile-model`. The standard and fault-aware models are evaluated with
paired fault masks at probabilities from zero through 0.1.

Every balanced-accuracy value is computed from aggregate class recalls over
the complete nominal test split; evaluation is not truncated and batches are
not averaged.

Per-model outputs include:

- `bit_flip_robustness.png` and `.pdf`: accuracy curves plus robustness gain;
- `bit_flip_results.csv`: probabilities, recalls, class counts, and seeds;
- `bit_flip_metadata.json`: exact fault and evaluation definitions;
- `bitflip_robust_model.pth`: fault-aware checkpoint.

The dependent CPU job writes
`bit_flip_robustness_all_models.png` and `.pdf` in the common result root.
