# Required work in `synthesis-SmartPixels`

Create and push a separate branch named `wip-rtl-handoff`.  The objective is
to synthesize the staged Model-2 and Model-3 LGNs against the EFLX4K target,
not merely to repeat the existing Xilinx estimate.

## Input contract

For a feature with `T` ordered thresholds, the upstream encoder emits the
unsigned ordinal count

```text
code[f] = number of thresholds crossed by feature f, in [0, T].
```

The on-chip combinational decoder reconstructs the feature-major,
threshold-minor thermometer vector expected by the trained logic:

```text
therm[f, t] = (min(code[f], T) > t),  t = 0 ... T-1.
```

Invalid binary codes above `T` must saturate to `T`.  For learned thresholds,
the external encoder must use the values in `thresholds.pth`; the compact
code is a rank/bin index, not the ordinary fixed-point representation of the
original real number.  If raw physical values are to enter the FPGA instead,
that is a separate comparator/quantizer design and its cost must be counted.

## Output contract

The classifier produces three unsigned population counts.  Training `tau`
and `beta` are not hardware operations: synthesis uses `tau=1`, `beta=0`, and
classifies with argmax using the lowest class index on an exact tie.

## Implementation checklist

1. Create `wip-rtl-handoff` from the synthesis repository's current LGN
   conversion branch and ingest `models/rtl_handoff/model2/LGN/*` and
   `models/rtl_handoff/model3/LGN/*`.
2. Extend model discovery to read `model.json`; do not infer `n_bits`, feature
   count, GroupSum width, or topology from an alias.
3. Keep the existing pre-binarized da4ml path as a reference, using
   `best_model.importable.pth`.  Also accept `discrete_netlist.pth` so the
   handoff does not depend on the dynamically registered training classes.
4. Add the compact ordinal input port and decoder exactly as specified above.
   Preserve feature-major/threshold-minor ordering.  Unit-test codes 0, 1,
   `T`, and every invalid code for the non-power-of-two T100 case.
5. Map the rank-2 logic network, OR pooling, compact decoder, and three
   population-count trees to the EFLX primitive/library flow.  Do not treat a
   Xilinx LUT estimate as proof of fit.
6. Report separately: classifier core, ordinal decoder, GroupSum, and combined
   utilization; external input/output pins; timing; and routing/place result.
   The hard constraints to screen are 632 user inputs and 2520 dual 5-input
   LUT sites (with the target's supported LUT mixtures).
7. Validate bit-exact outputs at three boundaries: portable PyTorch versus
   discrete netlist, pre-binarized RTL versus PyTorch, and compact-port RTL
   versus pre-binarized RTL.  Include random vectors and threshold-boundary
   vectors.  Preserve the documented argmax tie rule.
8. Write one machine-readable synthesis report per model and a combined TSV
   keyed by the exact alias from `model.json`.  A model is a tape-out
   candidate only after combined target-specific utilization and timing pass.

The first selection should prioritize at least one dense and one
convolutional point with compact input margin and small GroupSum width.  Final
selection must use measured EFLX results plus test balanced accuracy; the
training campaign deliberately does not pre-declare a winner.
