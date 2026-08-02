# S08 similarity-visualization strategy

Status: **design note only**. This deliberately adds no comparison script, no
radiation workflow, no training, and no changes to the existing visualizer.
The current scope remains the clean, discrete `s08_p1202M_tau40` LGN.

## Required output contract: exactly two PNGs

The future similarity mode must mirror the current visualizer and publish only
these two primary plots:

1. `regular_similarity_network.png`: all 95,700 regular discrete gates, with
   unchanged gates muted and changed gates highlighted;
2. `compacted_similarity_network.png`: only physical/reachable logic after
   compaction, projected back onto the same stable gate coordinates so visual
   regions correspond between models.

Any transition matrix, exact-count summary, or legend belongs as an inset in
one of those two PNGs and as machine-readable data, not as a third plot.

## The recommended visual mode: two fixed-coordinate LGN similarity atlases

Use one fixed two-dimensional pixel grid per LogicDense layer. One pixel is one
discrete gate. The gate at `(layer, neuron index)` must occupy the same pixel in
every future image, regardless of checkpoint or radiation level.

For s08, the seven layer panels would contain:

```text
L1       L2       L3       L4       L5       L6       L7
17,400   17,400   17,400   11,600   11,600   11,600    8,700 gates
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│░░░░░│  │░░░░░│  │░░▓░░│  │░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│
│░░▓▓░│  │░░░░░│  │░▓▓░░│  │░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│
│░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│  │░░░░░│
└─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
 unchanged = quiet gray                     changed = highlighted
```

Each plot is a circuit *atlas*, not a conventional neural-network diagram. It is
intended to answer the visual question, “is there a stable core with a small,
localized set of changing gates?” A 95,700-node force-directed drawing cannot
answer that reliably because edges obscure the gates and a new layout can move
unchanged nodes.

The static PNG can be rendered directly from an integer/RGBA array with
[Matplotlib `imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html),
using `interpolation="nearest"` so a colored cell is never visually blended
with its neighbors. At this scale, Matplotlib and NumPy are sufficient and are
already repository dependencies. [Datashader](https://datashader.org/user_guide/Grids.html)
is a reasonable later option for interactive re-rasterization across many
checkpoints, but it is unnecessary for the first static atlas.

## The visual grammar

Do not assign a unique color to every possible source-to-target LUT transition;
there are 256 transitions and the result would be illegible. Separate “where”
from “what changed”:

1. **Atlas pixels show where and how much a gate changed.**

   - unchanged selected LUT: light neutral gray;
   - one truth-table bit changed: yellow;
   - two bits changed: orange;
   - three bits changed: red;
   - all four bits changed: purple;
   - changed but unreachable after compaction: desaturated or hatched;
   - wiring mismatch: black, treated as an alignment failure rather than a LUT
     change.

2. **An inset 16-by-16 transition matrix shows what changed.** Rows are baseline LUT
   IDs, columns are comparison LUT IDs, and each cell contains an exact count.
   The whole-model matrix is included inside the relevant regular or compact
   PNG; layer-specific counts remain in machine-readable output.

3. **A compact summary bar reports exact fractions.** For every layer, report
   unchanged gates, changed-and-reachable gates, changed-but-dead gates, and
   gates whose wiring is not aligned.

The existing 16 gate colors remain appropriate for a single checkpoint. The
similarity atlas uses the change-distance colors above because similarity must
be visually dominant; the transition matrix retains the exact LUT identities.

## Alignment is a hard prerequisite

A dense LGN has no natural geometric region. Neuron index is only a meaningful
cross-checkpoint identity when the underlying wiring is the same. Before any
future comparison, require all of the following:

- identical s08 architecture and layer widths;
- bit-identical `connections.indices` for every LogicDense layer;
- identical input-bit ordering and three GroupSum class ranges;
- an explicitly recorded decision about whether binarization thresholds are
  frozen or allowed to change.

The safest future experimental design is to begin from one clean s08 checkpoint
and preserve its fixed connection arrays for every condition. The master
`regular_netlist.npz` should provide the connection hashes. If independently
trained checkpoints have different wiring, a same-pixel comparison must stop;
silently comparing neuron index 500 in two unrelated wirings would produce a
convincing but meaningless image.

Threshold changes should be visualized in a separate fourteen-by-one-hundred
“threshold bank” strip. Mixing threshold drift and LUT changes into one color
would make it impossible to tell whether radiation adaptation occurred at the
input discretization or inside the learned logic.

## Fixed ordering inside each layer

Raw neuron order is reproducible but visually arbitrary. A useful atlas should
place structurally related gates near one another without changing positions
between checkpoints.

Derive the ordering once from the clean s08 baseline, save it, and reuse it
verbatim. A good deterministic key is:

1. downstream class-influence bitmask from reverse reachability;
2. compacted reachability status;
3. ordered pair of upstream connection signatures (`A` and `B` must remain
   distinct because some truth tables are asymmetric);
4. original neuron index as the final tie-breaker.

Place that one-dimensional ordering into a near-square grid. The ordering is a
display transform only: all reports and saved data retain the original neuron
indices. It may make related gates form visual neighborhoods, but a colored
patch is not automatically a physically contiguous FPGA floorplan region.

## How compaction participates

Always align and compare the **regular discrete networks first**. Do not compare
compact node indices directly: constant propagation and identity elimination
can renumber or remove nodes differently in two checkpoints. During future
compaction, retain provenance from every physical node back to its original
`(layer, neuron index)` gate.

The second required PNG uses the same seven fixed grids as the regular PNG, but
projects the compact physical nodes back to those provenance coordinates:

- physical in both models with the same compacted LUT: neutral gray;
- physical in both with a changed compacted LUT: truth-table-distance color;
- physical only in the baseline: blue removal marker;
- physical only in the comparison: magenta addition marker;
- removed in both by compaction: blank;
- changed and connected to other changed gates: candidate sensitive component;
- stable gate on the boundary of a changed component: candidate interface net.

An inverter folded into a downstream LUT is therefore shown at the downstream
gate's original coordinate using its canonical compacted truth table. A
passthrough or constant that creates no physical node is blank. This preserves
the meaning of “regular plot” versus “compact plot” while keeping locations
stable enough for visual comparison.

This makes the future reconfiguration question concrete. A promising
replaceable region is not merely a set of changed pixels; it is a small
reachable connected component with a small boundary to the stable core.

## More than two conditions

For several future conditions, keep the baseline coordinates fixed. The same
two PNGs can contain aligned rows or columns for:

- **small multiples:** the same seven panels repeated for each condition;
- **consensus atlas:** pixel intensity is the fraction of checkpoints retaining
  the baseline LUT;
- **first-change atlas:** pixel color encodes the first ordered condition where
  the LUT changes;
- **change-count atlas:** pixel value counts how often the selected LUT changes
  along the ordered sequence.

These are modes within the regular and compacted similarity PNGs, not extra
plot files. They distinguish a monotonic response from random training variation.
They should be accompanied by replicate training seeds eventually; a single
pair of checkpoints cannot establish that a change is radiation-specific.

## Recommended tool boundary

| Tool | Recommended role | Not the primary role |
|---|---|---|
| NumPy + Matplotlib | The two exact fixed-grid PNG atlases, inset transition matrices, summary bars | Interactive graph exploration |
| NetworkX | Reachability, connected components, boundary/cut calculations | Drawing all 95,700 gates |
| Datashader | Optional interactive rasterization when the number of checkpoints becomes large | Initial static s08 plot |
| Graphviz `sfdp` | A zoomed sensitive component with a fixed saved layout | Cross-model full-network comparison |
| Yosys `show` | A selected post-RTL region | Whole-s08 comparison image |
| Yosys equivalence passes | Later formal validation of a proposed hybrid/reconfigured RTL model | Discovering visually similar regions |

Graphviz documents `sfdp` as its scalable multilevel force-directed layout for
large graphs, but even a scalable layout does not provide the stable
cross-checkpoint coordinates required here:
https://graphviz.org/docs/layouts/sfdp/. Yosys can later prepare and assert RTL
equivalence with
[`equiv_make`](https://yosyshq.readthedocs.io/projects/yosys/en/0.35/cmd/equiv_make.html)
and
[`equiv_status`](https://yosyshq.readthedocs.io/projects/yosys/en/v0.55/cmd/equiv_status.html).

## What would justify the future reconfigurable-LUT hypothesis

The atlas is exploratory evidence, not proof. The hypothesis becomes credible
only if later work finds all of the following:

1. most reachable gates remain identical across conditions and replicate seeds;
2. changes concentrate in reproducible connected components rather than being
   scattered uniformly;
3. those components have a small stable input/output boundary;
4. replacing only those components in the clean model reproduces the target
   model's discrete outputs on ordinary and edge cases;
5. the hybrid RTL is formally equivalent to the full target RTL before any
   hardware claim is made.

No percentage threshold should be chosen before observing the real models. The
first future implementation should therefore generate the two atlases and their
raw counts without labelling any region “reconfigurable.”

## Explicitly out of scope now

- training or fine-tuning any model;
- loading or comparing radiation checkpoints;
- changing the clean s08 visualizer or its compaction/equivalence behavior;
- choosing radiation levels, damage models, or retraining protocol;
- generating an RTL patch or claiming an FPGA-reconfigurable region.

The next implementation step, when explicitly requested, is only a read-only
two-checkpoint, two-PNG prototype guarded by strict s08 and connection-hash
checks.
