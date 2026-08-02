"""Fixed-coordinate, one-pixel-per-gate atlases for one discrete s08 LGN.

These images establish a reusable baseline layout for later comparisons.  This
module does not load or compare radiation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .circuit import (
    BINARY_KIND,
    CONST_ONE,
    CONST_ZERO,
    GATE_COLORS,
    GATE_NAMES,
    INPUT_OFFSET,
    NOT_KIND,
    CompactCircuit,
    DenseCircuit,
    _input_refs,
    _permute_input,
    _truth,
    build_compact_circuit,
)


PADDING_PIXEL = -2
REMOVED_PIXEL = -1


@dataclass(frozen=True)
class ProvenanceCompact:
    """A compact circuit plus the regular gate that created each physical node."""

    circuit: CompactCircuit
    node_origins: np.ndarray

    def __post_init__(self) -> None:
        origins = np.asarray(self.node_origins, dtype=np.int64)
        if origins.shape != (self.circuit.physical_gate_count, 2):
            raise ValueError(
                "Expected one (layer, neuron) origin per compact physical gate."
            )
        object.__setattr__(self, "node_origins", origins)


@dataclass(frozen=True)
class AtlasLayout:
    """Baseline-derived display order reused by both required PNGs."""

    orders: tuple[np.ndarray, ...]
    shapes: tuple[tuple[int, int], ...]
    class_influence_masks: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.orders) == len(self.shapes) == len(self.class_influence_masks)
        ):
            raise ValueError("Atlas layout fields must describe the same layers.")


def build_compact_with_provenance(circuit: DenseCircuit) -> ProvenanceCompact:
    """Repeat the audited compaction while retaining raw-gate provenance."""
    previous_refs = _input_refs(circuit.input_width)
    kinds: list[int] = []
    lut_ids: list[int] = []
    children: list[tuple[int, int]] = []
    node_layers: list[int] = []
    origins: list[tuple[int, int]] = []

    def add_node(
        kind: int,
        lut_id: int,
        left: int,
        right: int,
        layer_index: int,
        neuron_index: int,
    ) -> int:
        kinds.append(kind)
        lut_ids.append(lut_id)
        children.append((int(left), int(right)))
        node_layers.append(layer_index)
        origins.append((layer_index, neuron_index))
        return len(kinds) - 1

    def invert(ref: int, layer_index: int, neuron_index: int) -> int:
        if ref == CONST_ZERO:
            return CONST_ONE
        if ref == CONST_ONE:
            return CONST_ZERO
        if ref >= 0 and kinds[ref] == NOT_KIND:
            return children[ref][0]
        return add_node(
            NOT_KIND,
            12,
            ref,
            CONST_ZERO,
            layer_index,
            neuron_index,
        )

    def unary(
        ref: int,
        output_at_zero: int,
        output_at_one: int,
        layer_index: int,
        neuron_index: int,
    ) -> int:
        if output_at_zero == output_at_one:
            return CONST_ONE if output_at_zero else CONST_ZERO
        if output_at_zero == 0 and output_at_one == 1:
            return ref
        return invert(ref, layer_index, neuron_index)

    def apply_lut(
        lut_id: int,
        left: int,
        right: int,
        layer_index: int,
        neuron_index: int,
    ) -> int:
        if left >= 0 and kinds[left] == NOT_KIND:
            left = children[left][0]
            lut_id = _permute_input(lut_id, 0)
        if right >= 0 and kinds[right] == NOT_KIND:
            right = children[right][0]
            lut_id = _permute_input(lut_id, 1)

        if left in (CONST_ZERO, CONST_ONE) and right in (
            CONST_ZERO,
            CONST_ONE,
        ):
            value = _truth(lut_id, left == CONST_ONE, right == CONST_ONE)
            return CONST_ONE if value else CONST_ZERO

        if left == right:
            return unary(
                left,
                _truth(lut_id, 0, 0),
                _truth(lut_id, 1, 1),
                layer_index,
                neuron_index,
            )
        if left in (CONST_ZERO, CONST_ONE):
            a = int(left == CONST_ONE)
            return unary(
                right,
                _truth(lut_id, a, 0),
                _truth(lut_id, a, 1),
                layer_index,
                neuron_index,
            )
        if right in (CONST_ZERO, CONST_ONE):
            b = int(right == CONST_ONE)
            return unary(
                left,
                _truth(lut_id, 0, b),
                _truth(lut_id, 1, b),
                layer_index,
                neuron_index,
            )

        independent_of_right = all(
            _truth(lut_id, a, 0) == _truth(lut_id, a, 1) for a in (0, 1)
        )
        if independent_of_right:
            return unary(
                left,
                _truth(lut_id, 0, 0),
                _truth(lut_id, 1, 0),
                layer_index,
                neuron_index,
            )
        independent_of_left = all(
            _truth(lut_id, 0, b) == _truth(lut_id, 1, b) for b in (0, 1)
        )
        if independent_of_left:
            return unary(
                right,
                _truth(lut_id, 0, 0),
                _truth(lut_id, 0, 1),
                layer_index,
                neuron_index,
            )
        return add_node(
            BINARY_KIND,
            lut_id,
            left,
            right,
            layer_index,
            neuron_index,
        )

    for layer_index, layer in enumerate(circuit.layers):
        current_refs = np.empty(layer.width, dtype=np.int64)
        for neuron_index, lut_id in enumerate(layer.lut_ids):
            left = int(previous_refs[layer.connections[0, neuron_index]])
            right = int(previous_refs[layer.connections[1, neuron_index]])
            current_refs[neuron_index] = apply_lut(
                int(lut_id),
                left,
                right,
                layer_index,
                neuron_index,
            )
        previous_refs = current_refs

    stack = [int(ref) for ref in previous_refs if ref >= 0]
    reachable: set[int] = set()
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(child for child in children[node] if child >= 0)

    kept = sorted(reachable)
    remap = {old: new for new, old in enumerate(kept)}

    def remap_ref(ref: int) -> int:
        return remap[ref] if ref >= 0 else ref

    kept_children = [children[node] for node in kept]
    compact = CompactCircuit(
        input_width=circuit.input_width,
        num_classes=circuit.num_classes,
        tau=circuit.tau,
        beta=circuit.beta,
        node_kinds=np.asarray([kinds[node] for node in kept], dtype=np.uint8),
        node_lut_ids=np.asarray(
            [lut_ids[node] for node in kept], dtype=np.uint8
        ),
        child_a=np.asarray(
            [remap_ref(pair[0]) for pair in kept_children], dtype=np.int64
        ),
        child_b=np.asarray(
            [remap_ref(pair[1]) for pair in kept_children], dtype=np.int64
        ),
        node_layers=np.asarray(
            [node_layers[node] for node in kept], dtype=np.int32
        ),
        outputs=np.asarray(
            [remap_ref(int(ref)) for ref in previous_refs], dtype=np.int64
        ),
        raw_layer_widths=circuit.layer_widths,
    )
    return ProvenanceCompact(
        circuit=compact,
        node_origins=np.asarray([origins[node] for node in kept], dtype=np.int64),
    )


def assert_matches_reference_compactor(
    circuit: DenseCircuit, provenance: ProvenanceCompact
) -> None:
    """Prove the provenance extension did not alter the original compactor."""
    reference = build_compact_circuit(circuit)
    actual = provenance.circuit
    fields = (
        "node_kinds",
        "node_lut_ids",
        "child_a",
        "child_b",
        "node_layers",
        "outputs",
    )
    for field in fields:
        if not np.array_equal(getattr(reference, field), getattr(actual, field)):
            raise AssertionError(
                f"Similarity-atlas provenance changed compact field {field}."
            )


def _class_influence_masks(circuit: DenseCircuit) -> tuple[np.ndarray, ...]:
    """Return a three-bit mask of GroupSum classes reachable from each gate."""
    influences: list[np.ndarray | None] = [None] * len(circuit.layers)
    per_class = circuit.final_width // circuit.num_classes
    influences[-1] = np.repeat(
        (1 << np.arange(circuit.num_classes, dtype=np.uint8)), per_class
    )
    for layer_index in range(len(circuit.layers) - 2, -1, -1):
        current = np.zeros(circuit.layers[layer_index].width, dtype=np.uint8)
        downstream = circuit.layers[layer_index + 1]
        downstream_masks = influences[layer_index + 1]
        assert downstream_masks is not None
        np.bitwise_or.at(current, downstream.connections[0], downstream_masks)
        np.bitwise_or.at(current, downstream.connections[1], downstream_masks)
        influences[layer_index] = current
    return tuple(mask for mask in influences if mask is not None)


def build_atlas_layout(
    circuit: DenseCircuit, provenance: ProvenanceCompact
) -> AtlasLayout:
    """Derive one deterministic baseline order for both atlas images."""
    influences = _class_influence_masks(circuit)
    physical = [np.zeros(layer.width, dtype=bool) for layer in circuit.layers]
    for layer_index, neuron_index in provenance.node_origins:
        physical[int(layer_index)][int(neuron_index)] = True

    orders = []
    shapes = []
    for layer_index, layer in enumerate(circuit.layers):
        original = np.arange(layer.width, dtype=np.int64)
        # np.lexsort uses the final key as primary. The order groups gates by
        # downstream class influence, then physical reachability and A/B wiring.
        order = np.lexsort(
            (
                original,
                layer.connections[1],
                layer.connections[0],
                (~physical[layer_index]).astype(np.uint8),
                influences[layer_index],
            )
        )
        columns = int(np.ceil(np.sqrt(layer.width)))
        rows = int(np.ceil(layer.width / columns))
        orders.append(order.astype(np.int64, copy=False))
        shapes.append((rows, columns))
    return AtlasLayout(tuple(orders), tuple(shapes), influences)


def regular_atlas_grids(
    circuit: DenseCircuit, layout: AtlasLayout
) -> tuple[np.ndarray, ...]:
    grids = []
    for layer, order, shape in zip(circuit.layers, layout.orders, layout.shapes):
        flat = np.full(shape[0] * shape[1], PADDING_PIXEL, dtype=np.int16)
        flat[: layer.width] = layer.lut_ids[order]
        grids.append(flat.reshape(shape))
    return tuple(grids)


def compact_atlas_grids(
    circuit: DenseCircuit,
    provenance: ProvenanceCompact,
    layout: AtlasLayout,
) -> tuple[np.ndarray, ...]:
    compact_ids = [
        np.full(layer.width, REMOVED_PIXEL, dtype=np.int16)
        for layer in circuit.layers
    ]
    for node, (layer_index, neuron_index) in enumerate(provenance.node_origins):
        target = compact_ids[int(layer_index)]
        if target[int(neuron_index)] != REMOVED_PIXEL:
            raise AssertionError("Two compact nodes map to the same regular gate.")
        target[int(neuron_index)] = int(provenance.circuit.node_lut_ids[node])

    grids = []
    for layer, ids, order, shape in zip(
        circuit.layers, compact_ids, layout.orders, layout.shapes
    ):
        flat = np.full(shape[0] * shape[1], PADDING_PIXEL, dtype=np.int16)
        flat[: layer.width] = ids[order]
        grids.append(flat.reshape(shape))
    return tuple(grids)


def _atlas_colormap() -> ListedColormap:
    # Encoded index 0 is padding, 1 is a removed regular gate, and 2..17 are
    # the sixteen compact/regular truth-table IDs.
    return ListedColormap(("#ffffff", "#e0e0e0", *GATE_COLORS))


def _draw_grid(axis, grid: np.ndarray, title: str, subtitle: str) -> None:
    encoded = grid + 2
    axis.imshow(
        encoded,
        cmap=_atlas_colormap(),
        vmin=0,
        vmax=17,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    rows, columns = grid.shape
    axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            columns,
            rows,
            fill=False,
            edgecolor="#263238",
            linewidth=1.0,
        )
    )
    axis.set_title(title, fontsize=12, weight="bold", pad=8)
    axis.text(
        0.5,
        -0.055,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#455a64",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _stacked_layer_bars(
    axis,
    counts: list[np.ndarray],
    widths: Iterable[int],
    *,
    compact: bool,
) -> None:
    widths_array = np.asarray(tuple(widths), dtype=np.int64)
    y = np.arange(len(counts))
    if compact:
        axis.barh(y, widths_array, color="#eceff1", edgecolor="none")
    left = np.zeros(len(counts), dtype=np.int64)
    for lut_id in range(16):
        values = np.asarray([int(layer[lut_id]) for layer in counts])
        axis.barh(y, values, left=left, color=GATE_COLORS[lut_id], edgecolor="none")
        left += values
    axis.set_yticks(y, [f"L{index + 1}" for index in range(len(counts))])
    axis.invert_yaxis()
    axis.set_xlabel("Gate count")
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    for row, (kept, raw) in enumerate(zip(left, widths_array)):
        label = f"{kept:,}" if not compact else f"{kept:,}/{raw:,} physical"
        axis.text(raw + max(widths_array) * 0.008, row, label, va="center", fontsize=8)
    axis.set_xlim(0, max(widths_array) * 1.13)


def _gate_legend(*, include_removed: bool) -> list[Patch]:
    handles = [
        Patch(facecolor=color, label=f"{lut_id}: {name}")
        for lut_id, (name, color) in enumerate(zip(GATE_NAMES, GATE_COLORS))
    ]
    if include_removed:
        handles.append(Patch(facecolor="#e0e0e0", label="removed/absorbed"))
    return handles


def plot_regular_similarity_atlas(
    circuit: DenseCircuit, layout: AtlasLayout, output: Path
) -> None:
    grids = regular_atlas_grids(circuit, layout)
    fig = plt.figure(figsize=(23, 9.5))
    spec = fig.add_gridspec(2, len(grids), height_ratios=(3.2, 1.2), hspace=0.35)
    for index, (layer, grid) in enumerate(zip(circuit.layers, grids)):
        axis = fig.add_subplot(spec[0, index])
        _draw_grid(
            axis,
            grid,
            f"LogicDense {index + 1}",
            f"{layer.width:,} pixels = {layer.width:,} regular gates",
        )
    bars = fig.add_subplot(spec[1, :])
    counts = [np.bincount(layer.lut_ids, minlength=16) for layer in circuit.layers]
    _stacked_layer_bars(bars, counts, circuit.layer_widths, compact=False)
    bars.set_title("Exact selected truth-table counts")
    fig.suptitle(
        "S08 regular similarity atlas — one fixed pixel per discrete gate",
        fontsize=19,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.947,
        (
            "Baseline structural order: downstream class influence, compact "
            "reachability, ordered A/B connections, original gate index"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#455a64",
    )
    fig.legend(
        handles=_gate_legend(include_removed=False),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.003),
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_compacted_similarity_atlas(
    circuit: DenseCircuit,
    provenance: ProvenanceCompact,
    layout: AtlasLayout,
    output: Path,
) -> None:
    grids = compact_atlas_grids(circuit, provenance, layout)
    fig = plt.figure(figsize=(23, 9.5))
    spec = fig.add_gridspec(2, len(grids), height_ratios=(3.2, 1.2), hspace=0.35)
    for index, (layer, grid) in enumerate(zip(circuit.layers, grids)):
        physical = int(np.count_nonzero(grid >= 0))
        axis = fig.add_subplot(spec[0, index])
        _draw_grid(
            axis,
            grid,
            f"Source LogicDense {index + 1}",
            f"{physical:,} physical; {layer.width - physical:,} gray removed",
        )
    bars = fig.add_subplot(spec[1, :])
    compact = provenance.circuit
    counts = [
        np.bincount(
            compact.node_lut_ids[compact.node_layers == layer_index], minlength=16
        )
        for layer_index in range(len(circuit.layers))
    ]
    _stacked_layer_bars(bars, counts, circuit.layer_widths, compact=True)
    reduction = 100 * (1 - compact.physical_gate_count / circuit.raw_gate_count)
    bars.set_title(
        f"{compact.physical_gate_count:,} reachable physical gates; "
        f"{reduction:.1f}% removed or absorbed"
    )
    fig.suptitle(
        "S08 compacted similarity atlas — physical logic in regular-gate coordinates",
        fontsize=19,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.947,
        (
            "Same coordinates as the regular atlas; gray pixels were removed "
            "or absorbed, colors are canonical compacted truth tables"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#455a64",
    )
    fig.legend(
        handles=_gate_legend(include_removed=True),
        loc="lower center",
        ncol=9,
        frameon=False,
        fontsize=8.1,
        bbox_to_anchor=(0.5, 0.003),
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_atlas_layout(path: Path, layout: AtlasLayout) -> None:
    arrays: dict[str, np.ndarray] = {
        "layer_count": np.asarray(len(layout.orders), dtype=np.int64)
    }
    for index, (order, shape, influence) in enumerate(
        zip(layout.orders, layout.shapes, layout.class_influence_masks)
    ):
        arrays[f"layer_{index}_order"] = order
        arrays[f"layer_{index}_shape"] = np.asarray(shape, dtype=np.int64)
        arrays[f"layer_{index}_class_influence_mask"] = influence
    np.savez_compressed(path, **arrays)
