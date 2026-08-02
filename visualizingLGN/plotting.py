"""Readable overview plots for very large discrete LGN circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np

from .circuit import (
    BINARY_KIND,
    GATE_COLORS,
    GATE_NAMES,
    NOT_KIND,
    CompactCircuit,
    DenseCircuit,
)


def _gate_counts_regular(circuit: DenseCircuit) -> list[np.ndarray]:
    return [np.bincount(layer.lut_ids, minlength=16) for layer in circuit.layers]


def _gate_counts_compact(compact: CompactCircuit) -> list[np.ndarray]:
    result = []
    for layer_index in range(len(compact.raw_layer_widths)):
        result.append(
            np.bincount(
                compact.node_lut_ids[compact.node_layers == layer_index],
                minlength=16,
            )
        )
    return result


def _draw_stage(
    axis,
    x: float,
    counts: np.ndarray,
    *,
    title: str,
    subtitle: str,
    empty_label: str = "",
) -> None:
    width, bottom, height = 0.72, 0.18, 0.60
    total = int(np.sum(counts))
    box = FancyBboxPatch(
        (x - width / 2, bottom),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor="white",
        zorder=2,
    )
    axis.add_patch(box)
    if total:
        y = bottom
        for lut_id, count in enumerate(counts):
            if not count:
                continue
            segment = height * int(count) / total
            axis.add_patch(
                Rectangle(
                    (x - width / 2, y),
                    width,
                    segment,
                    facecolor=GATE_COLORS[lut_id],
                    edgecolor="none",
                    clip_path=box,
                    zorder=2.1,
                )
            )
            y += segment
    elif empty_label:
        axis.text(
            x,
            bottom + height / 2,
            empty_label,
            ha="center",
            va="center",
            fontsize=8,
            color="#616161",
        )
    axis.text(x, 0.86, title, ha="center", va="bottom", fontsize=10, weight="bold")
    axis.text(x, 0.12, subtitle, ha="center", va="top", fontsize=8.5)


def _draw_special_stage(axis, x: float, title: str, subtitle: str, color: str) -> None:
    box = FancyBboxPatch(
        (x - 0.40, 0.31),
        0.80,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.3,
        edgecolor="#263238",
        facecolor=color,
        zorder=2,
    )
    axis.add_patch(box)
    axis.text(x, 0.50, title, ha="center", va="center", fontsize=10, weight="bold")
    axis.text(x, 0.12, subtitle, ha="center", va="top", fontsize=8.5)


def _pipeline_axis(
    axis,
    counts: list[np.ndarray],
    subtitles: list[str],
    *,
    input_width: int,
    output_subtitle: str,
    compact: bool,
) -> None:
    number_layers = len(counts)
    for x in range(number_layers + 1):
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.41, 0.48),
                (x + 0.59, 0.48),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#546e7a",
                zorder=1,
            )
        )
    _draw_special_stage(
        axis,
        0,
        "Threshold bits",
        f"{input_width:,} Boolean inputs",
        "#e3f2fd",
    )
    for index, (layer_counts, subtitle) in enumerate(zip(counts, subtitles), start=1):
        _draw_stage(
            axis,
            index,
            layer_counts,
            title=f"LogicDense {index}",
            subtitle=subtitle,
            empty_label="no physical gates" if compact else "",
        )
    _draw_special_stage(
        axis,
        number_layers + 1,
        "GroupSum",
        output_subtitle,
        "#fff3e0",
    )
    axis.set_xlim(-0.55, number_layers + 1.55)
    axis.set_ylim(0.03, 0.98)
    axis.axis("off")


def _used_gate_legend(counts: list[np.ndarray]) -> list[Patch]:
    total = np.sum(np.stack(counts), axis=0)
    return [
        Patch(facecolor=GATE_COLORS[index], label=f"{index}: {GATE_NAMES[index]}")
        for index in range(16)
        if total[index]
    ]


def plot_regular_circuit(
    circuit: DenseCircuit, summary: dict[str, Any], output: Path
) -> None:
    counts = _gate_counts_regular(circuit)
    fig = plt.figure(figsize=(21, 10.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1.0), hspace=0.24, wspace=0.28)
    pipeline = fig.add_subplot(grid[0, :])
    widths = circuit.layer_widths
    _pipeline_axis(
        pipeline,
        counts,
        [f"{width:,} learned gates" for width in widths],
        input_width=circuit.input_width,
        output_subtitle=(
            f"{circuit.final_width // circuit.num_classes:,} bits/class "
            f"→ {circuit.num_classes} scores"
        ),
        compact=False,
    )
    pipeline.set_title(
        "Discrete dense LGN — regular (uncompacted) circuit",
        fontsize=18,
        weight="bold",
        pad=38,
    )
    pipeline.text(
        0.5,
        1.01,
        (
            "Each colored column contains every selected rank-2 truth table; "
            "height is its fraction of that layer."
        ),
        transform=pipeline.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#455a64",
    )

    width_axis = fig.add_subplot(grid[1, 0])
    positions = np.arange(len(widths))
    width_axis.barh(positions, widths, color="#607d8b")
    width_axis.set_yticks(positions, [f"LogicDense {i + 1}" for i in positions])
    width_axis.invert_yaxis()
    width_axis.set_xlabel("Learned gates")
    width_axis.set_title(f"All {circuit.raw_gate_count:,} gates retained")
    width_axis.grid(axis="x", alpha=0.22)
    for y, width in zip(positions, widths):
        width_axis.text(width, y, f"  {width:,}", va="center", fontsize=9)

    table_axis = fig.add_subplot(grid[1, 1])
    table_axis.axis("off")
    rows = []
    for layer in summary["layers"]:
        rows.append(
            [
                f"L{layer['index'] + 1}",
                f"{layer['connection_count']:,}",
                f"{layer['used_inputs']:,}/{layer['input_width']:,}",
                f"{layer['mean_fanout']:.2f}",
                f"{layer['max_fanout']:,}",
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=("Layer", "input wires", "used inputs", "mean fan-out", "max fan-out"),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    table_axis.set_title("Exact learned connectivity", pad=8)

    fig.legend(
        handles=_used_gate_legend(counts),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compact_adjacency(compact: CompactCircuit) -> np.ndarray:
    layer_count = len(compact.raw_layer_widths)
    # Rows: primary input followed by source layer. Columns: destination layer.
    adjacency = np.zeros((layer_count + 1, layer_count), dtype=np.int64)
    for node in range(compact.physical_gate_count):
        destination = int(compact.node_layers[node])
        children = [int(compact.child_a[node])]
        if int(compact.node_kinds[node]) == BINARY_KIND:
            children.append(int(compact.child_b[node]))
        for child in children:
            if child <= -3:
                adjacency[0, destination] += 1
            elif child >= 0:
                source = int(compact.node_layers[child])
                adjacency[source + 1, destination] += 1
    return adjacency


def plot_compact_circuit(
    compact: CompactCircuit, summary: dict[str, Any], output: Path
) -> None:
    counts = _gate_counts_compact(compact)
    raw = np.asarray(compact.raw_layer_widths)
    physical = np.asarray([int(count.sum()) for count in counts])
    fig = plt.figure(figsize=(21, 10.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1.0), hspace=0.28, wspace=0.28)
    pipeline = fig.add_subplot(grid[0, :])
    _pipeline_axis(
        pipeline,
        counts,
        [
            f"{kept:,} physical\n({100 * kept / original:.1f}% of raw)"
            for kept, original in zip(physical, raw)
        ],
        input_width=compact.input_width,
        output_subtitle=(
            f"{compact.final_width // compact.num_classes:,} bits/class "
            f"→ {compact.num_classes} scores"
        ),
        compact=True,
    )
    pipeline.set_title(
        "Discrete dense LGN — compacted, reachable circuit",
        fontsize=18,
        weight="bold",
        pad=38,
    )
    pipeline.text(
        0.5,
        1.01,
        (
            "Constants and identity gates are propagated, adjacent NOTs are "
            "folded, and logic not feeding GroupSum is removed."
        ),
        transform=pipeline.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#455a64",
    )

    width_axis = fig.add_subplot(grid[1, 0])
    positions = np.arange(len(raw))
    width_axis.barh(positions, raw, color="#cfd8dc", label="regular gates")
    width_axis.barh(positions, physical, color="#00897b", label="reachable physical gates")
    width_axis.set_yticks(positions, [f"Source layer {i + 1}" for i in positions])
    width_axis.invert_yaxis()
    width_axis.set_xlabel("Gate count")
    width_axis.grid(axis="x", alpha=0.22)
    reduction = 100 * float(summary["reduction_fraction"])
    width_axis.set_title(
        f"{compact.physical_gate_count:,} physical gates; {reduction:.1f}% removed/absorbed"
    )
    width_axis.legend(loc="lower right", frameon=False)

    adjacency_axis = fig.add_subplot(grid[1, 1])
    adjacency = _compact_adjacency(compact)
    image = adjacency_axis.imshow(np.log1p(adjacency), cmap="Blues", aspect="auto")
    adjacency_axis.set_xticks(
        np.arange(adjacency.shape[1]),
        [f"L{i + 1}" for i in range(adjacency.shape[1])],
    )
    adjacency_axis.set_yticks(
        np.arange(adjacency.shape[0]),
        ["input", *[f"L{i + 1}" for i in range(adjacency.shape[0] - 1)]],
    )
    adjacency_axis.set_xlabel("Destination source-layer bucket")
    adjacency_axis.set_ylabel("Source")
    adjacency_axis.set_title("Reachable connections (log color; labels are exact)")
    for row in range(adjacency.shape[0]):
        for column in range(adjacency.shape[1]):
            value = int(adjacency[row, column])
            if value:
                adjacency_axis.text(
                    column,
                    row,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if image.norm(np.log1p(value)) > 0.55 else "#263238",
                )

    fig.legend(
        handles=_used_gate_legend(counts),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
