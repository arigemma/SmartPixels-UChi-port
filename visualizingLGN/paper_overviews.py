"""Large standalone paper figures for the non-atlas LGN circuit overviews.

The existing ``regular_network`` and ``compacted_network`` plots combine
several panels for convenient inspection.  This module presents those same
aggregated circuit facts as separate figures with publication-scale text.
It does not render gate atlases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np

from .circuit import GATE_COLORS, GATE_NAMES, CompactCircuit, DenseCircuit
from .plotting import (
    _compact_adjacency,
    _gate_counts_compact,
    _gate_counts_regular,
)


SUPPORTED_FORMATS = ("png", "pdf")


def _validate_style(
    formats: Sequence[str], dpi: int, font_scale: float
) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).lower() for value in formats))
    if not normalized:
        raise ValueError("At least one paper-figure format is required.")
    unsupported = sorted(set(normalized) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise ValueError(
            f"Unsupported formats {unsupported}; expected one or more of "
            f"{SUPPORTED_FORMATS}."
        )
    if dpi < 72:
        raise ValueError("Paper-figure DPI must be at least 72.")
    if font_scale <= 0:
        raise ValueError("Paper-figure font scale must be positive.")
    return normalized


def _save(figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for output_format in formats:
        output = stem.with_suffix(f".{output_format}")
        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.22,
            facecolor="white",
        )
        outputs.append(output)
    return outputs


def _draw_pipeline_stage(
    axis,
    x: float,
    counts: np.ndarray,
    *,
    title: str,
    subtitle: str,
    font_scale: float,
    empty_label: str = "",
) -> None:
    width, bottom, height = 0.76, 0.24, 0.52
    total = int(np.sum(counts))
    box = FancyBboxPatch(
        (x - width / 2, bottom),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=2.0,
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
            fontsize=13 * font_scale,
            color="#616161",
        )
    axis.text(
        x,
        0.90,
        title,
        ha="center",
        va="center",
        fontsize=18 * font_scale,
        fontweight="bold",
        linespacing=0.90,
    )
    axis.text(
        x,
        0.12,
        subtitle,
        ha="center",
        va="center",
        fontsize=14.5 * font_scale,
        linespacing=1.15,
    )


def _draw_pipeline_endpoint(
    axis,
    x: float,
    *,
    title: str,
    subtitle: str,
    color: str,
    font_scale: float,
) -> None:
    box = FancyBboxPatch(
        (x - 0.41, 0.33),
        0.82,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.0,
        edgecolor="#263238",
        facecolor=color,
        zorder=2,
    )
    axis.add_patch(box)
    axis.text(
        x,
        0.50,
        title,
        ha="center",
        va="center",
        fontsize=18 * font_scale,
        fontweight="bold",
    )
    axis.text(
        x,
        0.12,
        subtitle,
        ha="center",
        va="center",
        fontsize=14.5 * font_scale,
        linespacing=1.15,
    )


def _plot_pipeline(
    counts: list[np.ndarray],
    subtitles: list[str],
    *,
    input_width: int,
    output_subtitle: str,
    compact: bool,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    number_layers = len(counts)
    figure, axis = plt.subplots(figsize=(16.5, 6.4))
    for x in range(number_layers + 1):
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.41, 0.50),
                (x + 0.59, 0.50),
                arrowstyle="-|>",
                mutation_scale=23,
                linewidth=2.0,
                color="#546e7a",
                zorder=1,
            )
        )
    _draw_pipeline_endpoint(
        axis,
        0,
        title="Threshold\nbits",
        subtitle=f"{input_width:,}\nBoolean inputs",
        color="#e3f2fd",
        font_scale=font_scale,
    )
    for index, (layer_counts, subtitle) in enumerate(zip(counts, subtitles), start=1):
        _draw_pipeline_stage(
            axis,
            index,
            layer_counts,
            title=f"LogicDense\n{index}",
            subtitle=subtitle,
            font_scale=font_scale,
            empty_label="no physical\ngates" if compact else "",
        )
    _draw_pipeline_endpoint(
        axis,
        number_layers + 1,
        title="GroupSum",
        subtitle=output_subtitle,
        color="#fff3e0",
        font_scale=font_scale,
    )
    axis.set_xlim(-0.55, number_layers + 1.55)
    axis.set_ylim(0.01, 1.0)
    axis.axis("off")
    axis.set_title(
        (
            "Discrete dense LGN — regular circuit"
            if not compact
            else "Discrete dense LGN — compacted reachable circuit"
        ),
        fontsize=30 * font_scale,
        fontweight="bold",
        pad=24,
    )
    figure.tight_layout(pad=1.6)
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def _plot_regular_gate_counts(
    circuit: DenseCircuit,
    *,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    widths = np.asarray(circuit.layer_widths)
    positions = np.arange(len(widths))
    figure, axis = plt.subplots(figsize=(11.5, 8.0))
    axis.barh(positions, widths, color="#607d8b", height=0.68)
    axis.set_yticks(
        positions,
        [f"LogicDense {index + 1}" for index in positions],
        fontsize=20 * font_scale,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Learned gate count", fontsize=23 * font_scale, labelpad=12)
    axis.tick_params(axis="x", labelsize=18 * font_scale)
    axis.set_title(
        f"Regular circuit: all {circuit.raw_gate_count:,} gates retained",
        fontsize=29 * font_scale,
        fontweight="bold",
        pad=24,
    )
    axis.grid(axis="x", alpha=0.22, linewidth=1.2)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(widths) * 1.18)
    for y, width in zip(positions, widths):
        axis.text(
            width + max(widths) * 0.018,
            y,
            f"{width:,}",
            va="center",
            fontsize=18 * font_scale,
        )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=2.0)
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def _plot_connectivity_table(
    summary: dict[str, Any],
    *,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    rows = [
        [
            f"LogicDense {layer['index'] + 1}",
            f"{layer['connection_count']:,}",
            f"{layer['used_inputs']:,} / {layer['input_width']:,}",
            f"{layer['mean_fanout']:.2f}",
            f"{layer['max_fanout']:,}",
        ]
        for layer in summary["layers"]
    ]
    figure, axis = plt.subplots(figsize=(14.5, 7.5))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "Layer",
            "Input wires",
            "Used inputs",
            "Mean fan-out",
            "Maximum fan-out",
        ),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(18 * font_scale)
    table.scale(1.0, 2.25 * font_scale)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_facecolor("#eceff1")
            cell.set_text_props(fontweight="bold")
    axis.set_title(
        "Regular circuit: exact learned connectivity",
        fontsize=30 * font_scale,
        fontweight="bold",
        pad=30,
    )
    figure.tight_layout(pad=2.0)
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def _plot_compact_gate_counts(
    compact: CompactCircuit,
    summary: dict[str, Any],
    counts: list[np.ndarray],
    *,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    raw = np.asarray(compact.raw_layer_widths)
    physical = np.asarray([int(count.sum()) for count in counts])
    positions = np.arange(len(raw))
    figure, axis = plt.subplots(figsize=(12.5, 8.2))
    axis.barh(
        positions,
        raw,
        color="#cfd8dc",
        height=0.70,
        label="Regular gates",
    )
    axis.barh(
        positions,
        physical,
        color="#00897b",
        height=0.70,
        label="Reachable physical gates",
    )
    axis.set_yticks(
        positions,
        [f"Source LogicDense {index + 1}" for index in positions],
        fontsize=19 * font_scale,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Gate count", fontsize=23 * font_scale, labelpad=12)
    axis.tick_params(axis="x", labelsize=18 * font_scale)
    axis.grid(axis="x", alpha=0.22, linewidth=1.2)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(raw) * 1.24)
    reduction = 100 * float(summary["reduction_fraction"])
    axis.set_title(
        (
            f"Compacted circuit: {compact.physical_gate_count:,} physical gates; "
            f"{reduction:.1f}% removed or absorbed"
        ),
        fontsize=27 * font_scale,
        fontweight="bold",
        pad=24,
    )
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=18 * font_scale,
    )
    for y, (kept, original) in enumerate(zip(physical, raw)):
        axis.text(
            original + max(raw) * 0.016,
            y,
            f"{kept:,} / {original:,}",
            va="center",
            fontsize=16 * font_scale,
        )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=2.0)
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def _plot_compact_adjacency(
    compact: CompactCircuit,
    *,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    adjacency = _compact_adjacency(compact)
    figure, axis = plt.subplots(figsize=(12.0, 10.0))
    image = axis.imshow(np.log1p(adjacency), cmap="Blues", aspect="auto")
    axis.set_xticks(
        np.arange(adjacency.shape[1]),
        [f"L{index + 1}" for index in range(adjacency.shape[1])],
        fontsize=19 * font_scale,
    )
    axis.set_yticks(
        np.arange(adjacency.shape[0]),
        ["Input", *[f"L{index + 1}" for index in range(adjacency.shape[0] - 1)]],
        fontsize=19 * font_scale,
    )
    axis.set_xlabel(
        "Destination source-layer bucket",
        fontsize=22 * font_scale,
        labelpad=13,
    )
    axis.set_ylabel("Source", fontsize=22 * font_scale, labelpad=13)
    axis.set_title(
        "Compacted circuit: reachable connections",
        fontsize=30 * font_scale,
        fontweight="bold",
        pad=24,
    )
    for row in range(adjacency.shape[0]):
        for column in range(adjacency.shape[1]):
            value = int(adjacency[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=13.5 * font_scale,
                    fontweight="bold",
                    color=(
                        "white"
                        if image.norm(np.log1p(value)) > 0.55
                        else "#263238"
                    ),
                )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label(
        "log(1 + connection count)",
        fontsize=18 * font_scale,
        labelpad=12,
    )
    colorbar.ax.tick_params(labelsize=15 * font_scale)
    figure.tight_layout(pad=2.0)
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def _plot_lut_legend(
    counts: list[np.ndarray],
    *,
    title: str,
    stem: Path,
    formats: Sequence[str],
    dpi: int,
    font_scale: float,
) -> list[Path]:
    total = np.sum(np.stack(counts), axis=0)
    handles = [
        Patch(facecolor=GATE_COLORS[index], label=f"LUT {index}: {GATE_NAMES[index]}")
        for index in range(16)
        if total[index]
    ]
    figure, axis = plt.subplots(figsize=(15.0, 6.5))
    axis.axis("off")
    figure.suptitle(
        title,
        fontsize=30 * font_scale,
        fontweight="bold",
        y=0.96,
    )
    figure.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=19 * font_scale,
        handlelength=1.8,
        handleheight=1.25,
        columnspacing=2.0,
        labelspacing=1.15,
    )
    outputs = _save(figure, stem, formats, dpi)
    plt.close(figure)
    return outputs


def render_paper_overview_set(
    circuit: DenseCircuit,
    compact: CompactCircuit,
    regular_summary: dict[str, Any],
    compact_summary: dict[str, Any],
    output_dir: Path,
    *,
    formats: Sequence[str] = SUPPORTED_FORMATS,
    dpi: int = 300,
    font_scale: float = 1.0,
) -> list[Path]:
    """Write the non-atlas overview panels as large standalone figures."""
    normalized_formats = _validate_style(formats, dpi, font_scale)
    regular_counts = _gate_counts_regular(circuit)
    compact_counts = _gate_counts_compact(compact)
    regular_dir = output_dir / "regular"
    compact_dir = output_dir / "compacted"
    outputs: list[Path] = []

    outputs.extend(
        _plot_pipeline(
            regular_counts,
            [f"{width:,}\nlearned gates" for width in circuit.layer_widths],
            input_width=circuit.input_width,
            output_subtitle=(
                f"{circuit.final_width // circuit.num_classes:,} bits/class\n"
                f"{circuit.num_classes} scores"
            ),
            compact=False,
            stem=regular_dir / "regular_pipeline",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_regular_gate_counts(
            circuit,
            stem=regular_dir / "regular_gate_counts",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_connectivity_table(
            regular_summary,
            stem=regular_dir / "regular_connectivity",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_lut_legend(
            regular_counts,
            title="Regular circuit: selected rank-2 truth tables",
            stem=regular_dir / "regular_lut_legend",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )

    raw = np.asarray(compact.raw_layer_widths)
    physical = np.asarray([int(count.sum()) for count in compact_counts])
    outputs.extend(
        _plot_pipeline(
            compact_counts,
            [
                f"{kept:,} physical\n{100 * kept / original:.1f}% of raw"
                for kept, original in zip(physical, raw)
            ],
            input_width=compact.input_width,
            output_subtitle=(
                f"{compact.final_width // compact.num_classes:,} bits/class\n"
                f"{compact.num_classes} scores"
            ),
            compact=True,
            stem=compact_dir / "compacted_pipeline",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_compact_gate_counts(
            compact,
            compact_summary,
            compact_counts,
            stem=compact_dir / "compacted_gate_counts",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_compact_adjacency(
            compact,
            stem=compact_dir / "compacted_reachable_connections",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    outputs.extend(
        _plot_lut_legend(
            compact_counts,
            title="Compacted circuit: reachable physical truth tables",
            stem=compact_dir / "compacted_lut_legend",
            formats=normalized_formats,
            dpi=dpi,
            font_scale=font_scale,
        )
    )
    return outputs
