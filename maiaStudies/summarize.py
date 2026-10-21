"""Create comprehensive MAIA model-comparison tables and plots.

The plotting contract is deliberately data-first: every plotted coordinate is
also written to TSV/CSV/JSON so figures can be restyled without re-evaluation.
Dense Model 2 is consistently blue and convolutional Model 3 consistently
orange; marker shape encodes GroupSum temperature.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .campaign import CAMPAIGNS, get_campaign_configs


FAMILY_STYLE = {
    "dense": {
        "label": "Dense LNN (Model 2)",
        "short": "M2",
        "color": "#0072B2",
    },
    "conv": {
        "label": "Convolutional LNN (Model 3)",
        "short": "M3",
        "color": "#D55E00",
    },
}
TAU_MARKERS = {1.0: "D", 3.0: "P", 10.0: "o", 20.0: "s", 40.0: "^"}
TAU_LINESTYLES = {
    1.0: (0, (1, 1)),
    3.0: (0, (3, 1, 1, 1)),
    10.0: "-",
    20.0: "--",
    40.0: ":",
}
TARGETS = (90, 95, 99)
SUBSET_LABEL = {"full_test": "Full test", "timing_cut_test": "Timing cut"}
CANONICAL_INPUT_SCHEMA = "maia_model2_yprofile_model3_yprofile_timing8_v3"
LEGACY_NATIVE_INPUT_SCHEMA = "maia_model2_model3_spatial_v2"


@dataclass(frozen=True)
class _LegacyNativeConfig:
    """Plot-only description of the completed 37/276-input campaign."""

    task_index: int
    alias: str
    family: str
    size: str
    tau: float
    thresholds_per_feature: int
    raw_input_features: int
    dense_widths: tuple[int, ...]
    conv_kernels: tuple[int, ...] = ()

    @property
    def final_logic_width(self) -> int:
        return self.dense_widths[-1]

    @property
    def group_width_per_class(self) -> int:
        return self.final_logic_width // 2


def _legacy_native_configs() -> tuple[_LegacyNativeConfig, ...]:
    dense_sizes = (
        ("s01", (1900, 1900, 1900)),
        ("s02", (2600, 2600, 2600)),
        ("s03", (3600, 3600, 3600)),
        ("s04", (5000, 5000, 5000)),
        ("s05", (7000, 7000, 7000)),
    )
    conv_sizes = (
        ("s01", (16, 32, 64, 64), (640, 640, 320)),
        ("s02", (24, 48, 96, 96), (896, 896, 448)),
        ("s03", (32, 64, 128, 128), (1152, 1152, 576)),
        ("s04", (48, 96, 192, 192), (1664, 1664, 832)),
        ("s05", (64, 128, 256, 256), (2112, 2112, 1056)),
    )
    rows = []
    for size, widths in dense_sizes:
        for tau in (10.0, 20.0, 40.0):
            rows.append(
                _LegacyNativeConfig(
                    len(rows), f"dense_{size}_tau{int(tau)}", "dense", size,
                    tau, 100, 37, widths,
                )
            )
    for size, kernels, widths in conv_sizes:
        for tau in (10.0, 20.0, 40.0):
            rows.append(
                _LegacyNativeConfig(
                    len(rows), f"conv_{size}_tau{int(tau)}", "conv", size,
                    tau, 100, 276, widths, kernels,
                )
            )
    return tuple(rows)


def _summary_configs(campaign: str):
    if campaign == "native_legacy":
        return _legacy_native_configs()
    return get_campaign_configs(campaign)


def _family_label(family: str, input_schema: str | None) -> str:
    if input_schema == LEGACY_NATIVE_INPUT_SCHEMA:
        return (
            "MAIA-native dense LNN (37 inputs)"
            if family == "dense"
            else "MAIA-native convolutional LNN (276 inputs)"
        )
    return (
        "Canonical dense LNN (14 inputs)"
        if family == "dense"
        else "Canonical convolutional LNN (105 inputs)"
    )


def _configure_matplotlib(font_scale: float) -> None:
    base = 12.0 * font_scale
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.labelsize": base,
            "axes.titlesize": 1.08 * base,
            "xtick.labelsize": 0.88 * base,
            "ytick.labelsize": 0.88 * base,
            "legend.fontsize": 0.82 * base,
            "figure.titlesize": 1.15 * base,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _model_label(row: dict, exact: bool = False) -> str:
    if exact:
        return str(row["alias"])
    style = FAMILY_STYLE[row["family"]]
    return f"{style['short']} {row['size']} τ{int(row['tau'])}"


def _operating_point(metrics: dict, target: int) -> dict:
    return metrics["validation_derived_operating_points"][
        f"signal_efficiency_{target}pct"
    ]


def _flatten(config, result: dict, subset: str) -> dict:
    metrics = result[subset]
    argmax = metrics["argmax"]
    family_label = _family_label(config.family, result.get("input_schema"))
    row = {
        "task_index": config.task_index,
        "alias": config.alias,
        "display_name": f"{family_label} {config.size}, tau={int(config.tau)}",
        "family": config.family,
        "family_label": family_label,
        "size": config.size,
        "tau": config.tau,
        "subset": subset,
        "subset_label": SUBSET_LABEL[subset],
        "thresholds_per_feature": config.thresholds_per_feature,
        "raw_input_features": config.raw_input_features,
        "conv_kernels": ",".join(map(str, config.conv_kernels)) or "-",
        "dense_widths": ",".join(map(str, config.dense_widths)),
        "final_logic_width": config.final_logic_width,
        "group_width_per_class": config.group_width_per_class,
        "checkpoint_step": result["checkpoint_step"],
        "trainable_parameters": result["trainable_parameters"],
        "logic_node_proxy": result["logic_node_proxy"],
        "group_sum_inputs": result["group_sum_inputs"],
        "thermometer_input_bits": result["thermometer_input_bits"],
        "balanced_accuracy": argmax["balanced_accuracy"],
        "signal_efficiency_argmax": argmax["signal_efficiency"],
        "bib_acceptance_argmax": argmax["bib_acceptance"],
        "bib_rejection_argmax": argmax["bib_rejection"],
        "bib_rejection_fraction_argmax": argmax["bib_rejection_fraction"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "n_examples": metrics["n_examples"],
        "n_signal": metrics["n_signal"],
        "n_bib": metrics["n_bib"],
        "timing_test_signal_acceptance": result["timing_cut"][
            "test_signal_acceptance"
        ],
        "timing_test_bib_acceptance": result["timing_cut"]["test_bib_acceptance"],
        "checkpoint": result["checkpoint"],
    }
    for target in TARGETS:
        point = _operating_point(metrics, target)
        prefix = f"at_{target}pct_signal_op"
        row[f"score_threshold_{prefix}"] = point["threshold"]
        row[f"signal_efficiency_{prefix}"] = point["signal_efficiency"]
        row[f"bib_acceptance_{prefix}"] = point["bib_acceptance"]
        row[f"bib_rejection_{prefix}"] = point["bib_rejection"]
        row[f"bib_rejection_fraction_{prefix}"] = point["bib_rejection_fraction"]
    return row


def _write_delimited(path: Path, rows: list[dict], delimiter: str) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def _wide_rows(rows: list[dict]) -> list[dict]:
    by_alias: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_alias[row["alias"]][row["subset"]] = row
    identity = (
        "task_index",
        "alias",
        "display_name",
        "family",
        "family_label",
        "size",
        "tau",
        "thresholds_per_feature",
        "raw_input_features",
        "conv_kernels",
        "dense_widths",
        "final_logic_width",
        "group_width_per_class",
        "checkpoint_step",
        "trainable_parameters",
        "logic_node_proxy",
        "group_sum_inputs",
        "thermometer_input_bits",
        "timing_test_signal_acceptance",
        "timing_test_bib_acceptance",
        "checkpoint",
    )
    output = []
    for alias in sorted(by_alias, key=lambda name: by_alias[name]["full_test"]["task_index"]):
        subsets = by_alias[alias]
        full = subsets["full_test"]
        combined = {key: full[key] for key in identity}
        for subset in ("full_test", "timing_cut_test"):
            for key, value in subsets[subset].items():
                if key not in identity and key not in {"subset", "subset_label"}:
                    combined[f"{subset}_{key}"] = value
        output.append(combined)
    return output


def _assert_campaign_learned(rows: list[dict]) -> None:
    """Catch chance-level campaigns before producing presentation figures."""
    full = [row for row in rows if row["subset"] == "full_test"]
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in full])
    auc = np.asarray([float(row["roc_auc"]) for row in full])
    if balanced.size == 0:
        raise RuntimeError("No full-test rows are available")
    if float(balanced.max()) < 0.55 and float(np.median(auc)) < 0.55:
        raise RuntimeError(
            "MAIA campaign is at chance level; refusing to generate polished plots. "
            f"best balanced accuracy={balanced.max():.4f}, median ROC AUC="
            f"{np.median(auc):.4f}. Inspect the data/model pipeline. Pass "
            "--allow-collapsed only to make explicitly diagnostic plots."
        )


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        written.append(str(path))
    plt.close(figure)
    return written


def _add_style_legends(
    axis: plt.Axes, taus: list[float], *, location: str = "best"
) -> None:
    family_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=style["color"], markeredgecolor=style["color"],
            label=style["label"],
        )
        for style in FAMILY_STYLE.values()
    ]
    tau_handles = [
        Line2D(
            [0], [0], marker=TAU_MARKERS[tau], color="#333333", linestyle="none",
            markersize=8, label=f"τ={int(tau)}",
        )
        for tau in sorted(taus)
    ]
    first = axis.legend(handles=family_handles, loc=location, title="Architecture")
    axis.add_artist(first)
    axis.legend(handles=tau_handles, loc="lower right", title="GroupSum")


def _annotate_points(axis: plt.Axes, selected: list[dict], x_key: str, y_key: str) -> None:
    for index, row in enumerate(selected):
        x, y = row.get(x_key), row.get(y_key)
        if x is None or y is None or not np.isfinite(float(x)) or not np.isfinite(float(y)):
            continue
        offset_y = 5 if index % 2 == 0 else -10
        axis.annotate(
            _model_label(row),
            (x, y),
            xytext=(4, offset_y),
            textcoords="offset points",
            fontsize=7.3,
            alpha=0.92,
        )


def _family_tau_scatter(
    rows: list[dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    *,
    x_log: bool = True,
    y_log: bool = False,
    annotate: bool = True,
) -> plt.Figure:
    selected = [
        row
        for row in rows
        if row["subset"] == "full_test"
        and row.get(x_key) is not None
        and row.get(y_key) is not None
    ]
    observed_taus = sorted({float(row["tau"]) for row in selected})
    figure, axis = plt.subplots(figsize=(10.8, 7.0))
    for family, style in FAMILY_STYLE.items():
        for tau in observed_taus:
            points = sorted(
                (
                    row
                    for row in selected
                    if row["family"] == family and float(row["tau"]) == tau
                ),
                key=lambda row: float(row[x_key]),
            )
            if not points:
                continue
            x = [float(row[x_key]) for row in points]
            y = [float(row[y_key]) for row in points]
            axis.plot(
                x,
                y,
                color=style["color"],
                linestyle=TAU_LINESTYLES[tau],
                linewidth=1.4,
                alpha=0.55,
            )
            axis.scatter(
                x,
                y,
                color=style["color"],
                marker=TAU_MARKERS[tau],
                s=72,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
    if annotate:
        _annotate_points(axis, selected, x_key, y_key)
    if x_log:
        axis.set_xscale("log")
    if y_log:
        axis.set_yscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.22)
    _add_style_legends(axis, observed_taus, location="best")
    figure.tight_layout()
    return figure


def _comparison_plot(
    rows: list[dict], metric: str, label: str, title: str, *, log_scale: bool = False
) -> plt.Figure:
    by_alias = defaultdict(dict)
    for row in rows:
        by_alias[row["alias"]][row["subset"]] = row
    pairs = []
    for subsets in by_alias.values():
        full = subsets.get("full_test", {}).get(metric)
        cut = subsets.get("timing_cut_test", {}).get(metric)
        if full is not None and cut is not None and float(full) > 0 and float(cut) > 0:
            pairs.append((subsets["full_test"], float(full), float(cut)))
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    if not pairs:
        axis.text(
            0.5,
            0.5,
            "No finite values available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        axis.set_title(title)
        figure.tight_layout()
        return figure
    values = [value for _, full, cut in pairs for value in (full, cut)]
    lower, upper = min(values), max(values)
    padding = 0.04 * (upper - lower if upper > lower else upper)
    lower = max(np.finfo(float).tiny, lower - padding) if log_scale else lower - padding
    upper += padding
    axis.plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=1)
    for row, full, cut in pairs:
        style = FAMILY_STYLE[row["family"]]
        axis.scatter(
            full,
            cut,
            color=style["color"],
            marker=TAU_MARKERS[float(row["tau"])],
            s=70,
            edgecolors="white",
            linewidths=0.7,
        )
        axis.annotate(_model_label(row), (full, cut), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    if log_scale:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel(f"Full test: {label}")
    axis.set_ylabel(f"After timing cut: {label}")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.22)
    observed_taus = sorted({float(row["tau"]) for row, _, _ in pairs})
    _add_style_legends(axis, observed_taus)
    figure.tight_layout()
    return figure


def _ranking_plot(rows: list[dict], metric: str, label: str, title: str) -> plt.Figure:
    selected = sorted(
        (
            row
            for row in rows
            if row["subset"] == "full_test" and row.get(metric) is not None
        ),
        key=lambda row: float(row[metric]),
    )
    figure, axis = plt.subplots(figsize=(11.5, max(8.0, 0.34 * len(selected))))
    if not selected:
        axis.text(
            0.5,
            0.5,
            "No finite values available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        axis.set_title(title)
        figure.tight_layout()
        return figure
    y = np.arange(len(selected))
    colors = [FAMILY_STYLE[row["family"]]["color"] for row in selected]
    bars = axis.barh(y, [float(row[metric]) for row in selected], color=colors, alpha=0.88)
    axis.set_yticks(y, [_model_label(row, exact=True) for row in selected])
    axis.set_xlabel(label)
    axis.set_title(title)
    axis.grid(True, axis="x", alpha=0.22)
    for bar, row in zip(bars, selected):
        value = float(row[metric])
        formatting = f"{value:.3f}" if value < 10 else f"{value:.1f}"
        axis.text(value, bar.get_y() + bar.get_height() / 2, f" {formatting}", va="center", fontsize=7.5)
    family_handles = [
        Line2D([0], [0], color=style["color"], linewidth=7, label=style["label"])
        for style in FAMILY_STYLE.values()
    ]
    axis.legend(handles=family_handles, loc="lower right")
    figure.tight_layout()
    return figure


def _heatmap(rows: list[dict], metric: str, label: str, title: str) -> plt.Figure:
    full = [row for row in rows if row["subset"] == "full_test"]
    sizes = sorted({row["size"] for row in full})
    taus = sorted({float(row["tau"]) for row in full})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    finite_values = [float(row[metric]) for row in full if row.get(metric) is not None]
    if not finite_values:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No finite values available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        figure.suptitle(title)
        return figure
    minimum, maximum = min(finite_values), max(finite_values)
    image = None
    for axis, (family, style) in zip(axes, FAMILY_STYLE.items()):
        matrix = np.full((len(taus), len(sizes)), np.nan)
        for row in full:
            if row["family"] == family and row.get(metric) is not None:
                matrix[taus.index(float(row["tau"])), sizes.index(row["size"])] = float(row[metric])
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=minimum, vmax=maximum)
        axis.set_xticks(range(len(sizes)), sizes)
        axis.set_yticks(range(len(taus)), [f"τ={int(tau)}" for tau in taus])
        axis.set_xlabel("Architecture size")
        axis.set_title(style["label"])
        for row_index in range(len(taus)):
            for column_index in range(len(sizes)):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
                    axis.text(column_index, row_index, text, ha="center", va="center", color="white", fontsize=9)
    axes[0].set_ylabel("GroupSum temperature")
    figure.colorbar(image, ax=axes, label=label, shrink=0.88)
    figure.suptitle(title)
    return figure


def _operating_points_plot(rows: list[dict]) -> plt.Figure:
    full = [row for row in rows if row["subset"] == "full_test"]
    observed_taus = sorted({float(row["tau"]) for row in full})
    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    for row in full:
        style = FAMILY_STYLE[row["family"]]
        x, y = [], []
        for target in TARGETS:
            efficiency = row[f"signal_efficiency_at_{target}pct_signal_op"]
            rejection = row[f"bib_rejection_at_{target}pct_signal_op"]
            if rejection is not None:
                x.append(float(efficiency))
                y.append(float(rejection))
        if not x:
            continue
        axis.plot(
            x,
            y,
            color=style["color"],
            linestyle=TAU_LINESTYLES[float(row["tau"])],
            linewidth=1.0,
            alpha=0.55,
        )
        axis.scatter(
            x,
            y,
            color=style["color"],
            marker=TAU_MARKERS[float(row["tau"])],
            s=34,
            alpha=0.8,
        )
        axis.annotate(_model_label(row), (x[0], y[0]), xytext=(3, 2), textcoords="offset points", fontsize=6.7)
    axis.set_xlabel("Observed test signal efficiency")
    axis.set_ylabel("BIB rejection factor")
    axis.set_yscale("log")
    axis.set_title("Validation-derived 90%, 95%, and 99% signal-efficiency operating points")
    axis.grid(True, which="both", alpha=0.22)
    _add_style_legends(axis, observed_taus)
    figure.tight_layout()
    return figure


def _timing_retention_plot(rows: list[dict]) -> plt.Figure:
    selected = sorted(
        (row for row in rows if row["subset"] == "full_test"),
        key=lambda row: row["task_index"],
    )
    figure, axes = plt.subplots(2, 1, figsize=(14.5, 8.5), sharex=True)
    x = np.arange(len(selected))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        [row["timing_test_signal_acceptance"] for row in selected],
        width,
        color="#009E73",
        label="Signal retained",
    )
    axes[0].bar(
        x + width / 2,
        [row["timing_test_bib_acceptance"] for row in selected],
        width,
        color="#CC79A7",
        label="BIB retained",
    )
    delta = []
    by_alias = defaultdict(dict)
    for row in rows:
        by_alias[row["alias"]][row["subset"]] = row
    for row in selected:
        delta.append(
            by_alias[row["alias"]]["timing_cut_test"]["balanced_accuracy"]
            - row["balanced_accuracy"]
        )
    axes[1].bar(
        x,
        delta,
        color=[FAMILY_STYLE[row["family"]]["color"] for row in selected],
    )
    axes[0].set_ylabel("Fraction retained")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=2)
    axes[0].set_title("Effect of the post-training timing selection")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_ylabel("Δ balanced accuracy\n(timing cut − full)")
    axes[1].set_xticks(x, [_model_label(row) for row in selected], rotation=70, ha="right")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
    figure.tight_layout()
    return figure


def _read_roc(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {key: float(value) for key, value in row.items()}
            if (
                parsed["signal_efficiency"] >= 0.5
                and parsed["bib_acceptance"] > 0
                and np.isfinite(parsed["bib_rejection"])
            ):
                rows.append(parsed)
    if len(rows) > 500:
        indices = np.unique(np.linspace(0, len(rows) - 1, 500, dtype=int))
        rows = [rows[index] for index in indices]
    return rows


def _all_roc_plot(result_root: Path, rows: list[dict], timing_cut: bool) -> plt.Figure:
    selected = [row for row in rows if row["subset"] == "full_test"]
    observed_taus = sorted({float(row["tau"]) for row in selected})
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    filename = "roc_timing_cut.csv" if timing_cut else "roc_full.csv"
    for row in selected:
        roc = _read_roc(result_root / row["alias"] / filename)
        if not roc:
            continue
        style = FAMILY_STYLE[row["family"]]
        axis.plot(
            [point["signal_efficiency"] for point in roc],
            [point["bib_rejection"] for point in roc],
            color=style["color"],
            linestyle=TAU_LINESTYLES[float(row["tau"])],
            linewidth=1.0,
            alpha=0.42,
        )
    axis.set_xlabel("Signal efficiency")
    axis.set_ylabel("BIB rejection factor")
    axis.set_yscale("log")
    axis.set_xlim(0.5, 1.0)
    axis.set_title(
        "All model rejection curves — " + ("after timing cut" if timing_cut else "full test")
    )
    axis.grid(True, which="both", alpha=0.22)
    _add_style_legends(axis, observed_taus)
    figure.tight_layout()
    return figure


def _dashboard(rows: list[dict]) -> plt.Figure:
    full = [row for row in rows if row["subset"] == "full_test"]
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.5))
    specifications = (
        ("trainable_parameters", "balanced_accuracy", "Trainable parameters", "Balanced accuracy", True, False),
        ("logic_node_proxy", "balanced_accuracy", "Logic-node proxy", "Balanced accuracy", True, False),
        ("trainable_parameters", "roc_auc", "Trainable parameters", "ROC AUC", True, False),
        (
            "trainable_parameters",
            "bib_rejection_fraction_at_99pct_signal_op",
            "Trainable parameters",
            "BIB fraction rejected @ 99% signal OP",
            True,
            False,
        ),
        (
            "signal_efficiency_at_99pct_signal_op",
            "bib_rejection_fraction_at_99pct_signal_op",
            "Signal efficiency",
            "BIB fraction rejected",
            False,
            False,
        ),
    )
    for axis, (x_key, y_key, x_label, y_label, x_log, y_log) in zip(axes.flat, specifications):
        for row in full:
            if row.get(x_key) is None or row.get(y_key) is None:
                continue
            style = FAMILY_STYLE[row["family"]]
            axis.scatter(
                row[x_key], row[y_key], color=style["color"],
                marker=TAU_MARKERS[float(row["tau"])], s=42, alpha=0.85,
            )
        if x_log:
            axis.set_xscale("log")
        if y_log:
            axis.set_yscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, which="both", alpha=0.2)
    # Final panel: balanced-accuracy change after timing selection.
    comparison = defaultdict(dict)
    for row in rows:
        comparison[row["alias"]][row["subset"]] = row
    axis = axes.flat[-1]
    for subsets in comparison.values():
        full_row, cut_row = subsets["full_test"], subsets["timing_cut_test"]
        style = FAMILY_STYLE[full_row["family"]]
        axis.scatter(
            full_row["balanced_accuracy"], cut_row["balanced_accuracy"],
            color=style["color"], marker=TAU_MARKERS[float(full_row["tau"])], s=42,
        )
    all_balanced = [row["balanced_accuracy"] for row in rows]
    low, high = min(all_balanced), max(all_balanced)
    axis.plot([low, high], [low, high], "--", color="#555555", linewidth=1)
    axis.set_xlabel("Full-test balanced accuracy")
    axis.set_ylabel("Timing-cut balanced accuracy")
    axis.grid(True, alpha=0.2)
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=style["color"], label=style["label"])
        for style in FAMILY_STYLE.values()
    ]
    observed_taus = sorted({float(row["tau"]) for row in rows})
    tau_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color="#333333", label=f"τ={int(tau)}")
        for tau, marker in ((tau, TAU_MARKERS[tau]) for tau in observed_taus)
    ]
    figure.suptitle("MAIA LNN campaign overview", y=0.995)
    figure.legend(
        handles=family_handles + tau_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def _best_rows(rows: list[dict]) -> list[dict]:
    full = [row for row in rows if row["subset"] == "full_test"]
    output = []
    criteria = (
        (
            "bib_rejection_fraction_at_99pct_signal_op",
            "highest BIB rejection fraction at 99% signal OP",
        ),
        ("balanced_accuracy", "highest balanced accuracy"),
        ("roc_auc", "highest ROC AUC"),
        ("bib_rejection_at_90pct_signal_op", "highest BIB rejection at 90% signal OP"),
    )
    for family in FAMILY_STYLE:
        family_rows = [row for row in full if row["family"] == family]
        for metric, description in criteria:
            eligible = [row for row in family_rows if row.get(metric) is not None]
            ranked = sorted(eligible, key=lambda row: float(row[metric]), reverse=True)
            for rank, row in enumerate(ranked[:5], 1):
                output.append(
                    {
                        "family": family,
                        "criterion": description,
                        "rank": rank,
                        "alias": row["alias"],
                        "value": row[metric],
                        "trainable_parameters": row["trainable_parameters"],
                        "logic_node_proxy": row["logic_node_proxy"],
                        "tau": row["tau"],
                    }
                )
    return output


def _write_report(
    result_root: Path,
    rows: list[dict],
    missing: list[str],
    products: list[str],
    expected_models: int,
) -> Path:
    full = [row for row in rows if row["subset"] == "full_test"]
    lines = [
        "# MAIA LNN campaign summary",
        "",
        f"Models summarized: **{len(full)} / {expected_models}**.",
        "",
        "Dense Model 2 is blue; convolutional Model 3 is orange. Marker shape encodes tau.",
        "All fixed-efficiency score thresholds are derived on validation data and applied unchanged to test data.",
        "",
    ]
    for family, style in FAMILY_STYLE.items():
        candidates = [row for row in full if row["family"] == family]
        if not candidates:
            continue
        best_balanced = max(candidates, key=lambda row: float(row["balanced_accuracy"]))
        best_tail = max(
            candidates,
            key=lambda row: float(
                row["bib_rejection_fraction_at_99pct_signal_op"]
            ),
        )
        finite_rejection = [
            row for row in candidates if row["bib_rejection_at_90pct_signal_op"] is not None
        ]
        lines.extend(
            [
                f"## {style['label']}",
                "",
                f"- Best BIB rejection fraction at the validation-derived 99% "
                f"signal-efficiency OP: `{best_tail['alias']}` = "
                f"{100 * best_tail['bib_rejection_fraction_at_99pct_signal_op']:.2f}%.",
                f"- Best balanced accuracy: `{best_balanced['alias']}` = "
                f"{best_balanced['balanced_accuracy']:.5f}.",
            ]
        )
        if finite_rejection:
            best_rejection = max(
                finite_rejection,
                key=lambda row: float(row["bib_rejection_at_90pct_signal_op"]),
            )
            lines.append(
                f"- Best finite BIB rejection at the validation-derived 90% "
                f"signal-efficiency OP: `{best_rejection['alias']}` = "
                f"{best_rejection['bib_rejection_at_90pct_signal_op']:.3g}."
            )
        else:
            lines.append(
                "- BIB rejection at the 90% signal-efficiency OP is unbounded "
                "in this test sample (zero accepted BIB) for every model in the family."
            )
        lines.append("")
    if missing:
        lines.extend(["## Missing evaluations", ""] + [f"- `{path}`" for path in missing] + [""])
    lines.extend(["## Generated products", ""] + [f"- `{Path(path).name}`" for path in products] + [""])
    path = result_root / "SUMMARY.md"
    path.write_text("\n".join(lines))
    return path


def generate_summary(
    result_root: Path,
    *,
    campaign: str = "main",
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 400,
    font_scale: float = 1.0,
    allow_partial: bool = False,
    allow_collapsed: bool = False,
    annotate: bool = True,
) -> dict:
    _configure_matplotlib(font_scale)
    configs = _summary_configs(campaign)
    expected_schema = (
        LEGACY_NATIVE_INPUT_SCHEMA
        if campaign == "native_legacy"
        else CANONICAL_INPUT_SCHEMA
    )
    output_dir = output_dir or result_root / "summary_plots"
    rows: list[dict] = []
    missing: list[str] = []
    for config in configs:
        path = result_root / config.alias / "evaluation_complete.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open() as handle:
            result = json.load(handle)
        actual_schema = result.get("input_schema")
        if actual_schema != expected_schema:
            raise ValueError(
                f"Input-schema mismatch for {config.alias}: expected "
                f"{expected_schema!r} for campaign {campaign!r}, found "
                f"{actual_schema!r}. Select the matching campaign instead of "
                "mixing model definitions in one summary."
            )
        rows.append(_flatten(config, result, "full_test"))
        rows.append(_flatten(config, result, "timing_cut_test"))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(configs)} evaluations:\n"
            + "\n".join(missing)
        )
    if not rows:
        raise FileNotFoundError(f"No evaluation_complete.json files found below {result_root}")
    if not allow_collapsed:
        _assert_campaign_learned(rows)

    wide = _wide_rows(rows)
    _write_delimited(result_root / "performance_summary.tsv", rows, "\t")
    _write_delimited(result_root / "performance_summary.csv", rows, ",")
    _write_delimited(result_root / "model_comparison_wide.tsv", wide, "\t")
    _write_delimited(result_root / "best_models.tsv", _best_rows(rows), "\t")
    with (result_root / "performance_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, allow_nan=False)
        handle.write("\n")

    products = []
    plots = (
        (
            "01_balanced_accuracy_vs_parameters",
            _family_tau_scatter(
                rows, "trainable_parameters", "balanced_accuracy",
                "Trainable parameters", "Balanced accuracy",
                "Classification performance versus trainable parameters", annotate=annotate,
            ),
        ),
        (
            "02_balanced_accuracy_vs_logic_nodes",
            _family_tau_scatter(
                rows, "logic_node_proxy", "balanced_accuracy",
                "Logic-node proxy (not synthesized LUT+FF)", "Balanced accuracy",
                "Classification performance versus structural logic proxy", annotate=annotate,
            ),
        ),
        (
            "03_signal_efficiency_vs_parameters",
            _family_tau_scatter(
                rows, "trainable_parameters", "signal_efficiency_argmax",
                "Trainable parameters", "Signal efficiency at argmax",
                "Signal efficiency versus model size", annotate=annotate,
            ),
        ),
        (
            "04_bib_rejection_fraction_vs_parameters_at_99pct_signal",
            _family_tau_scatter(
                rows,
                "trainable_parameters",
                "bib_rejection_fraction_at_99pct_signal_op",
                "Trainable parameters",
                "Fraction of BIB rejected",
                "BIB rejection at the validation-derived 99% signal-efficiency OP",
                annotate=annotate,
            ),
        ),
        (
            "05_roc_auc_vs_parameters",
            _family_tau_scatter(
                rows, "trainable_parameters", "roc_auc",
                "Trainable parameters", "ROC AUC", "ROC AUC versus model size",
                annotate=annotate,
            ),
        ),
        (
            "06_signal_efficiency_vs_bib_rejection",
            _family_tau_scatter(
                rows,
                "signal_efficiency_at_99pct_signal_op",
                "bib_rejection_fraction_at_99pct_signal_op",
                "Observed test signal efficiency",
                "Fraction of BIB rejected",
                "Validation-derived 99% signal-efficiency operating point",
                x_log=False,
                annotate=annotate,
            ),
        ),
        (
            "07_full_vs_timing_balanced_accuracy",
            _comparison_plot(
                rows, "balanced_accuracy", "balanced accuracy",
                "Effect of timing selection on balanced accuracy",
            ),
        ),
        (
            "08_full_vs_timing_bib_rejection",
            _comparison_plot(
                rows,
                "bib_rejection_fraction_at_99pct_signal_op",
                "BIB fraction rejected at 99% signal efficiency",
                "Effect of timing selection on BIB rejection",
            ),
        ),
        (
            "09_balanced_accuracy_heatmaps",
            _heatmap(rows, "balanced_accuracy", "Balanced accuracy", "Architecture/tau sweep"),
        ),
        (
            "10_bib_rejection_heatmaps",
            _heatmap(
                rows,
                "bib_rejection_fraction_at_99pct_signal_op",
                "BIB fraction rejected",
                "BIB rejection at 99% signal-efficiency operating point",
            ),
        ),
        (
            "11_balanced_accuracy_ranking",
            _ranking_plot(rows, "balanced_accuracy", "Balanced accuracy", "All models ranked by balanced accuracy"),
        ),
        (
            "12_bib_rejection_ranking",
            _ranking_plot(
                rows,
                "bib_rejection_fraction_at_99pct_signal_op",
                "Fraction of BIB rejected",
                "All models ranked by BIB rejection at 99% signal efficiency",
            ),
        ),
        ("13_operating_points_90_95_99", _operating_points_plot(rows)),
        ("14_timing_cut_retention", _timing_retention_plot(rows)),
        ("15_all_rejection_curves_full_test", _all_roc_plot(result_root, rows, False)),
        ("16_all_rejection_curves_timing_cut", _all_roc_plot(result_root, rows, True)),
        ("17_campaign_dashboard", _dashboard(rows)),
        (
            "18_parameters_vs_logic_nodes",
            _family_tau_scatter(
                rows, "trainable_parameters", "logic_node_proxy",
                "Trainable parameters", "Logic-node proxy",
                "Software parameter count versus structural logic proxy",
                y_log=True, annotate=annotate,
            ),
        ),
    )
    for stem, figure in plots:
        products.extend(_save_figure(figure, output_dir, stem, formats, dpi))

    # Backward-compatible headline artifact at the campaign root.
    products.extend(
        _save_figure(
            _dashboard(rows), result_root, "maia_lnn_screening", formats, dpi
        )
    )
    report_path = _write_report(
        result_root, rows, missing, products, expected_models=len(configs)
    )
    manifest = {
        "models_found": len(wide),
        "campaign": campaign,
        "models_expected": len(configs),
        "summary_rows": len(rows),
        "missing_evaluations": missing,
        "output_directory": str(output_dir),
        "tables": [
            str(result_root / "performance_summary.tsv"),
            str(result_root / "performance_summary.csv"),
            str(result_root / "performance_summary.json"),
            str(result_root / "model_comparison_wide.tsv"),
            str(result_root / "best_models.tsv"),
            str(report_path),
        ],
        "plots": products,
        "style": {
            "dense_model2_color": FAMILY_STYLE["dense"]["color"],
            "conv_model3_color": FAMILY_STYLE["conv"]["color"],
            "tau_markers": {str(int(key)): value for key, value in TAU_MARKERS.items()},
        },
    }
    with (result_root / "summary_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    with (result_root / "summary_complete.json").open("w") as handle:
        json.dump(
            {"models": len(wide), "rows": len(rows), "plots": len(products)},
            handle,
            indent=2,
        )
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        choices=tuple(CAMPAIGNS) + ("native_legacy",),
        default="main",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-collapsed", action="store_true")
    parser.add_argument("--no-annotations", action="store_true")
    args = parser.parse_args()
    manifest = generate_summary(
        args.result_root,
        campaign=args.campaign,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
        font_scale=args.font_scale,
        allow_partial=args.allow_partial,
        allow_collapsed=args.allow_collapsed,
        annotate=not args.no_annotations,
    )
    print(f"[DONE] Models summarized: {manifest['models_found']}/{manifest['models_expected']}")
    print(f"[DONE] Tables: {args.result_root / 'performance_summary.tsv'}")
    print(f"[DONE] Plots:  {manifest['output_directory']}")
    print(f"[DONE] Report: {args.result_root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
