"""Publication plots for exact-count model-2 bit-flip experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


MODEL_COLORS = ("#3366a6", "#d05a3a", "#2f8a62")
CLASS_COLORS = ("#3366a6", "#d05a3a", "#2f8a62")
AGGREGATE_METRICS = (
    "balanced_accuracy",
    "prediction_change_fraction",
    "balanced_repair_gain",
    "balanced_damage_loss",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
)
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
}


def _confidence_half_width(values: np.ndarray) -> float:
    """Two-sided 95% Student-t interval for independent fault realizations."""
    if len(values) < 2:
        return 0.0
    degrees_of_freedom = len(values) - 1
    critical = T_CRITICAL_95.get(degrees_of_freedom, 1.96)
    return float(critical * values.std(ddof=1) / np.sqrt(len(values)))


def _aggregate(rows: Iterable[dict]) -> dict[str, np.ndarray]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in AGGREGATE_METRICS}
    )
    percentages: dict[int, float] = {}
    for row in rows:
        count = int(row["flip_count_per_event"])
        for metric in AGGREGATE_METRICS:
            grouped[count][metric].append(float(row[metric]))
        percentages[count] = float(row["equivalent_fault_percentage"])

    counts = np.asarray(sorted(grouped), dtype=float)
    output: dict[str, np.ndarray] = {
        "count": counts,
        "percentage": np.asarray(
            [percentages[int(count)] for count in counts], dtype=float
        ),
    }
    for metric in AGGREGATE_METRICS:
        samples = [
            np.asarray(grouped[int(count)][metric], dtype=float)
            for count in counts
        ]
        output[f"{metric}_mean"] = np.asarray(
            [values.mean() for values in samples]
        )
        output[f"{metric}_ci95"] = np.asarray(
            [_confidence_half_width(values) for values in samples]
        )
    return output


def _count_formatter(value: float, _position: int) -> str:
    return f"{value:g}"


def _percentage_formatter(value: float, _position: int) -> str:
    if np.isclose(value, 0):
        return "0"
    return f"{value:.2g}"


def _configure_x_axis(axis, x: np.ndarray, x_mode: str) -> None:
    positive = x[x > 0]
    if x_mode == "count":
        axis.set_xscale("symlog", base=2, linthresh=0.5, linscale=0.75)
        axis.xaxis.set_major_formatter(FuncFormatter(_count_formatter))
        axis.set_xlabel("Random transient bit flips per event, K")
    elif x_mode == "percentage":
        threshold = positive.min() / 2 if len(positive) else 1e-6
        axis.set_xscale("symlog", base=10, linthresh=threshold, linscale=0.75)
        axis.xaxis.set_major_formatter(FuncFormatter(_percentage_formatter))
        axis.set_xlabel("Flipped activation sites per event [%]")
    else:
        raise ValueError(f"Unknown x-axis mode: {x_mode}")
    axis.set_xticks(x)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(True, which="both", alpha=0.25)


def _plot_metric(
    axis,
    x: np.ndarray,
    mean: np.ndarray,
    ci95: np.ndarray,
    color: str,
    label: str | None = None,
) -> None:
    mean_percent = 100 * mean
    ci_percent = 100 * ci95
    axis.plot(
        x,
        mean_percent,
        color=color,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label=label,
    )
    if np.any(ci_percent > 0):
        axis.fill_between(
            x,
            mean_percent - ci_percent,
            mean_percent + ci_percent,
            color=color,
            alpha=0.2,
            linewidth=0,
        )


def plot_single_model(
    rows: list[dict],
    model_alias: str,
    output_stem: Path,
    x_mode: str = "count",
) -> None:
    values = _aggregate(rows)
    x = values[x_mode]

    fig, (accuracy_axis, recall_axis, change_axis) = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 2.0, 1.8)},
        constrained_layout=True,
    )
    _plot_metric(
        accuracy_axis,
        x,
        values["balanced_accuracy_mean"],
        values["balanced_accuracy_ci95"],
        MODEL_COLORS[0],
    )
    accuracy_axis.set_ylabel("Balanced accuracy [%]")
    accuracy_axis.set_title(
        f"{model_alias}: inference robustness to transient bit flips\n"
        "Mean and 95% CI over random fault realizations"
    )
    accuracy_axis.grid(True, which="both", alpha=0.25)

    class_labels = (
        "positive low-pT",
        "negative low-pT",
        "high-pT",
    )
    for class_index, (class_label, color) in enumerate(
        zip(class_labels, CLASS_COLORS)
    ):
        metric = f"recall_class_{class_index}"
        _plot_metric(
            recall_axis,
            x,
            values[f"{metric}_mean"],
            values[f"{metric}_ci95"],
            color,
            label=class_label,
        )
    recall_axis.set_ylabel("Recall by true class [%]")
    recall_axis.legend(loc="best", fontsize=8)
    recall_axis.grid(True, which="both", alpha=0.25)

    _plot_metric(
        change_axis,
        x,
        values["prediction_change_fraction_mean"],
        values["prediction_change_fraction_ci95"],
        "#666666",
        label="prediction changed",
    )
    _plot_metric(
        change_axis,
        x,
        values["balanced_repair_gain_mean"],
        values["balanced_repair_gain_ci95"],
        "#2f8a62",
        label="balanced repair gain",
    )
    _plot_metric(
        change_axis,
        x,
        values["balanced_damage_loss_mean"],
        values["balanced_damage_loss_ci95"],
        "#b33a3a",
        label="balanced damage loss",
    )
    change_axis.set_ylabel("Changed / repaired / damaged [%]")
    change_axis.legend(loc="best", fontsize=8)
    _configure_x_axis(change_axis, x, x_mode)

    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def plot_combined_models(
    rows_by_model: dict[str, list[dict]],
    output_stem: Path,
    x_mode: str = "count",
) -> None:
    models = list(rows_by_model)
    if not models:
        raise ValueError("No model results supplied.")

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.7 * len(models), 4.5),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    for index, (axis, model_alias) in enumerate(zip(axes, models)):
        values = _aggregate(rows_by_model[model_alias])
        x = values[x_mode]
        _plot_metric(
            axis,
            x,
            values["balanced_accuracy_mean"],
            values["balanced_accuracy_ci95"],
            MODEL_COLORS[index % len(MODEL_COLORS)],
        )
        axis.set_title(model_alias)
        _configure_x_axis(axis, x, x_mode)
        if index == 0:
            axis.set_ylabel("Balanced accuracy [%]")

    fig.suptitle(
        "Model-2 LGN robustness to exact-count transient bit flips\n"
        "(mean and 95% CI over random fault realizations)"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)
