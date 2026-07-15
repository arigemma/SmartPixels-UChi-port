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


def _append_unique_tick(ticks: list[float], value: float) -> None:
    if not any(np.isclose(value, tick, rtol=1e-9, atol=1e-12) for tick in ticks):
        ticks.append(float(value))


def _percentage_axis_ticks(x: np.ndarray) -> np.ndarray:
    """Choose percentage ticks sparse enough for readable labels."""
    x = np.asarray(x, dtype=float)
    positive = np.unique(x[x > 0])
    if len(positive) == 0:
        return np.asarray([0.0], dtype=float)
    if len(x) <= 10:
        return np.asarray(sorted(np.unique(x)), dtype=float)

    minimum = float(positive.min())
    maximum = float(positive.max())
    ticks: list[float] = [0.0]
    exponent_min = int(np.floor(np.log10(minimum)))
    exponent_max = int(np.ceil(np.log10(maximum)))
    for exponent in range(exponent_min, exponent_max + 1):
        for mantissa in (1.0, 3.0):
            candidate = mantissa * 10**exponent
            if minimum <= candidate <= maximum:
                _append_unique_tick(ticks, candidate)
    _append_unique_tick(ticks, maximum)
    return np.asarray(sorted(ticks), dtype=float)


def _configure_x_axis(axis, x: np.ndarray, x_mode: str) -> None:
    positive = x[x > 0]
    if x_mode == "count":
        axis.set_xscale("symlog", base=2, linthresh=0.5, linscale=0.75)
        axis.xaxis.set_major_formatter(FuncFormatter(_count_formatter))
        axis.set_xlabel("Random transient bit flips per event, K")
        axis.set_xticks(x)
        axis.tick_params(axis="x", labelsize=8)
    elif x_mode == "percentage":
        threshold = positive.min() / 2 if len(positive) else 1e-6
        axis.set_xscale("symlog", base=10, linthresh=threshold, linscale=0.75)
        axis.xaxis.set_major_formatter(FuncFormatter(_percentage_formatter))
        axis.set_xlabel("Flipped activation sites per event [%]")
        axis.set_xticks(_percentage_axis_ticks(x))
        axis.tick_params(axis="x", labelsize=7, labelrotation=30)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        raise ValueError(f"Unknown x-axis mode: {x_mode}")
    axis.grid(True, which="both", alpha=0.25)


def _plot_metric(
    axis,
    x: np.ndarray,
    mean: np.ndarray,
    ci95: np.ndarray,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    marker: str = "o",
    ci_alpha: float = 0.2,
) -> None:
    mean_percent = 100 * mean
    ci_percent = 100 * ci95
    axis.plot(
        x,
        mean_percent,
        color=color,
        marker=marker,
        markersize=5,
        linewidth=1.8,
        linestyle=linestyle,
        label=label,
    )
    if np.any(ci_percent > 0):
        axis.fill_between(
            x,
            mean_percent - ci_percent,
            mean_percent + ci_percent,
            color=color,
            alpha=ci_alpha,
            linewidth=0,
        )


def plot_single_model(
    rows: list[dict],
    model_alias: str,
    output_stem: Path,
    x_mode: str = "count",
    title_context: str | None = None,
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
    title_lines = [
        f"{model_alias}: inference robustness to transient bit flips",
        "Mean and 95% CI over random fault realizations",
    ]
    if title_context:
        title_lines.append(title_context)
    accuracy_axis.set_title("\n".join(title_lines))
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
    title_context: str | None = None,
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

    title_lines = [
        "Model-2 LGN robustness to exact-count transient bit flips",
        "(mean and 95% CI over random fault realizations)",
    ]
    if title_context:
        title_lines.append(title_context)
    fig.suptitle("\n".join(title_lines))
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def plot_fault_aware_training_comparison(
    inference_rows_by_model: dict[str, list[dict]],
    fault_aware_rows_by_model: dict[str, list[dict]],
    output_dir: Path,
    x_mode: str = "percentage",
    title_context: str | None = None,
) -> None:
    """Write one balanced-accuracy comparison graph for each shared model."""
    models = [
        model
        for model in inference_rows_by_model
        if model in fault_aware_rows_by_model
    ]
    if not models:
        raise ValueError("No shared models supplied for comparison.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, model_alias in enumerate(models):
        inference_values = _aggregate(inference_rows_by_model[model_alias])
        fault_aware_values = _aggregate(fault_aware_rows_by_model[model_alias])
        x = np.asarray(
            sorted(
                np.unique(
                    np.concatenate(
                        (
                            inference_values[x_mode],
                            fault_aware_values[x_mode],
                        )
                    )
                )
            ),
            dtype=float,
        )
        color = MODEL_COLORS[index % len(MODEL_COLORS)]

        fig, axis = plt.subplots(
            1,
            1,
            figsize=(7.2, 4.8),
            constrained_layout=True,
        )
        _plot_metric(
            axis,
            inference_values[x_mode],
            inference_values["balanced_accuracy_mean"],
            inference_values["balanced_accuracy_ci95"],
            color,
            label="Inference-time bit flips",
            linestyle="-",
            marker="o",
        )
        _plot_metric(
            axis,
            fault_aware_values[x_mode],
            fault_aware_values["balanced_accuracy_mean"],
            fault_aware_values["balanced_accuracy_ci95"],
            color,
            label="Fault-aware training",
            linestyle="--",
            marker="s",
            ci_alpha=0.12,
        )
        axis.set_ylabel("Balanced accuracy [%]")
        title_lines = [
            f"{model_alias}: bit-flip robustness comparison",
            "Mean and 95% CI over random fault realizations",
        ]
        if title_context:
            title_lines.append(title_context)
        axis.set_title("\n".join(title_lines))
        axis.legend(loc="best", fontsize=8)
        _configure_x_axis(axis, x, x_mode)

        output_stem = output_dir / f"bit_flip_accuracy_comparison_{model_alias}"
        for suffix in (".png", ".pdf"):
            fig.savefig(output_stem.with_suffix(suffix), dpi=300)
        plt.close(fig)


def _fit_exponential_decay(
    x: np.ndarray,
    y_percent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit y = floor + amplitude * exp(-rate * x) without SciPy."""
    x = np.asarray(x, dtype=float)
    y_percent = np.asarray(y_percent, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y_percent = y_percent[order]
    if len(x) < 4 or np.allclose(y_percent, y_percent[0]):
        return None

    y_min = float(y_percent.min())
    y_max = float(y_percent.max())
    floor_candidates = np.linspace(
        max(0.0, y_min - 15.0),
        min(y_min - 1e-3, y_max - 1e-3),
        80,
    )
    best_error = np.inf
    best_parameters: tuple[float, float, float] | None = None
    for floor in floor_candidates:
        shifted = y_percent - floor
        valid = shifted > 1e-6
        if valid.sum() < 3:
            continue
        slope, intercept = np.polyfit(x[valid], np.log(shifted[valid]), deg=1)
        rate = -float(slope)
        amplitude = float(np.exp(intercept))
        fitted = floor + amplitude * np.exp(-rate * x)
        error = float(np.mean((fitted - y_percent) ** 2))
        if error < best_error:
            best_error = error
            best_parameters = (floor, amplitude, rate)

    if best_parameters is None:
        return None
    floor, amplitude, rate = best_parameters
    x_fit = np.linspace(float(x.min()), float(x.max()), 240)
    y_fit = floor + amplitude * np.exp(-rate * x_fit)
    return x_fit, y_fit


def _percentage_axis_positions(
    x: np.ndarray,
) -> tuple[np.ndarray, callable, float]:
    x = np.asarray(x, dtype=float)
    positive = x[x > 0]
    offset = positive.min() / 2 if len(positive) else 1e-6

    def transform(values):
        return np.log10(np.asarray(values, dtype=float) + offset)

    return transform(x), transform, offset


def _configure_3d_percentage_axis(axis, x_values: np.ndarray) -> None:
    _, transform, _ = _percentage_axis_positions(x_values)
    ticks = _percentage_axis_ticks(x_values)
    axis.set_xticks(transform(ticks))
    axis.set_xticklabels([_percentage_formatter(value, 0) for value in ticks])
    axis.set_xlabel("Inference-time bit flips [%]", labelpad=10)


def _plot_training_sweep_line(
    axis,
    values: dict[str, np.ndarray],
    color: str,
    label: str,
    linestyle: str,
    marker: str,
    fit_exponential_decay: bool,
) -> None:
    x = values["percentage"]
    y = 100 * values["balanced_accuracy_mean"]
    ci = 100 * values["balanced_accuracy_ci95"]
    axis.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker=marker,
        markersize=4.5,
        label=label,
    )
    if np.any(ci > 0):
        axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.08, linewidth=0)
    if fit_exponential_decay:
        fit = _fit_exponential_decay(x, y)
        if fit is not None:
            axis.plot(
                fit[0],
                fit[1],
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.9,
            )


def plot_training_percentage_sweep(
    inference_rows: list[dict],
    fault_aware_rows_by_percentage: dict[float, list[dict]],
    output_stem: Path,
    model_alias: str,
    fit_exponential_decay: bool = False,
) -> None:
    """Plot accuracy curves across training and inference fault percentages."""
    inference_values = _aggregate(inference_rows)
    training_items = sorted(fault_aware_rows_by_percentage.items())
    if not training_items:
        raise ValueError("No fault-aware training curves supplied.")
    training_values = {
        percentage: _aggregate(rows) for percentage, rows in training_items
    }
    all_x = np.unique(
        np.concatenate(
            [
                inference_values["percentage"],
                *[values["percentage"] for values in training_values.values()],
            ]
        )
    )

    colors = plt.cm.viridis(np.linspace(0.18, 0.85, len(training_items)))
    fig, axis = plt.subplots(figsize=(8.6, 5.3), constrained_layout=True)
    _plot_training_sweep_line(
        axis,
        inference_values,
        color="#333333",
        label="Inference-only training",
        linestyle="-",
        marker="o",
        fit_exponential_decay=fit_exponential_decay,
    )
    for color, (percentage, values) in zip(colors, training_values.items()):
        _plot_training_sweep_line(
            axis,
            values,
            color=color,
            label=f"Fault-aware training {percentage:g}%",
            linestyle="--",
            marker="s",
            fit_exponential_decay=fit_exponential_decay,
        )

    axis.set_ylabel("Balanced accuracy [%]")
    title = f"{model_alias}: robustness vs training fault percentage"
    if fit_exponential_decay:
        title += " with exponential-decay fits"
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8, ncol=2)
    _configure_x_axis(axis, all_x, "percentage")
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9.4, 6.6))
    axis3d = fig.add_subplot(111, projection="3d")
    _, transform, _ = _percentage_axis_positions(all_x)

    def plot_3d_line(
        values: dict[str, np.ndarray],
        training_percentage: float,
        color,
        label: str,
        linestyle: str,
        marker: str,
    ) -> None:
        x_raw = values["percentage"]
        x = transform(x_raw)
        y = np.full_like(x, training_percentage, dtype=float)
        z = 100 * values["balanced_accuracy_mean"]
        axis3d.plot(
            x,
            y,
            z,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=4.0,
            label=label,
        )
        if fit_exponential_decay:
            fit = _fit_exponential_decay(x_raw, z)
            if fit is not None:
                axis3d.plot(
                    transform(fit[0]),
                    np.full_like(fit[0], training_percentage, dtype=float),
                    fit[1],
                    color=color,
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                )

    plot_3d_line(
        inference_values,
        0.0,
        "#333333",
        "Inference-only training",
        "-",
        "o",
    )
    for color, (percentage, values) in zip(colors, training_values.items()):
        plot_3d_line(
            values,
            percentage,
            color,
            f"Fault-aware {percentage:g}%",
            "--",
            "s",
        )

    _configure_3d_percentage_axis(axis3d, all_x)
    axis3d.set_ylabel("Training flip probability [% per site]", labelpad=10)
    axis3d.set_zlabel("Balanced accuracy [%]", labelpad=10)
    axis3d.set_title(f"{model_alias}: 3D view of training and inference faults")
    axis3d.view_init(elev=24, azim=-58)
    axis3d.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_name(output_stem.name + "_3d").with_suffix(suffix),
            dpi=300,
        )
    plt.close(fig)


def plot_fault_scope_training_percentage_comparison(
    scope_rows: dict[str, dict],
    output_stem: Path,
    model_alias: str,
    fit_exponential_decay: bool = False,
) -> None:
    """Compare training-percentage sweeps for different fault scopes."""
    if not scope_rows:
        raise ValueError("No fault-scope rows supplied.")

    training_percentages = sorted(
        {
            float(percentage)
            for scope_data in scope_rows.values()
            for percentage in scope_data["fault_aware_rows_by_percentage"]
        }
    )
    if not training_percentages:
        raise ValueError("No fault-aware training curves supplied.")
    percentage_colors = {
        percentage: color
        for percentage, color in zip(
            training_percentages,
            plt.cm.viridis(np.linspace(0.18, 0.85, len(training_percentages))),
        )
    }
    scope_items = list(scope_rows.items())

    fig, axes = plt.subplots(
        1,
        len(scope_items),
        figsize=(6.8 * len(scope_items), 5.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(scope_items) == 1:
        axes = [axes]

    for index, (axis, (_scope_key, scope_data)) in enumerate(zip(axes, scope_items)):
        inference_values = _aggregate(scope_data["inference_rows"])
        training_values = {
            float(percentage): _aggregate(rows)
            for percentage, rows in sorted(
                scope_data["fault_aware_rows_by_percentage"].items()
            )
        }
        all_x = np.unique(
            np.concatenate(
                [
                    inference_values["percentage"],
                    *[values["percentage"] for values in training_values.values()],
                ]
            )
        )
        _plot_training_sweep_line(
            axis,
            inference_values,
            color="#333333",
            label="Inference-only training",
            linestyle="-",
            marker="o",
            fit_exponential_decay=fit_exponential_decay,
        )
        for percentage, values in training_values.items():
            _plot_training_sweep_line(
                axis,
                values,
                color=percentage_colors[percentage],
                label=f"Fault-aware training {percentage:g}%",
                linestyle="--",
                marker="s",
                fit_exponential_decay=fit_exponential_decay,
            )
        axis.set_title(scope_data.get("label", _scope_key))
        if index == 0:
            axis.set_ylabel("Balanced accuracy [%]")
        axis.legend(loc="best", fontsize=7)
        _configure_x_axis(axis, all_x, "percentage")

    title = f"{model_alias}: input-only vs whole-model fault scope"
    if fit_exponential_decay:
        title += " with exponential-decay fits"
    fig.suptitle(title)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6.9 * len(scope_items), 6.3))

    for index, (_scope_key, scope_data) in enumerate(scope_items, start=1):
        axis3d = fig.add_subplot(1, len(scope_items), index, projection="3d")
        inference_values = _aggregate(scope_data["inference_rows"])
        training_values = {
            float(percentage): _aggregate(rows)
            for percentage, rows in sorted(
                scope_data["fault_aware_rows_by_percentage"].items()
            )
        }
        all_x = np.unique(
            np.concatenate(
                [
                    inference_values["percentage"],
                    *[values["percentage"] for values in training_values.values()],
                ]
            )
        )
        _, transform, _ = _percentage_axis_positions(all_x)

        def plot_3d_line(
            values: dict[str, np.ndarray],
            training_percentage: float,
            color,
            label: str,
            linestyle: str,
            marker: str,
        ) -> None:
            x_raw = values["percentage"]
            x = transform(x_raw)
            y = np.full_like(x, training_percentage, dtype=float)
            z = 100 * values["balanced_accuracy_mean"]
            axis3d.plot(
                x,
                y,
                z,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markersize=4.0,
                label=label,
            )
            if fit_exponential_decay:
                fit = _fit_exponential_decay(x_raw, z)
                if fit is not None:
                    axis3d.plot(
                        transform(fit[0]),
                        np.full_like(fit[0], training_percentage, dtype=float),
                        fit[1],
                        color=color,
                        linestyle=":",
                        linewidth=1.4,
                        alpha=0.9,
                    )

        plot_3d_line(
            inference_values,
            0.0,
            "#333333",
            "Inference-only training",
            "-",
            "o",
        )
        for percentage, values in training_values.items():
            plot_3d_line(
                values,
                percentage,
                percentage_colors[percentage],
                f"Fault-aware {percentage:g}%",
                "--",
                "s",
            )

        _configure_3d_percentage_axis(axis3d, all_x)
        axis3d.set_ylabel("Training flip probability [% per site]", labelpad=10)
        axis3d.set_zlabel("Balanced accuracy [%]", labelpad=10)
        axis3d.set_title(scope_data.get("label", f"Scope {index}"))
        axis3d.view_init(elev=24, azim=-58)
        if index == 1:
            axis3d.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)

    fig.suptitle(f"{model_alias}: 3D view by fault scope")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.88, wspace=0.02)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_name(output_stem.name + "_3d").with_suffix(suffix),
            dpi=300,
        )
    plt.close(fig)
