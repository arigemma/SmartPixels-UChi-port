"""Paper-ready plots for model-2 random bit-flip robustness."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


VARIANT_LABELS = {
    "clean": "Standard training",
    "robust": "Fault-aware fine-tuning",
}
VARIANT_STYLES = {
    "clean": dict(color="#3b6fb6", marker="o"),
    "robust": dict(color="#d1495b", marker="s"),
}


def _aggregate(rows: Iterable[dict]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["variant"])][float(row["inference_flip_probability"])].append(
            float(row["balanced_accuracy"])
        )

    result = {}
    for variant, by_probability in grouped.items():
        probabilities = np.asarray(sorted(by_probability), dtype=float)
        samples = [np.asarray(by_probability[p], dtype=float) for p in probabilities]
        result[variant] = {
            "probability": probabilities,
            "mean": np.asarray([values.mean() for values in samples]),
            "std": np.asarray(
                [values.std(ddof=1) if len(values) > 1 else 0.0 for values in samples]
            ),
        }
    return result


def _probability_formatter(value: float, _position: int) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.0e}"


def _configure_probability_axis(
    axis, probabilities: np.ndarray, sparse_ticks: bool = False
) -> None:
    positive = probabilities[probabilities > 0]
    linear_threshold = positive.min() / 2 if len(positive) else 1e-7
    axis.set_xscale("symlog", linthresh=linear_threshold, linscale=0.8)
    ticks = probabilities
    if sparse_ticks:
        power_ticks = np.asarray(
            [
                value
                for value in probabilities
                if value == 0
                or np.isclose(np.log10(value), round(np.log10(value)))
            ]
        )
        positive_powers = power_ticks[power_ticks > 0]
        if len(positive_powers) > 4:
            positive_powers = np.concatenate(
                (positive_powers[::2], positive_powers[-1:])
            )
            positive_powers = np.unique(positive_powers)
        ticks = np.concatenate(([0.0], positive_powers))
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(FuncFormatter(_probability_formatter))
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(True, which="both", alpha=0.25)


def _plot_curve(axis, variant: str, values: dict, train_probability: float) -> None:
    style = VARIANT_STYLES[variant]
    label = VARIANT_LABELS[variant]
    if variant == "robust":
        exponent = np.log10(train_probability) if train_probability > 0 else None
        if exponent is not None and np.isclose(exponent, round(exponent)):
            label += rf" ($p_{{train}}=10^{{{int(round(exponent))}}}$)"
        else:
            label += rf" ($p_{{train}}={train_probability:.0e}$)"
    x = values["probability"]
    y = 100 * values["mean"]
    std = 100 * values["std"]
    axis.plot(x, y, linewidth=2.0, markersize=5.5, label=label, **style)
    if np.any(std > 0):
        axis.fill_between(
            x, y - std, y + std, color=style["color"], alpha=0.18, linewidth=0
        )


def plot_single_model(
    rows: list[dict],
    model_alias: str,
    train_probability: float,
    output_stem: Path,
) -> None:
    aggregated = _aggregate(rows)
    if not {"clean", "robust"}.issubset(aggregated):
        return

    clean = aggregated["clean"]
    robust = aggregated["robust"]
    common = np.intersect1d(clean["probability"], robust["probability"])
    if not len(common):
        return

    clean_lookup = dict(zip(clean["probability"], clean["mean"]))
    robust_lookup = dict(zip(robust["probability"], robust["mean"]))
    gain = 100 * np.asarray(
        [robust_lookup[p] - clean_lookup[p] for p in common], dtype=float
    )

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.25)},
        constrained_layout=True,
    )
    _plot_curve(top, "clean", clean, train_probability)
    _plot_curve(top, "robust", robust, train_probability)
    top.axhline(
        100 / 3,
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label="Random-class baseline",
    )
    top.set_ylabel("Balanced accuracy [%]")
    top.set_title(
        f"{model_alias}: robustness to random activation bit flips"
    )
    top.legend(loc="best", frameon=True)
    top.grid(True, which="both", alpha=0.25)

    bottom.axhline(0, color="0.4", linewidth=1.0)
    bottom.plot(common, gain, color="#6a4c93", marker="D", linewidth=1.8)
    bottom.fill_between(common, 0, gain, color="#6a4c93", alpha=0.14)
    bottom.set_ylabel("Robustness\ngain [pp]")
    bottom.set_xlabel("Independent bit-flip probability per activation")
    _configure_probability_axis(bottom, common)

    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def plot_combined_models(
    rows_by_model: dict[str, list[dict]],
    train_probability: float,
    output_stem: Path,
) -> None:
    models = list(rows_by_model)
    if not models:
        raise ValueError("No model results supplied.")

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.0 * len(models), 4.6), sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    legend_handles = legend_labels = None
    for index, (axis, model_alias) in enumerate(zip(axes, models)):
        aggregated = _aggregate(rows_by_model[model_alias])
        if not {"clean", "robust"}.issubset(aggregated):
            raise ValueError(f"Incomplete clean/robust results for {model_alias}.")
        _plot_curve(axis, "clean", aggregated["clean"], train_probability)
        _plot_curve(axis, "robust", aggregated["robust"], train_probability)
        axis.axhline(100 / 3, color="0.45", linestyle=":", linewidth=1.2)
        probabilities = aggregated["clean"]["probability"]
        _configure_probability_axis(axis, probabilities, sparse_ticks=True)
        axis.set_title(model_alias)
        axis.set_xlabel("Bit-flip probability")
        if index == 0:
            axis.set_ylabel("Balanced accuracy [%]")
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside upper center",
            ncol=2,
            frameon=True,
        )
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)
