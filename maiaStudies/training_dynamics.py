"""Plot paired relaxed/discrete validation trajectories for MAIA LNN runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .campaign import get_campaign_configs


FAMILY_LABELS = {"dense": "Model 2 (dense)", "conv": "Model 3 (convolutional)"}
SIZE_COLORS = {
    size: color
    for size, color in zip(
        ("s01", "s02", "s03", "s04", "s05"),
        ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"),
    )
}
TAU_MARKERS = {1.0: "D", 3.0: "P"}


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    key: float(value)
                    for key, value in row.items()
                    if value not in (None, "")
                }
            )
    return rows


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No training-history rows are available")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def generate_training_dynamics(
    result_root: Path,
    *,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 400,
    allow_partial: bool = False,
) -> dict:
    configs = get_campaign_configs("tau_extension")
    output_dir = output_dir or result_root / "summary_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[str, list[dict[str, float]]] = {}
    table_rows: list[dict] = []
    missing = []
    for config in configs:
        path = result_root / config.alias / "training_history.tsv"
        if not path.is_file():
            missing.append(str(path))
            continue
        history = _read_history(path)
        required = {
            "elapsed_hours",
            "discrete_balanced_accuracy",
            "relaxed_balanced_accuracy",
            "discrete_cross_entropy",
            "relaxed_cross_entropy",
        }
        if not history or not required.issubset(history[0]):
            raise ValueError(f"Incomplete relaxed/discrete history: {path}")
        histories[config.alias] = history
        for row in history:
            table_rows.append(
                {
                    "alias": config.alias,
                    "family": config.family,
                    "size": config.size,
                    "tau": config.tau,
                    **row,
                }
            )
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(configs)} training histories:\n"
            + "\n".join(missing)
        )
    if not histories:
        raise FileNotFoundError(f"No training_history.tsv files found below {result_root}")

    table_path = result_root / "relaxed_discrete_training_history.tsv"
    _write_table(table_path, table_rows)

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex="col")
    metrics = (
        ("balanced_accuracy", "Validation balanced accuracy"),
        ("cross_entropy", "Validation cross entropy"),
    )
    for row_index, family in enumerate(("dense", "conv")):
        family_configs = [config for config in configs if config.family == family]
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for config in family_configs:
                history = histories.get(config.alias)
                if not history:
                    continue
                x = [row["elapsed_hours"] for row in history]
                color = SIZE_COLORS[config.size]
                marker = TAU_MARKERS[config.tau]
                for mode, linestyle, alpha in (
                    ("discrete", "-", 0.95),
                    ("relaxed", "--", 0.70),
                ):
                    y = [row[f"{mode}_{metric}"] for row in history]
                    axis.plot(x, y, color=color, linestyle=linestyle, alpha=alpha)
                    axis.scatter(
                        x,
                        y,
                        color=color,
                        marker=marker,
                        s=25,
                        alpha=alpha,
                        edgecolors="white",
                        linewidths=0.35,
                    )
            axis.set_ylabel(label)
            axis.set_title(f"{FAMILY_LABELS[family]} — {label}")
            axis.grid(True, alpha=0.22)
            if row_index == 1:
                axis.set_xlabel("Elapsed training time [hours]")

    size_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=size)
        for size, color in SIZE_COLORS.items()
    ]
    tau_handles = [
        Line2D(
            [0], [0], marker=marker, color="#333333", linestyle="none",
            markersize=7, label=f"tau={int(tau)}",
        )
        for tau, marker in TAU_MARKERS.items()
    ]
    mode_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", label="Discrete"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Relaxed"),
    ]
    figure.legend(
        handles=size_handles + tau_handles + mode_handles,
        loc="upper center",
        ncol=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.suptitle("MAIA tau=1,3 relaxed versus discrete validation trajectories", y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    written = []
    for extension in formats:
        output = output_dir / f"19_relaxed_vs_discrete_training.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        figure.savefig(output, **kwargs)
        written.append(str(output))
    plt.close(figure)
    return {
        "models_found": len(histories),
        "models_expected": len(configs),
        "missing_histories": missing,
        "table": str(table_path),
        "plots": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = generate_training_dynamics(
        args.result_root,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
        allow_partial=args.allow_partial,
    )
    print(
        f"[DONE] Relaxed/discrete histories: "
        f"{result['models_found']}/{result['models_expected']}"
    )
    print(f"[DONE] Table: {result['table']}")
    for path in result["plots"]:
        print(f"[DONE] Plot:  {path}")


if __name__ == "__main__":
    main()
