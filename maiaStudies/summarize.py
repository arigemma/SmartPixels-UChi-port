"""Combine all MAIA evaluations into tables and screening plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .campaign import CAMPAIGN_CONFIGS


def _flatten(config, result, subset: str) -> dict:
    metrics = result[subset]
    op90 = metrics["validation_derived_operating_points"]["signal_efficiency_90pct"]
    return {
        "task_index": config.task_index,
        "alias": config.alias,
        "family": config.family,
        "size": config.size,
        "tau": config.tau,
        "subset": subset,
        "balanced_accuracy": metrics["argmax"]["balanced_accuracy"],
        "signal_efficiency_argmax": metrics["argmax"]["signal_efficiency"],
        "bib_rejection_argmax": metrics["argmax"]["bib_rejection"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "signal_efficiency_at_validation_90pct_op": op90["signal_efficiency"],
        "bib_acceptance_at_validation_90pct_op": op90["bib_acceptance"],
        "bib_rejection_at_validation_90pct_op": op90["bib_rejection"],
        "trainable_parameters": result["trainable_parameters"],
        "logic_node_proxy": result["logic_node_proxy"],
        "group_sum_inputs": result["group_sum_inputs"],
        "thermometer_input_bits": result["thermometer_input_bits"],
        "checkpoint": result["checkpoint"],
    }


def _write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], output: Path) -> None:
    full = [row for row in rows if row["subset"] == "full_test"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    colors = {10.0: "#0072B2", 20.0: "#D55E00", 40.0: "#009E73"}
    markers = {"dense": "o", "conv": "s"}
    for family in ("dense", "conv"):
        for tau in (10.0, 20.0, 40.0):
            selected = sorted(
                (row for row in full if row["family"] == family and row["tau"] == tau),
                key=lambda row: row["logic_node_proxy"],
            )
            label = f"{family.capitalize()}, tau={int(tau)}"
            x = [row["logic_node_proxy"] for row in selected]
            axes[0].plot(
                x,
                [row["balanced_accuracy"] for row in selected],
                marker=markers[family], color=colors[tau], linestyle="-" if family == "dense" else "--",
                label=label,
            )
            rejection = [row["bib_rejection_at_validation_90pct_op"] for row in selected]
            finite = [(xx, yy) for xx, yy in zip(x, rejection) if yy is not None]
            if finite:
                axes[1].plot(
                    [item[0] for item in finite], [item[1] for item in finite],
                    marker=markers[family], color=colors[tau],
                    linestyle="-" if family == "dense" else "--", label=label,
                )
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].set_ylabel("BIB rejection at 90% validation signal-efficiency OP")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Logic-node proxy (not post-synthesis LUT+FF)")
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    missing = []
    for config in CAMPAIGN_CONFIGS:
        path = args.result_root / config.alias / "evaluation_complete.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open() as handle:
            result = json.load(handle)
        rows.append(_flatten(config, result, "full_test"))
        rows.append(_flatten(config, result, "timing_cut_test"))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of 30 evaluations:\n" + "\n".join(missing)
        )
    _write_table(args.result_root / "performance_summary.tsv", rows)
    _plot(rows, args.result_root / "maia_lnn_screening")
    with (args.result_root / "summary_complete.json").open("w") as handle:
        json.dump({"models": 30, "rows": 60}, handle, indent=2)
        handle.write("\n")
    print(f"[DONE] {args.result_root / 'performance_summary.tsv'}")
    print(f"[DONE] {args.result_root / 'maia_lnn_screening.png'}")


if __name__ == "__main__":
    main()
