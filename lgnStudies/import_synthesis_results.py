#!/usr/bin/env python3
"""Join actual Vivado LUT counts to the training-stage accuracy trajectory."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTUAL_SOURCES = {"vivado", "vivado_fast"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def join_stage_results(
    training_rows: list[dict],
    synthesis_rows: list[dict],
    model_alias: str,
    *,
    allow_partial: bool = False,
) -> list[dict]:
    pattern = re.compile(rf"^{re.escape(model_alias)}_step(\d+)$")
    by_step = {}
    for row in synthesis_rows:
        match = pattern.match(str(row.get("model_name", "")))
        if not match or row.get("source") not in ACTUAL_SOURCES:
            continue
        if not row.get("lut"):
            continue
        by_step[int(match.group(1))] = row

    joined = []
    missing = []
    for training in training_rows:
        step = int(training["step"])
        synthesis = by_step.get(step)
        if synthesis is None:
            missing.append(step)
            continue
        joined.append(
            {
                **training,
                "vivado_lut": int(float(synthesis["lut"])),
                "vivado_ff": (
                    int(float(synthesis["ff"])) if synthesis.get("ff") else ""
                ),
                "vivado_dsp": (
                    int(float(synthesis["dsp"])) if synthesis.get("dsp") else ""
                ),
                "vivado_source": synthesis["source"],
                "vivado_clock_ns": synthesis.get("clock_ns", ""),
            }
        )

    if missing and not allow_partial:
        raise ValueError(
            "Missing actual Vivado synthesis results for training steps: "
            + ", ".join(map(str, missing))
        )
    if not joined:
        raise ValueError(f"No actual Vivado results matched {model_alias}.")
    return joined


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_actual_luts(rows: list[dict], model_alias: str, output_stem: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["step"]))
    steps = np.asarray([int(row["step"]) for row in rows])
    x = np.asarray([float(row["vivado_lut"]) / 1_000 for row in rows])
    y = 100 * np.asarray([float(row["balanced_accuracy"]) for row in rows])

    fig, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    axis.plot(x, y, color="0.55", linewidth=1.2, zorder=1)
    points = axis.scatter(x, y, c=steps, cmap="viridis", s=62, zorder=2)
    for x_value, y_value, step in zip(x, y, steps):
        axis.annotate(
            str(step),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Post-synthesis FPGA LUTs [thousands]")
    axis.set_ylabel("Balanced accuracy (macro recall) [%]")
    axis.set_title(
        f"{model_alias}: synthesized resource vs accuracy during training"
    )
    axis.grid(True, alpha=0.25)
    colorbar = fig.colorbar(points, ax=axis)
    colorbar.set_label("Training step")
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--synthesis-csv", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined = join_stage_results(
        read_csv(args.training_csv),
        read_csv(args.synthesis_csv),
        args.model_alias,
        allow_partial=args.allow_partial,
    )
    write_csv(args.output_dir / "vivado_resource_precision.csv", joined)
    plot_actual_luts(
        joined,
        args.model_alias,
        args.output_dir / "vivado_luts_vs_balanced_accuracy",
    )
    print(f"[DONE] Joined {len(joined)} synthesized stages in {args.output_dir}")


if __name__ == "__main__":
    main()
