#!/usr/bin/env python3
"""Write final QKeras input-vs-whole fault-scope plots into temp/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/smartpixels-matplotlib")

from lgnStudies.bitFlipStudy.plots import plot_training_percentage_sweep_2d
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MAX_EVALUATION_PERCENTAGE,
)
from lgnStudies.bitFlipStudy.campaign import REPEATS
from qkerasStudies.bitFlipStudy.combine_results import write_combined_artifacts
from qkerasStudies.bitFlipTrainingStudy.fault_scope_sweep import (
    INPUT_SCOPE_DIR,
    WHOLE_SCOPE_DIR,
    evaluation_flip_count_chunks,
    evaluation_flip_counts,
    parse_model_selection,
    parse_training_percentages,
    percentage_tag,
)


def _load_rows(
    root: Path,
    model_alias: str,
    scope_dir: str,
    fault_scope: str,
    percentages: tuple[float, ...],
) -> tuple[list[dict], dict[float, list[dict]]]:
    config = parse_model_selection(model_alias)[0]
    chunks = evaluation_flip_count_chunks(config, fault_scope)
    expected_counts = evaluation_flip_counts(config, fault_scope)
    tags = [percentage_tag(None), *[percentage_tag(value) for value in percentages]]

    rows_by_tag = {}
    for tag in tags:
        model_root = root / scope_dir / tag / config.alias
        rows_by_tag[tag] = write_combined_artifacts(
            model_root,
            config.alias,
            chunks,
            expected_counts,
            REPEATS,
        )
    return (
        rows_by_tag[percentage_tag(None)],
        {
            percentage: rows_by_tag[percentage_tag(percentage)]
            for percentage in percentages
        },
    )


def _write_model_plots(
    root: Path,
    output_dir: Path,
    model_alias: str,
    percentages: tuple[float, ...],
    max_inference_percentage: float,
) -> dict:
    whole_inference_rows, whole_fault_rows = _load_rows(
        root,
        model_alias,
        WHOLE_SCOPE_DIR,
        "all",
        percentages,
    )
    input_inference_rows, input_fault_rows = _load_rows(
        root,
        model_alias,
        INPUT_SCOPE_DIR,
        "input",
        percentages,
    )

    whole_stem = output_dir / f"{model_alias}_internal_bit_flips_xaxis_extended"
    input_stem = output_dir / f"{model_alias}_input_bit_flips_xaxis_extended"
    plot_training_percentage_sweep_2d(
        whole_inference_rows,
        whole_fault_rows,
        whole_stem,
        model_alias,
        title=(
            f"{model_alias}: internal activation-bit flips, x-axis extended "
            f"to {max_inference_percentage:g}%"
        ),
        x_limit=max_inference_percentage,
    )
    plot_training_percentage_sweep_2d(
        input_inference_rows,
        input_fault_rows,
        input_stem,
        model_alias,
        title=(
            f"{model_alias}: input activation-bit flips, x-axis extended "
            f"to {max_inference_percentage:g}%"
        ),
        x_limit=max_inference_percentage,
    )
    return {
        "model_alias": model_alias,
        "training_percentages": list(percentages),
        "maximum_evaluation_percentage": MAX_EVALUATION_PERCENTAGE,
        "whole_model_counts": sorted(
            {int(row["flip_count_per_event"]) for row in whole_inference_rows}
        ),
        "input_counts": sorted(
            {int(row["flip_count_per_event"]) for row in input_inference_rows}
        ),
        "plots": [
            str(whole_stem.with_suffix(".png")),
            str(input_stem.with_suffix(".png")),
            str(whole_stem.with_suffix(".pdf")),
            str(input_stem.with_suffix(".pdf")),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine existing QKeras fault-scope chunks and write only the two "
            "final 2D plots into a chosen output directory."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--training-percentages", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("temp"))
    parser.add_argument(
        "--max-inference-percentage",
        type=float,
        default=MAX_EVALUATION_PERCENTAGE,
    )
    args = parser.parse_args()

    percentages = parse_training_percentages(args.training_percentages)
    configs = parse_model_selection(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        _write_model_plots(
            args.root,
            args.output_dir,
            config.alias,
            percentages,
            args.max_inference_percentage,
        )
        for config in configs
    ]
    summary = {
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "models": summaries,
    }
    (args.output_dir / "qkeras_final_fault_scope_plot_sources.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("[DONE] Wrote final QKeras plots:")
    for model_summary in summaries:
        for path in model_summary["plots"]:
            print(path)
    print(args.output_dir / "qkeras_final_fault_scope_plot_sources.json")


if __name__ == "__main__":
    main()
