#!/usr/bin/env python3
"""Merge completed S08 fault-scope sweeps and redraw input-vs-whole plots."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.campaign import REPEATS
from lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep import (
    INPUT_SCOPE_DIR,
    INPUT_SCOPE_NAME,
    S08_INPUT_FAULT_SITES,
    WHOLE_SCOPE_NAME,
    input_evaluation_flip_count_chunks,
    input_evaluation_flip_counts,
)
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MAX_EVALUATION_PERCENTAGE,
    MODEL_ALIAS,
    REGISTRY_NAME,
    S08_TOTAL_FAULT_SITES,
    evaluation_flip_count_chunks as whole_evaluation_flip_count_chunks,
    evaluation_flip_counts as whole_evaluation_flip_counts,
    parse_training_percentages,
    percentage_tag,
    write_rows,
)


def _load_or_combine_rows(
    root: Path,
    tag: str,
    chunks: tuple[tuple[int, ...], ...],
    expected_counts: tuple[int, ...],
    scope_prefix: Path,
) -> list[dict]:
    from lgnStudies.bitFlipStudy.combine_results import (
        load_and_combine_model_chunks,
        read_csv,
    )

    model_root = root / scope_prefix / tag / MODEL_ALIAS
    csv_path = model_root / "bit_flip_results.csv"
    if csv_path.is_file():
        rows = read_csv(csv_path)
    else:
        rows, metadata, prediction_source = load_and_combine_model_chunks(
            model_root,
            MODEL_ALIAS,
            flip_count_chunks=chunks,
            expected_flip_counts=expected_counts,
        )
        model_root.mkdir(parents=True, exist_ok=True)
        write_rows(csv_path, rows)
        (model_root / "bit_flip_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copyfile(prediction_source, model_root / "clean_predictions.npy")

    observed_counts = tuple(
        sorted({int(row["flip_count_per_event"]) for row in rows})
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected flip-count grid in {csv_path}: expected "
            f"{expected_counts}, observed {observed_counts}."
        )
    return rows


def _load_rows_by_percentage(
    base_root: Path,
    extension_root: Path,
    base_percentages: tuple[float, ...],
    extension_percentages: tuple[float, ...],
    chunks: tuple[tuple[int, ...], ...],
    expected_counts: tuple[int, ...],
    scope_prefix: Path,
) -> tuple[list[dict], dict[float, list[dict]]]:
    inference_rows = _load_or_combine_rows(
        base_root,
        percentage_tag(None),
        chunks,
        expected_counts,
        scope_prefix,
    )
    rows_by_percentage = {
        percentage: _load_or_combine_rows(
            base_root,
            percentage_tag(percentage),
            chunks,
            expected_counts,
            scope_prefix,
        )
        for percentage in base_percentages
    }
    rows_by_percentage.update(
        {
            percentage: _load_or_combine_rows(
                extension_root,
                percentage_tag(percentage),
                chunks,
                expected_counts,
                scope_prefix,
            )
            for percentage in extension_percentages
        }
    )
    return inference_rows, dict(sorted(rows_by_percentage.items()))


def merge_and_plot(
    base_whole_root: Path,
    base_input_root: Path,
    extension_whole_root: Path,
    extension_input_root: Path,
    output_root: Path,
    base_percentages: tuple[float, ...],
    extension_percentages: tuple[float, ...],
    fit_exponential_decay: bool,
) -> None:
    from lgnStudies.bitFlipStudy.plots import (
        plot_fault_scope_training_percentage_comparison,
        plot_training_percentage_sweep,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    whole_inference_rows, whole_fault_rows = _load_rows_by_percentage(
        base_whole_root,
        extension_whole_root,
        base_percentages,
        extension_percentages,
        whole_evaluation_flip_count_chunks(),
        whole_evaluation_flip_counts(),
        Path("."),
    )
    input_inference_rows, input_fault_rows = _load_rows_by_percentage(
        base_input_root,
        extension_input_root,
        base_percentages,
        extension_percentages,
        input_evaluation_flip_count_chunks(),
        input_evaluation_flip_counts(),
        Path(INPUT_SCOPE_DIR),
    )

    plot_training_percentage_sweep(
        input_inference_rows,
        input_fault_rows,
        output_root / "s08_input_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    plot_training_percentage_sweep(
        whole_inference_rows,
        whole_fault_rows,
        output_root / "s08_whole_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    plot_fault_scope_training_percentage_comparison(
        {
            WHOLE_SCOPE_NAME: {
                "label": "Whole-model faults",
                "inference_rows": whole_inference_rows,
                "fault_aware_rows_by_percentage": whole_fault_rows,
            },
            INPUT_SCOPE_NAME: {
                "label": "Input-only faults",
                "inference_rows": input_inference_rows,
                "fault_aware_rows_by_percentage": input_fault_rows,
            },
        },
        output_root / "s08_fault_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    if fit_exponential_decay:
        plot_training_percentage_sweep(
            input_inference_rows,
            input_fault_rows,
            output_root / "s08_input_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )
        plot_training_percentage_sweep(
            whole_inference_rows,
            whole_fault_rows,
            output_root / "s08_whole_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )
        plot_fault_scope_training_percentage_comparison(
            {
                WHOLE_SCOPE_NAME: {
                    "label": "Whole-model faults",
                    "inference_rows": whole_inference_rows,
                    "fault_aware_rows_by_percentage": whole_fault_rows,
                },
                INPUT_SCOPE_NAME: {
                    "label": "Input-only faults",
                    "inference_rows": input_inference_rows,
                    "fault_aware_rows_by_percentage": input_fault_rows,
                },
            },
            output_root / "s08_fault_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )

    summary = {
        "study": "s08_input_vs_whole_fault_scope_extension_merge",
        "model_alias": MODEL_ALIAS,
        "registry_name": REGISTRY_NAME,
        "base_training_fault_percentages": list(base_percentages),
        "extension_training_fault_percentages": list(extension_percentages),
        "merged_training_fault_percentages": [
            *base_percentages,
            *extension_percentages,
        ],
        "maximum_evaluation_percentage": MAX_EVALUATION_PERCENTAGE,
        "repeats_per_nonzero_count": REPEATS,
        "base_whole_root": str(base_whole_root),
        "base_input_root": str(base_input_root),
        "extension_whole_root": str(extension_whole_root),
        "extension_input_root": str(extension_input_root),
        "whole_model_scope": {
            "fault_scope": "all",
            "total_fault_sites_per_event": S08_TOTAL_FAULT_SITES,
            "evaluation_flip_counts": list(whole_evaluation_flip_counts()),
        },
        "input_only_scope": {
            "fault_scope": "input",
            "total_fault_sites_per_event": S08_INPUT_FAULT_SITES,
            "evaluation_flip_counts": list(input_evaluation_flip_counts()),
        },
    }
    (output_root / "s08_fault_scope_extension_merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] Wrote merged S08 fault-scope plots under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-whole-root", type=Path, required=True)
    parser.add_argument("--base-input-root", type=Path, required=True)
    parser.add_argument("--extension-whole-root", type=Path, required=True)
    parser.add_argument("--extension-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-training-percentages", required=True)
    parser.add_argument("--extension-training-percentages", required=True)
    parser.add_argument("--fit-exponential-decay", action="store_true")
    args = parser.parse_args()

    try:
        merge_and_plot(
            args.base_whole_root,
            args.base_input_root,
            args.extension_whole_root,
            args.extension_input_root,
            args.output_root,
            parse_training_percentages(args.base_training_percentages),
            parse_training_percentages(args.extension_training_percentages),
            args.fit_exponential_decay,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
