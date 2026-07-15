#!/usr/bin/env python3
"""S08-only input-vs-whole fault-scope sweep helpers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.campaign import REPEATS
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    COUNTS_PER_CHUNK,
    MAX_EVALUATION_PERCENTAGE,
    MODEL_ALIAS,
    REGISTRY_NAME,
    S08_TOTAL_FAULT_SITES,
    evaluation_flip_count_chunks as whole_evaluation_flip_count_chunks,
    evaluation_flip_counts as whole_evaluation_flip_counts,
    parse_training_percentages,
    percentage_tag,
    training_configuration,
    write_rows,
)


DEFAULT_WHOLE_SCOPE_ROOT = Path(
    "results/SLURM/2026_07_14_s08_fault_aware_training_pct_sweep_5h_20k"
)
INPUT_SCOPE_NAME = "input_only"
WHOLE_SCOPE_NAME = "whole_model"
INPUT_SCOPE_DIR = "input_scope"
S08_INPUT_FAULT_SITES = 1_400


def input_evaluation_flip_counts() -> tuple[int, ...]:
    return whole_evaluation_flip_counts(total_fault_sites=S08_INPUT_FAULT_SITES)


def input_evaluation_flip_count_chunks() -> tuple[tuple[int, ...], ...]:
    nonzero_counts = input_evaluation_flip_counts()[1:]
    return tuple(
        tuple(nonzero_counts[index : index + COUNTS_PER_CHUNK])
        for index in range(0, len(nonzero_counts), COUNTS_PER_CHUNK)
    )


def input_scope_root(root: Path) -> Path:
    return root / INPUT_SCOPE_DIR


def evaluation_configuration(
    task_id: int,
    percentages: tuple[float, ...],
) -> tuple[str, str, str, str, str, str, int, tuple[int, ...]]:
    chunks = input_evaluation_flip_count_chunks()
    source_count = 1 + len(percentages)
    task_count = source_count * len(chunks)
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"Evaluation task ID must lie in [0, {task_count - 1}].")

    source_index, chunk_index = divmod(task_id, len(chunks))
    if source_index == 0:
        condition = "inference_only"
        percentage_text = "none"
        tag = percentage_tag(None)
    else:
        condition = "fault_aware_training"
        percentage = percentages[source_index - 1]
        percentage_text = f"{percentage:.12g}"
        tag = percentage_tag(percentage)
    return (
        condition,
        INPUT_SCOPE_NAME,
        tag,
        percentage_text,
        MODEL_ALIAS,
        REGISTRY_NAME,
        chunk_index,
        chunks[chunk_index],
    )


def _validate_counts(
    rows: list[dict],
    expected_counts: tuple[int, ...],
    path: Path,
) -> None:
    observed_counts = tuple(
        sorted({int(row["flip_count_per_event"]) for row in rows})
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected flip-count grid in {path}: expected {expected_counts}, "
            f"observed {observed_counts}."
        )


def _load_rows_for_tag(
    root: Path,
    tag: str,
    chunks: tuple[tuple[int, ...], ...],
    expected_counts: tuple[int, ...],
) -> list[dict]:
    from lgnStudies.bitFlipStudy.combine_results import (
        load_and_combine_model_chunks,
        read_csv,
    )

    model_root = root / tag / MODEL_ALIAS
    csv_path = model_root / "bit_flip_results.csv"
    if csv_path.is_file():
        rows = read_csv(csv_path)
    else:
        rows, _metadata, _prediction_source = load_and_combine_model_chunks(
            model_root,
            MODEL_ALIAS,
            flip_count_chunks=chunks,
            expected_flip_counts=expected_counts,
        )
    _validate_counts(rows, expected_counts, csv_path)
    return rows


def _load_scope_rows(
    root: Path,
    percentages: tuple[float, ...],
    chunks: tuple[tuple[int, ...], ...],
    expected_counts: tuple[int, ...],
) -> tuple[list[dict], dict[float, list[dict]]]:
    inference_rows = _load_rows_for_tag(
        root,
        percentage_tag(None),
        chunks,
        expected_counts,
    )
    fault_aware_rows = {
        percentage: _load_rows_for_tag(
            root,
            percentage_tag(percentage),
            chunks,
            expected_counts,
        )
        for percentage in percentages
    }
    return inference_rows, fault_aware_rows


def _combine_input_scope_rows(
    root: Path,
    percentages: tuple[float, ...],
) -> tuple[list[dict], dict[float, list[dict]]]:
    from lgnStudies.bitFlipStudy.combine_results import load_and_combine_model_chunks

    input_root = input_scope_root(root)
    chunks = input_evaluation_flip_count_chunks()
    expected_counts = input_evaluation_flip_counts()
    source_tags = [
        percentage_tag(None),
        *[percentage_tag(value) for value in percentages],
    ]

    rows_by_tag = {}
    for tag in source_tags:
        model_root = input_root / tag / MODEL_ALIAS
        rows, metadata, prediction_source = load_and_combine_model_chunks(
            model_root,
            MODEL_ALIAS,
            flip_count_chunks=chunks,
            expected_flip_counts=expected_counts,
        )
        model_root.mkdir(parents=True, exist_ok=True)
        write_rows(model_root / "bit_flip_results.csv", rows)
        metadata = {
            **metadata,
            "fault_scope": "input",
            "total_fault_sites_per_event": S08_INPUT_FAULT_SITES,
        }
        (model_root / "bit_flip_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copyfile(prediction_source, model_root / "clean_predictions.npy")
        rows_by_tag[tag] = rows

    return (
        rows_by_tag[percentage_tag(None)],
        {
            percentage: rows_by_tag[percentage_tag(percentage)]
            for percentage in percentages
        },
    )


def combine_sweep(
    root: Path,
    whole_scope_root: Path,
    percentages: tuple[float, ...],
    fit_exponential_decay: bool = False,
) -> None:
    from lgnStudies.bitFlipStudy.plots import (
        plot_fault_scope_training_percentage_comparison,
        plot_training_percentage_sweep,
    )

    input_inference_rows, input_fault_aware_rows = _combine_input_scope_rows(
        root,
        percentages,
    )
    whole_inference_rows, whole_fault_aware_rows = _load_scope_rows(
        whole_scope_root,
        percentages,
        whole_evaluation_flip_count_chunks(),
        whole_evaluation_flip_counts(),
    )

    plot_training_percentage_sweep(
        input_inference_rows,
        input_fault_aware_rows,
        root / "s08_input_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    plot_training_percentage_sweep(
        whole_inference_rows,
        whole_fault_aware_rows,
        root / "s08_whole_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    plot_fault_scope_training_percentage_comparison(
        {
            WHOLE_SCOPE_NAME: {
                "label": "Whole-model faults",
                "inference_rows": whole_inference_rows,
                "fault_aware_rows_by_percentage": whole_fault_aware_rows,
            },
            INPUT_SCOPE_NAME: {
                "label": "Input-only faults",
                "inference_rows": input_inference_rows,
                "fault_aware_rows_by_percentage": input_fault_aware_rows,
            },
        },
        root / "s08_fault_scope_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )

    if fit_exponential_decay:
        plot_training_percentage_sweep(
            input_inference_rows,
            input_fault_aware_rows,
            root / "s08_input_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )
        plot_training_percentage_sweep(
            whole_inference_rows,
            whole_fault_aware_rows,
            root / "s08_whole_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )
        plot_fault_scope_training_percentage_comparison(
            {
                WHOLE_SCOPE_NAME: {
                    "label": "Whole-model faults",
                    "inference_rows": whole_inference_rows,
                    "fault_aware_rows_by_percentage": whole_fault_aware_rows,
                },
                INPUT_SCOPE_NAME: {
                    "label": "Input-only faults",
                    "inference_rows": input_inference_rows,
                    "fault_aware_rows_by_percentage": input_fault_aware_rows,
                },
            },
            root / "s08_fault_scope_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )

    summary = {
        "model_alias": MODEL_ALIAS,
        "registry_name": REGISTRY_NAME,
        "training_fault_percentages": list(percentages),
        "maximum_evaluation_percentage": MAX_EVALUATION_PERCENTAGE,
        "repeats_per_nonzero_count": REPEATS,
        "whole_model_scope": {
            "source_root": str(whole_scope_root),
            "fault_scope": "all",
            "total_fault_sites_per_event": S08_TOTAL_FAULT_SITES,
            "evaluation_flip_counts": list(whole_evaluation_flip_counts()),
        },
        "input_only_scope": {
            "source_root": str(input_scope_root(root)),
            "fault_scope": "input",
            "total_fault_sites_per_event": S08_INPUT_FAULT_SITES,
            "evaluation_flip_counts": list(input_evaluation_flip_counts()),
        },
    }
    (root / "s08_fault_scope_sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] Combined s08 fault-scope sweep under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-percentages", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--training-task-id", type=int)
    task_group.add_argument("--evaluation-task-id", type=int)
    task_group.add_argument("--print-evaluation-task-count", action="store_true")
    task_group.add_argument("--print-training-task-count", action="store_true")
    task_group.add_argument("--print-input-evaluation-counts", action="store_true")
    task_group.add_argument("--combine", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--whole-scope-root",
        type=Path,
        default=DEFAULT_WHOLE_SCOPE_ROOT,
    )
    parser.add_argument("--fit-exponential-decay", action="store_true")
    args = parser.parse_args()

    try:
        percentages = parse_training_percentages(args.training_percentages)
        if args.training_task_id is not None:
            percentage, tag, model, registry = training_configuration(
                args.training_task_id,
                percentages,
            )
            print(
                "\t".join(
                    (f"{percentage:.12g}", INPUT_SCOPE_NAME, tag, model, registry)
                )
            )
        elif args.evaluation_task_id is not None:
            condition, scope, tag, percentage_text, model, registry, chunk, counts = (
                evaluation_configuration(args.evaluation_task_id, percentages)
            )
            print(
                "\t".join(
                    (
                        condition,
                        scope,
                        tag,
                        percentage_text,
                        model,
                        registry,
                        f"{chunk:02d}",
                        " ".join(map(str, counts)),
                    )
                )
            )
        elif args.print_training_task_count:
            print(len(percentages))
        elif args.print_evaluation_task_count:
            print((1 + len(percentages)) * len(input_evaluation_flip_count_chunks()))
        elif args.print_input_evaluation_counts:
            print(" ".join(map(str, input_evaluation_flip_counts())))
        elif args.combine:
            if args.root is None:
                parser.error("--root is required with --combine.")
            combine_sweep(
                args.root,
                args.whole_scope_root,
                percentages,
                fit_exponential_decay=args.fit_exponential_decay,
            )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
