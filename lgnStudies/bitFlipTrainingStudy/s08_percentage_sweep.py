#!/usr/bin/env python3
"""S08-only fault-aware training percentage sweep helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.campaign import EXPECTED_FLIP_COUNTS, REPEATS


MODEL_ALIAS = "s08_p1202M_tau40"
REGISTRY_NAME = "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40"
REFERENCE_MODEL_ALIAS = "s06_p760M_tau40"
REFERENCE_TOTAL_FAULT_SITES = 77_300
S08_TOTAL_FAULT_SITES = 97_100
DEFAULT_MAX_EVALUATION_PERCENTAGE = 5.3
COUNTS_PER_CHUNK = 3


def configured_max_evaluation_percentage() -> float:
    raw = os.environ.get("S08_MAX_EVALUATION_PERCENTAGE")
    if raw is None:
        return DEFAULT_MAX_EVALUATION_PERCENTAGE
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"S08_MAX_EVALUATION_PERCENTAGE must be numeric, got {raw!r}."
        ) from error
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(
            "S08_MAX_EVALUATION_PERCENTAGE must lie in (0, 100], "
            f"got {raw!r}."
        )
    return value


MAX_EVALUATION_PERCENTAGE = configured_max_evaluation_percentage()


def validate_percentage(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Percentage must be numeric, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(f"Percentage must lie in (0, 100], got {value!r}.")
    return value


def parse_training_percentages(text: str) -> tuple[float, ...]:
    values = tuple(validate_percentage(part) for part in text.split())
    if not values:
        raise ValueError("At least one training percentage is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate training percentages are not allowed: {text}")
    return tuple(sorted(values))


def percentage_tag(percentage: float | None) -> str:
    if percentage is None:
        return "inference_only"
    formatted = f"{percentage:.12g}".replace(".", "p")
    return f"train_{formatted}pct"


def evaluation_flip_counts(
    total_fault_sites: int = S08_TOTAL_FAULT_SITES,
) -> tuple[int, ...]:
    """Map the old s06 count grid onto the s08 fault-space percentage grid."""
    counts = [0]
    for reference_count in EXPECTED_FLIP_COUNTS[1:]:
        percentage = 100 * reference_count / REFERENCE_TOTAL_FAULT_SITES
        count = int(round(total_fault_sites * percentage / 100))
        if count <= counts[-1]:
            count = counts[-1] + 1
        counts.append(count)
    counts[-1] = int(round(total_fault_sites * MAX_EVALUATION_PERCENTAGE / 100))
    return tuple(counts)


def evaluation_flip_count_chunks() -> tuple[tuple[int, ...], ...]:
    nonzero_counts = evaluation_flip_counts()[1:]
    return tuple(
        tuple(nonzero_counts[index : index + COUNTS_PER_CHUNK])
        for index in range(0, len(nonzero_counts), COUNTS_PER_CHUNK)
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def training_configuration(
    task_id: int,
    percentages: tuple[float, ...],
) -> tuple[float, str, str, str]:
    if task_id < 0 or task_id >= len(percentages):
        raise ValueError(f"Training task ID must lie in [0, {len(percentages) - 1}].")
    percentage = percentages[task_id]
    return percentage, percentage_tag(percentage), MODEL_ALIAS, REGISTRY_NAME


def evaluation_configuration(
    task_id: int,
    percentages: tuple[float, ...],
) -> tuple[str, str, str, str, str, int, tuple[int, ...]]:
    chunks = evaluation_flip_count_chunks()
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
        tag,
        percentage_text,
        MODEL_ALIAS,
        REGISTRY_NAME,
        chunk_index,
        chunks[chunk_index],
    )


def combine_sweep(
    root: Path,
    percentages: tuple[float, ...],
    fit_exponential_decay: bool = False,
) -> None:
    from lgnStudies.bitFlipStudy.combine_results import load_and_combine_model_chunks
    from lgnStudies.bitFlipStudy.plots import plot_training_percentage_sweep

    chunks = evaluation_flip_count_chunks()
    expected_counts = evaluation_flip_counts()
    source_tags = [percentage_tag(None), *[percentage_tag(value) for value in percentages]]

    rows_by_tag = {}
    for tag in source_tags:
        model_root = root / tag / MODEL_ALIAS
        rows, metadata, prediction_source = load_and_combine_model_chunks(
            model_root,
            MODEL_ALIAS,
            flip_count_chunks=chunks,
            expected_flip_counts=expected_counts,
        )
        model_root.mkdir(parents=True, exist_ok=True)
        write_rows(model_root / "bit_flip_results.csv", rows)
        (model_root / "bit_flip_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copyfile(prediction_source, model_root / "clean_predictions.npy")
        rows_by_tag[tag] = rows

    inference_rows = rows_by_tag[percentage_tag(None)]
    fault_aware_rows_by_percentage = {
        percentage: rows_by_tag[percentage_tag(percentage)]
        for percentage in percentages
    }
    plot_training_percentage_sweep(
        inference_rows,
        fault_aware_rows_by_percentage,
        root / "s08_training_percentage_sweep_accuracy",
        MODEL_ALIAS,
        fit_exponential_decay=False,
    )
    if fit_exponential_decay:
        plot_training_percentage_sweep(
            inference_rows,
            fault_aware_rows_by_percentage,
            root / "s08_training_percentage_sweep_accuracy_fit",
            MODEL_ALIAS,
            fit_exponential_decay=True,
        )

    summary = {
        "model_alias": MODEL_ALIAS,
        "registry_name": REGISTRY_NAME,
        "training_fault_percentages": list(percentages),
        "evaluation_flip_counts": list(expected_counts),
        "evaluation_flip_count_chunks": [list(chunk) for chunk in chunks],
        "maximum_evaluation_percentage": MAX_EVALUATION_PERCENTAGE,
        "total_fault_sites_per_event": S08_TOTAL_FAULT_SITES,
        "repeats_per_nonzero_count": REPEATS,
    }
    (root / "s08_percentage_sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] Combined s08 percentage sweep under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-percentages", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--training-task-id", type=int)
    task_group.add_argument("--evaluation-task-id", type=int)
    task_group.add_argument("--print-evaluation-task-count", action="store_true")
    task_group.add_argument("--print-training-task-count", action="store_true")
    task_group.add_argument("--print-evaluation-counts", action="store_true")
    task_group.add_argument("--combine", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fit-exponential-decay", action="store_true")
    args = parser.parse_args()

    try:
        percentages = parse_training_percentages(args.training_percentages)
        if args.training_task_id is not None:
            percentage, tag, model, registry = training_configuration(
                args.training_task_id,
                percentages,
            )
            print("\t".join((f"{percentage:.12g}", tag, model, registry)))
        elif args.evaluation_task_id is not None:
            condition, tag, percentage_text, model, registry, chunk, counts = (
                evaluation_configuration(args.evaluation_task_id, percentages)
            )
            print(
                "\t".join(
                    (
                        condition,
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
            print((1 + len(percentages)) * len(evaluation_flip_count_chunks()))
        elif args.print_evaluation_counts:
            print(" ".join(map(str, evaluation_flip_counts())))
        elif args.combine:
            if args.root is None:
                parser.error("--root is required with --combine.")
            combine_sweep(
                args.root,
                percentages,
                fit_exponential_decay=args.fit_exponential_decay,
            )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
