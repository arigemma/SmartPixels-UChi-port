#!/usr/bin/env python3
"""QKeras input-vs-whole activation-bit fault-scope sweep helpers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.campaign import EXPECTED_FLIP_COUNTS, REPEATS
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MAX_EVALUATION_PERCENTAGE,
)
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    percentage_tag as lgn_percentage_tag,
)
from qkerasStudies.fault_injection import stage_metadata
from qkerasStudies.models import MODEL_CONFIGS, QKerasModelConfig


COUNTS_PER_CHUNK = 3
REFERENCE_TOTAL_FAULT_SITES = 77_300
INPUT_SCOPE_NAME = "input_only"
WHOLE_SCOPE_NAME = "whole_model"
INPUT_SCOPE_DIR = "input_scope"
WHOLE_SCOPE_DIR = "whole_scope"
CLEAN_DIR = "clean_training"
SCOPE_LAYOUT = (
    (WHOLE_SCOPE_NAME, "all", WHOLE_SCOPE_DIR, "Whole-model activation-bit faults"),
    (INPUT_SCOPE_NAME, "input", INPUT_SCOPE_DIR, "Input activation-bit faults"),
)


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
    return lgn_percentage_tag(percentage)


def _target_percentages() -> tuple[float, ...]:
    percentages = [
        100 * count / REFERENCE_TOTAL_FAULT_SITES
        for count in EXPECTED_FLIP_COUNTS[1:]
    ]
    percentages[-1] = MAX_EVALUATION_PERCENTAGE
    return tuple(percentages)


def evaluation_flip_counts(
    config: QKerasModelConfig,
    fault_scope: str,
) -> tuple[int, ...]:
    """Map the LGN S08 percentage grid onto a QKeras activation-bit space."""
    total_sites = int(stage_metadata(config, fault_scope)["total_fault_sites_per_event"])
    max_count = max(1, int(round(total_sites * MAX_EVALUATION_PERCENTAGE / 100)))
    counts = [0]
    for percentage in _target_percentages():
        count = int(round(total_sites * percentage / 100))
        count = min(max_count, max(1, count))
        if count not in counts:
            counts.append(count)
    if counts[-1] != max_count:
        counts.append(max_count)
    return tuple(sorted(counts))


def evaluation_flip_count_chunks(
    config: QKerasModelConfig,
    fault_scope: str,
) -> tuple[tuple[int, ...], ...]:
    nonzero_counts = evaluation_flip_counts(config, fault_scope)[1:]
    return tuple(
        tuple(nonzero_counts[index : index + COUNTS_PER_CHUNK])
        for index in range(0, len(nonzero_counts), COUNTS_PER_CHUNK)
    )


def training_tasks(percentages: tuple[float, ...]) -> list[dict]:
    tasks: list[dict] = []
    for config in MODEL_CONFIGS:
        tasks.append(
            {
                "condition": "inference_only",
                "fault_scope_key": "clean",
                "fault_scope": "all",
                "scope_dir": CLEAN_DIR,
                "tag": percentage_tag(None),
                "percentage": None,
                "model_alias": config.alias,
                "registry_name": config.registry_name,
            }
        )
    for scope_key, fault_scope, scope_dir, _label in SCOPE_LAYOUT:
        for percentage in percentages:
            for config in MODEL_CONFIGS:
                tasks.append(
                    {
                        "condition": "fault_aware_training",
                        "fault_scope_key": scope_key,
                        "fault_scope": fault_scope,
                        "scope_dir": scope_dir,
                        "tag": percentage_tag(percentage),
                        "percentage": percentage,
                        "model_alias": config.alias,
                        "registry_name": config.registry_name,
                    }
                )
    return tasks


def evaluation_tasks(percentages: tuple[float, ...]) -> list[dict]:
    tasks: list[dict] = []
    source_percentages: tuple[float | None, ...] = (None, *percentages)
    for config in MODEL_CONFIGS:
        for scope_key, fault_scope, scope_dir, _label in SCOPE_LAYOUT:
            chunks = evaluation_flip_count_chunks(config, fault_scope)
            for percentage in source_percentages:
                condition = (
                    "inference_only"
                    if percentage is None
                    else "fault_aware_training"
                )
                tag = percentage_tag(percentage)
                for chunk_index, counts in enumerate(chunks):
                    tasks.append(
                        {
                            "condition": condition,
                            "fault_scope_key": scope_key,
                            "fault_scope": fault_scope,
                            "scope_dir": scope_dir,
                            "tag": tag,
                            "percentage": percentage,
                            "model_alias": config.alias,
                            "registry_name": config.registry_name,
                            "chunk_index": chunk_index,
                            "flip_counts": counts,
                        }
                    )
    return tasks


def training_configuration(
    task_id: int,
    percentages: tuple[float, ...],
) -> dict:
    tasks = training_tasks(percentages)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Training task ID must lie in [0, {len(tasks) - 1}].")
    return tasks[task_id]


def evaluation_configuration(
    task_id: int,
    percentages: tuple[float, ...],
) -> dict:
    tasks = evaluation_tasks(percentages)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Evaluation task ID must lie in [0, {len(tasks) - 1}].")
    return tasks[task_id]


def _model_root(root: Path, scope_dir: str, tag: str, model_alias: str) -> Path:
    return root / scope_dir / tag / model_alias


def _load_scope_rows(
    root: Path,
    config: QKerasModelConfig,
    scope_dir: str,
    fault_scope: str,
    percentages: tuple[float, ...],
) -> tuple[list[dict], dict[float, list[dict]]]:
    from qkerasStudies.bitFlipStudy.combine_results import write_combined_artifacts

    chunks = evaluation_flip_count_chunks(config, fault_scope)
    expected_counts = evaluation_flip_counts(config, fault_scope)
    source_tags = [
        percentage_tag(None),
        *[percentage_tag(value) for value in percentages],
    ]
    rows_by_tag = {}
    for tag in source_tags:
        model_root = _model_root(root, scope_dir, tag, config.alias)
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


def combine_sweep(
    root: Path,
    percentages: tuple[float, ...],
    fit_exponential_decay: bool = False,
) -> None:
    from lgnStudies.bitFlipStudy.plots import (
        plot_fault_scope_training_percentage_comparison,
        plot_training_percentage_sweep,
    )

    summary_models = []
    for config in MODEL_CONFIGS:
        whole_inference_rows, whole_fault_aware_rows = _load_scope_rows(
            root,
            config,
            WHOLE_SCOPE_DIR,
            "all",
            percentages,
        )
        input_inference_rows, input_fault_aware_rows = _load_scope_rows(
            root,
            config,
            INPUT_SCOPE_DIR,
            "input",
            percentages,
        )

        plot_training_percentage_sweep(
            input_inference_rows,
            input_fault_aware_rows,
            root / f"{config.alias}_input_scope_training_percentage_sweep_accuracy",
            config.alias,
            fit_exponential_decay=False,
        )
        plot_training_percentage_sweep(
            whole_inference_rows,
            whole_fault_aware_rows,
            root / f"{config.alias}_whole_scope_training_percentage_sweep_accuracy",
            config.alias,
            fit_exponential_decay=False,
        )
        plot_fault_scope_training_percentage_comparison(
            {
                WHOLE_SCOPE_NAME: {
                    "label": "Whole-model activation-bit faults",
                    "inference_rows": whole_inference_rows,
                    "fault_aware_rows_by_percentage": whole_fault_aware_rows,
                },
                INPUT_SCOPE_NAME: {
                    "label": "Input activation-bit faults",
                    "inference_rows": input_inference_rows,
                    "fault_aware_rows_by_percentage": input_fault_aware_rows,
                },
            },
            root / f"{config.alias}_fault_scope_training_percentage_sweep_accuracy",
            config.alias,
            fit_exponential_decay=False,
        )

        if fit_exponential_decay:
            plot_training_percentage_sweep(
                input_inference_rows,
                input_fault_aware_rows,
                root / f"{config.alias}_input_scope_training_percentage_sweep_accuracy_fit",
                config.alias,
                fit_exponential_decay=True,
            )
            plot_training_percentage_sweep(
                whole_inference_rows,
                whole_fault_aware_rows,
                root / f"{config.alias}_whole_scope_training_percentage_sweep_accuracy_fit",
                config.alias,
                fit_exponential_decay=True,
            )
            plot_fault_scope_training_percentage_comparison(
                {
                    WHOLE_SCOPE_NAME: {
                        "label": "Whole-model activation-bit faults",
                        "inference_rows": whole_inference_rows,
                        "fault_aware_rows_by_percentage": whole_fault_aware_rows,
                    },
                    INPUT_SCOPE_NAME: {
                        "label": "Input activation-bit faults",
                        "inference_rows": input_inference_rows,
                        "fault_aware_rows_by_percentage": input_fault_aware_rows,
                    },
                },
                root / f"{config.alias}_fault_scope_training_percentage_sweep_accuracy_fit",
                config.alias,
                fit_exponential_decay=True,
            )

        summary_models.append(
            {
                "model_alias": config.alias,
                "registry_name": config.registry_name,
                "weight_bits": config.weight_bits,
                "activation_bits": config.activation_bits,
                "hidden_units": list(config.hidden_units),
                "whole_model_scope": {
                    "fault_scope": "all",
                    "total_fault_sites_per_event": stage_metadata(
                        config,
                        "all",
                    )["total_fault_sites_per_event"],
                    "evaluation_flip_counts": list(
                        evaluation_flip_counts(config, "all")
                    ),
                },
                "input_only_scope": {
                    "fault_scope": "input",
                    "total_fault_sites_per_event": stage_metadata(
                        config,
                        "input",
                    )["total_fault_sites_per_event"],
                    "evaluation_flip_counts": list(
                        evaluation_flip_counts(config, "input")
                    ),
                },
            }
        )

    summary = {
        "study": "qkeras_input_vs_whole_activation_bit_fault_scope_sweep",
        "training_fault_percentages": list(percentages),
        "maximum_evaluation_percentage": MAX_EVALUATION_PERCENTAGE,
        "repeats_per_nonzero_count": REPEATS,
        "models": summary_models,
    }
    (root / "qkeras_fault_scope_sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] Combined QKeras fault-scope sweep under {root}")


def _format_percentage(percentage: float | None) -> str:
    return "none" if percentage is None else f"{percentage:.12g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-percentages", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--training-task-id", type=int)
    task_group.add_argument("--evaluation-task-id", type=int)
    task_group.add_argument("--print-evaluation-task-count", action="store_true")
    task_group.add_argument("--print-training-task-count", action="store_true")
    task_group.add_argument("--print-model-counts", action="store_true")
    task_group.add_argument("--combine", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--fit-exponential-decay", action="store_true")
    args = parser.parse_args()

    try:
        percentages = parse_training_percentages(args.training_percentages)
        if args.training_task_id is not None:
            task = training_configuration(args.training_task_id, percentages)
            print(
                "\t".join(
                    (
                        task["condition"],
                        task["fault_scope_key"],
                        task["fault_scope"],
                        task["scope_dir"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        task["model_alias"],
                        task["registry_name"],
                    )
                )
            )
        elif args.evaluation_task_id is not None:
            task = evaluation_configuration(args.evaluation_task_id, percentages)
            print(
                "\t".join(
                    (
                        task["condition"],
                        task["fault_scope_key"],
                        task["fault_scope"],
                        task["scope_dir"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        task["model_alias"],
                        task["registry_name"],
                        f"{task['chunk_index']:02d}",
                        " ".join(map(str, task["flip_counts"])),
                    )
                )
            )
        elif args.print_training_task_count:
            print(len(training_tasks(percentages)))
        elif args.print_evaluation_task_count:
            print(len(evaluation_tasks(percentages)))
        elif args.print_model_counts:
            for config in MODEL_CONFIGS:
                for scope_key, fault_scope, _scope_dir, _label in SCOPE_LAYOUT:
                    print(
                        "\t".join(
                            (
                                config.alias,
                                scope_key,
                                str(
                                    stage_metadata(config, fault_scope)[
                                        "total_fault_sites_per_event"
                                    ]
                                ),
                                " ".join(
                                    map(str, evaluation_flip_counts(config, fault_scope))
                                ),
                            )
                        )
                    )
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
