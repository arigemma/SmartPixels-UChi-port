#!/usr/bin/env python3
"""Task map and result combining for the selected input-bit-flip study."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path

import numpy as np

from lgnStudies.bitFlipStudy.campaign import REPEATS
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import write_rows
from qkerasStudies.fault_injection import stage_metadata
from qkerasStudies.models import get_config


TRAINING_PERCENTAGES = tuple(float(value) for value in range(1, 11))
EVALUATION_PERCENTAGES = tuple(float(value) for value in range(0, 61, 5))
REFINEMENT_PERCENTAGES = (2.5, 7.5)
REFINED_EVALUATION_PERCENTAGES = tuple(
    sorted((*EVALUATION_PERCENTAGES, *REFINEMENT_PERCENTAGES))
)
PAPER_REFINEMENT_PERCENTAGES = (1.25, 3.75, 12.5, 17.5, 22.5)
PAPER_REFINED_EVALUATION_PERCENTAGES = tuple(
    sorted(
        (
            *EVALUATION_PERCENTAGES,
            *REFINEMENT_PERCENTAGES,
            *PAPER_REFINEMENT_PERCENTAGES,
        )
    )
)
COUNTS_PER_CHUNK = 3
REFINEMENT_DIRECTORY = "refinement_0to10pct"
PAPER_REFINEMENT_DIRECTORY = "paper_refinement_0to25pct"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    alias: str
    registry_name: str
    display_name: str
    input_fault_sites: int
    minimum_training_hours: float
    minimum_training_steps: int


QKERAS_CONFIG = get_config("qkeras_y256_64_w4a8")
MODELS = (
    ModelSpec(
        family="lgn",
        alias="s02_p163M_tau20",
        registry_name=(
            "lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_20"
        ),
        display_name="Dense LGN 0.163M (tau=20)",
        input_fault_sites=14 * 100,
        minimum_training_hours=5.0,
        minimum_training_steps=20_000,
    ),
    ModelSpec(
        family="qkeras",
        alias=QKERAS_CONFIG.alias,
        registry_name=QKERAS_CONFIG.registry_name,
        display_name="Dense QKeras Y256/64 W4/A8",
        input_fault_sites=int(
            stage_metadata(QKERAS_CONFIG, "input")["total_fault_sites_per_event"]
        ),
        minimum_training_hours=3.0,
        minimum_training_steps=6_000,
    ),
)


SOURCE_POINT_PROVENANCE = {
    "plot": "fit_10_triplet_architecture_slide_title",
    "source_table": (
        "synthesis-SmartPixels/plots/all_models_balanced_accuracy_resources.csv"
    ),
    "selection": [
        {
            "family": "dense_lgn",
            "rule": "second point from the left by LUT+FF utilization",
            "model": "s02_p163M_tau20",
            "balanced_accuracy": 0.807266,
            "lut_plus_ff": 11_578,
            "selected_tau": 20,
        },
        {
            "family": "dense_qkeras",
            "rule": "third point from the right by LUT+FF utilization",
            "model": "qkeras-model-2-yprofile-d256-d64-w4a8",
            "balanced_accuracy": 0.798000,
            "lut_plus_ff": 275_815,
        },
    ],
}


def percentage_tag(percentage: float | None) -> str:
    if percentage is None:
        return "baseline_clean"
    return f"train_{percentage:g}pct".replace(".", "p")


def evaluation_flip_counts(total_fault_sites: int) -> tuple[int, ...]:
    """Convert the common 0--60% grid to exact, unique per-event counts."""
    if total_fault_sites < 1:
        raise ValueError("The input fault space must contain at least one site.")
    counts = []
    for percentage in EVALUATION_PERCENTAGES:
        count = int(round(total_fault_sites * percentage / 100.0))
        count = min(total_fault_sites, max(0, count))
        if not counts or count != counts[-1]:
            counts.append(count)
    if counts[0] != 0 or len(counts) < 2:
        raise ValueError(f"Invalid evaluation grid for {total_fault_sites} sites.")
    return tuple(counts)


def evaluation_flip_count_chunks(
    total_fault_sites: int,
) -> tuple[tuple[int, ...], ...]:
    nonzero = evaluation_flip_counts(total_fault_sites)[1:]
    return tuple(
        tuple(nonzero[index : index + COUNTS_PER_CHUNK])
        for index in range(0, len(nonzero), COUNTS_PER_CHUNK)
    )


def _additional_flip_counts(
    total_fault_sites: int,
    percentages: tuple[float, ...],
    excluded_counts: tuple[int, ...],
) -> tuple[int, ...]:
    if total_fault_sites < 1:
        raise ValueError("The input fault space must contain at least one site.")
    counts = tuple(
        sorted(
            {
                min(
                    total_fault_sites,
                    max(1, int(round(total_fault_sites * percentage / 100.0))),
                )
                for percentage in percentages
            }
        )
    )
    overlap = set(excluded_counts).intersection(counts)
    if len(counts) != len(percentages) or overlap:
        raise ValueError(
            f"Additional grid is not distinct for {total_fault_sites} sites: "
            f"counts={counts}, overlap={sorted(overlap)}"
        )
    return counts


def refinement_flip_counts(total_fault_sites: int) -> tuple[int, ...]:
    """Return unique exact counts for the 2.5% and 7.5% refinement."""
    return _additional_flip_counts(
        total_fault_sites,
        REFINEMENT_PERCENTAGES,
        evaluation_flip_counts(total_fault_sites),
    )


def paper_refinement_flip_counts(total_fault_sites: int) -> tuple[int, ...]:
    """Return the five new exact counts used by the 0--50% paper plot."""
    excluded = tuple(
        sorted(
            (
                *evaluation_flip_counts(total_fault_sites),
                *refinement_flip_counts(total_fault_sites),
            )
        )
    )
    return _additional_flip_counts(
        total_fault_sites,
        PAPER_REFINEMENT_PERCENTAGES,
        excluded,
    )


def training_tasks() -> list[dict]:
    tasks = []
    for model in MODELS:
        for percentage in (None, *TRAINING_PERCENTAGES):
            tasks.append(
                {
                    "family": model.family,
                    "condition": (
                        "clean_baseline"
                        if percentage is None
                        else "fault_aware_training"
                    ),
                    "tag": percentage_tag(percentage),
                    "percentage": percentage,
                    "probability": 0.0 if percentage is None else percentage / 100,
                    "model_alias": model.alias,
                    "registry_name": model.registry_name,
                    "minimum_training_hours": model.minimum_training_hours,
                    "minimum_training_steps": model.minimum_training_steps,
                }
            )
    return tasks


def evaluation_tasks() -> list[dict]:
    tasks = []
    for model in MODELS:
        chunks = evaluation_flip_count_chunks(model.input_fault_sites)
        for percentage in (None, *TRAINING_PERCENTAGES):
            for chunk_index, counts in enumerate(chunks):
                tasks.append(
                    {
                        "family": model.family,
                        "condition": (
                            "clean_baseline"
                            if percentage is None
                            else "fault_aware_training"
                        ),
                        "tag": percentage_tag(percentage),
                        "percentage": percentage,
                        "model_alias": model.alias,
                        "registry_name": model.registry_name,
                        "chunk_index": chunk_index,
                        "flip_counts": counts,
                        "minimum_training_hours": model.minimum_training_hours,
                        "minimum_training_steps": model.minimum_training_steps,
                    }
                )
    return tasks


def refinement_tasks() -> list[dict]:
    """One two-point inference-only refinement task per trained checkpoint."""
    tasks = []
    for model in MODELS:
        for percentage in (None, *TRAINING_PERCENTAGES):
            tasks.append(
                {
                    "family": model.family,
                    "condition": (
                        "clean_baseline"
                        if percentage is None
                        else "fault_aware_training"
                    ),
                    "tag": percentage_tag(percentage),
                    "percentage": percentage,
                    "model_alias": model.alias,
                    "registry_name": model.registry_name,
                    "flip_counts": refinement_flip_counts(
                        model.input_fault_sites
                    ),
                    "minimum_training_hours": model.minimum_training_hours,
                    "minimum_training_steps": model.minimum_training_steps,
                }
            )
    return tasks


def paper_refinement_tasks() -> list[dict]:
    """One five-point paper-refinement task per trained checkpoint."""
    tasks = []
    for model in MODELS:
        for percentage in (None, *TRAINING_PERCENTAGES):
            tasks.append(
                {
                    "family": model.family,
                    "condition": (
                        "clean_baseline"
                        if percentage is None
                        else "fault_aware_training"
                    ),
                    "tag": percentage_tag(percentage),
                    "percentage": percentage,
                    "model_alias": model.alias,
                    "registry_name": model.registry_name,
                    "flip_counts": paper_refinement_flip_counts(
                        model.input_fault_sites
                    ),
                    "minimum_training_hours": model.minimum_training_hours,
                    "minimum_training_steps": model.minimum_training_steps,
                }
            )
    return tasks


def _task_at(tasks: list[dict], task_id: int, label: str) -> dict:
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"{label} task ID must lie in [0, {len(tasks) - 1}].")
    return tasks[task_id]


def training_configuration(task_id: int) -> dict:
    return _task_at(training_tasks(), task_id, "Training")


def evaluation_configuration(task_id: int) -> dict:
    return _task_at(evaluation_tasks(), task_id, "Evaluation")


def refinement_configuration(task_id: int) -> dict:
    return _task_at(refinement_tasks(), task_id, "Refinement")


def paper_refinement_configuration(task_id: int) -> dict:
    return _task_at(paper_refinement_tasks(), task_id, "Paper refinement")


def _model_root(root: Path, model: ModelSpec, tag: str) -> Path:
    return root / model.family / tag / model.alias


def _combine_lgn_source(root: Path, model: ModelSpec, tag: str) -> list[dict]:
    from lgnStudies.bitFlipStudy.combine_results import (
        load_and_combine_model_chunks,
    )

    model_root = _model_root(root, model, tag)
    chunks = evaluation_flip_count_chunks(model.input_fault_sites)
    counts = evaluation_flip_counts(model.input_fault_sites)
    rows, metadata, prediction_source = load_and_combine_model_chunks(
        model_root,
        model.alias,
        flip_count_chunks=chunks,
        expected_flip_counts=counts,
    )
    model_root.mkdir(parents=True, exist_ok=True)
    write_rows(model_root / "bit_flip_results.csv", rows)
    metadata = {
        **metadata,
        "fault_scope": "input",
        "target_evaluation_percentages": list(EVALUATION_PERCENTAGES),
    }
    (model_root / "bit_flip_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    shutil.copyfile(prediction_source, model_root / "clean_predictions.npy")
    return rows


def _combine_qkeras_source(root: Path, model: ModelSpec, tag: str) -> list[dict]:
    from qkerasStudies.bitFlipStudy.combine_results import (
        write_combined_artifacts,
    )

    return write_combined_artifacts(
        _model_root(root, model, tag),
        model.alias,
        evaluation_flip_count_chunks(model.input_fault_sites),
        evaluation_flip_counts(model.input_fault_sites),
        REPEATS,
    )


def _read_result_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing result CSV: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result CSV is empty: {path}")
    return rows


def _validate_result_grid(
    rows: list[dict],
    model: ModelSpec,
    expected_counts: tuple[int, ...],
    source: Path,
) -> None:
    if {row["model_alias"] for row in rows} != {model.alias}:
        raise ValueError(f"Wrong model alias in {source}.")
    if {row["registry_name"] for row in rows} != {model.registry_name}:
        raise ValueError(f"Wrong registry name in {source}.")
    if {
        int(row["total_fault_sites_per_event"]) for row in rows
    } != {model.input_fault_sites}:
        raise ValueError(f"Wrong input-fault-site total in {source}.")

    observed = Counter(int(row["flip_count_per_event"]) for row in rows)
    expected = {
        count: 1 if count == 0 else REPEATS for count in expected_counts
    }
    if dict(observed) != expected:
        raise ValueError(
            f"Wrong count/repeat grid in {source}: expected={expected}, "
            f"observed={dict(observed)}"
        )

    keys = [
        (int(row["flip_count_per_event"]), int(row["repeat"]))
        for row in rows
    ]
    expected_keys = {
        (count, repeat)
        for count in expected_counts
        for repeat in (range(1) if count == 0 else range(REPEATS))
    }
    if set(keys) != expected_keys or len(keys) != len(expected_keys):
        raise ValueError(f"Wrong count/repeat identities in {source}.")


def _merge_additional_result_source(
    model: ModelSpec,
    base_csv: Path,
    base_counts: tuple[int, ...],
    base_predictions: Path,
    additional_csv: Path,
    additional_counts: tuple[int, ...],
    additional_predictions: Path,
    merged_csv: Path,
    context: str,
) -> tuple[list[dict], tuple[int, ...]]:
    base_rows = _read_result_rows(base_csv)
    additional_rows = _read_result_rows(additional_csv)

    _validate_result_grid(base_rows, model, base_counts, base_csv)
    _validate_result_grid(
        additional_rows,
        model,
        (0, *additional_counts),
        additional_csv,
    )

    base_clean = next(
        row for row in base_rows if int(row["flip_count_per_event"]) == 0
    )
    additional_clean = next(
        row
        for row in additional_rows
        if int(row["flip_count_per_event"]) == 0
    )
    if base_clean != additional_clean:
        raise ValueError(f"K=0 metrics differ for {context}.")

    if not base_predictions.is_file() or not additional_predictions.is_file():
        raise ValueError(f"Missing clean predictions for {context}.")
    if not np.array_equal(
        np.load(base_predictions),
        np.load(additional_predictions),
    ):
        raise ValueError(f"K=0 predictions differ for {context}.")

    merged_rows = [
        *base_rows,
        *(
            row
            for row in additional_rows
            if int(row["flip_count_per_event"]) > 0
        ),
    ]
    merged_rows.sort(
        key=lambda row: (
            int(row["flip_count_per_event"]),
            int(row["repeat"]),
        )
    )
    merged_counts = tuple(sorted((*base_counts, *additional_counts)))
    _validate_result_grid(merged_rows, model, merged_counts, merged_csv)
    write_rows(merged_csv, merged_rows)
    return merged_rows, merged_counts


def _combine_refined_source(
    root: Path,
    model: ModelSpec,
    tag: str,
) -> list[dict]:
    model_root = _model_root(root, model, tag)
    base_csv = model_root / "bit_flip_results.csv"
    refinement_root = model_root / REFINEMENT_DIRECTORY
    refinement_csv = refinement_root / "bit_flip_results.csv"
    refined_csv = model_root / "bit_flip_results_refined_0to10pct.csv"
    extra_counts = refinement_flip_counts(model.input_fault_sites)
    merged_rows, refined_counts = _merge_additional_result_source(
        model,
        base_csv,
        evaluation_flip_counts(model.input_fault_sites),
        model_root / "clean_predictions.npy",
        refinement_csv,
        extra_counts,
        refinement_root / "clean_predictions.npy",
        refined_csv,
        f"{model.family}/{tag}/{model.alias} low-percentage refinement",
    )

    metadata = {
        "study": "selected_models_input_bit_flip_0to10pct_refinement",
        "complete": True,
        "model_alias": model.alias,
        "registry_name": model.registry_name,
        "fault_scope": "input",
        "training_tag": tag,
        "base_result_csv": str(base_csv),
        "refinement_result_csv": str(refinement_csv),
        "merged_result_csv": str(refined_csv),
        "base_target_percentages": list(EVALUATION_PERCENTAGES),
        "additional_target_percentages": list(REFINEMENT_PERCENTAGES),
        "refined_target_percentages": list(REFINED_EVALUATION_PERCENTAGES),
        "additional_flip_counts": list(extra_counts),
        "merged_flip_counts": list(refined_counts),
        "realized_merged_percentages": [
            100 * count / model.input_fault_sites for count in refined_counts
        ],
        "repeats_per_nonzero_point": REPEATS,
    }
    (model_root / "bit_flip_metadata_refined_0to10pct.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return merged_rows


def _combine_paper_refined_source(
    root: Path,
    model: ModelSpec,
    tag: str,
) -> list[dict]:
    model_root = _model_root(root, model, tag)
    base_csv = model_root / "bit_flip_results_refined_0to10pct.csv"
    paper_root = model_root / PAPER_REFINEMENT_DIRECTORY
    paper_csv = paper_root / "bit_flip_results.csv"
    merged_csv = model_root / "bit_flip_results_paper_0to50pct.csv"
    base_counts = tuple(
        sorted(
            (
                *evaluation_flip_counts(model.input_fault_sites),
                *refinement_flip_counts(model.input_fault_sites),
            )
        )
    )
    extra_counts = paper_refinement_flip_counts(model.input_fault_sites)
    merged_rows, merged_counts = _merge_additional_result_source(
        model,
        base_csv,
        base_counts,
        model_root / "clean_predictions.npy",
        paper_csv,
        extra_counts,
        paper_root / "clean_predictions.npy",
        merged_csv,
        f"{model.family}/{tag}/{model.alias} paper refinement",
    )

    metadata = {
        "study": "selected_models_input_bit_flip_paper_refinement",
        "complete": True,
        "model_alias": model.alias,
        "registry_name": model.registry_name,
        "fault_scope": "input",
        "training_tag": tag,
        "base_refined_result_csv": str(base_csv),
        "paper_refinement_result_csv": str(paper_csv),
        "merged_result_csv": str(merged_csv),
        "additional_target_percentages": list(
            PAPER_REFINEMENT_PERCENTAGES
        ),
        "paper_target_percentages": list(
            PAPER_REFINED_EVALUATION_PERCENTAGES
        ),
        "additional_flip_counts": list(extra_counts),
        "merged_flip_counts": list(merged_counts),
        "realized_merged_percentages": [
            100 * count / model.input_fault_sites for count in merged_counts
        ],
        "plot_x_range_percent": [0, 50],
        "repeats_per_nonzero_point": REPEATS,
    }
    (model_root / "bit_flip_metadata_paper_0to50pct.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return merged_rows


def combine_campaign(root: Path) -> None:
    from lgnStudies.bitFlipStudy.plots import plot_training_percentage_sweep_2d

    if not root.is_dir():
        raise ValueError(f"Campaign root does not exist: {root}")

    model_summaries = []
    for model in MODELS:
        combine_source = (
            _combine_lgn_source if model.family == "lgn" else _combine_qkeras_source
        )
        baseline_rows = combine_source(root, model, percentage_tag(None))
        fault_aware_rows = {
            percentage: combine_source(root, model, percentage_tag(percentage))
            for percentage in TRAINING_PERCENTAGES
        }
        output_stem = root / f"{model.alias}_input_bit_flip_robustness_linear"
        plot_training_percentage_sweep_2d(
            baseline_rows,
            fault_aware_rows,
            output_stem,
            model.alias,
            title=f"{model.display_name}: random input bit flips",
            x_limit=max(EVALUATION_PERCENTAGES),
            x_label="Random input bit flips per event [%]",
            linear_x_axis=True,
        )
        counts = evaluation_flip_counts(model.input_fault_sites)
        model_summaries.append(
            {
                **asdict(model),
                "training_fault_percentages": list(TRAINING_PERCENTAGES),
                "target_evaluation_percentages": list(EVALUATION_PERCENTAGES),
                "evaluation_flip_counts": list(counts),
                "realized_evaluation_percentages": [
                    100 * count / model.input_fault_sites for count in counts
                ],
                "plot_png": str(output_stem.with_suffix(".png")),
                "plot_pdf": str(output_stem.with_suffix(".pdf")),
            }
        )

    summary = {
        "study": "selected_models_random_input_bit_flip_training_sweep",
        "fault_scope": "input",
        "training_fault_definition": (
            "Independent Bernoulli bit flips at the model input during training."
        ),
        "evaluation_fault_definition": (
            "Exactly K unique random input bits per event, sampled without "
            "replacement independently for each event and repeat."
        ),
        "repeats_per_nonzero_evaluation_point": REPEATS,
        "x_axis_scale": "linear",
        "source_point_provenance": SOURCE_POINT_PROVENANCE,
        "models": model_summaries,
    }
    (root / "input_bit_flip_campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[DONE] Combined selected-model input-fault campaign under {root}")


def combine_refined_campaign(root: Path) -> None:
    """Merge the two new low-percentage points into the completed campaign."""
    from lgnStudies.bitFlipStudy.plots import plot_training_percentage_sweep_2d

    if not root.is_dir():
        raise ValueError(f"Campaign root does not exist: {root}")
    if not (root / "input_bit_flip_campaign_summary.json").is_file():
        raise ValueError(
            "The base campaign has not been combined successfully: "
            f"{root / 'input_bit_flip_campaign_summary.json'}"
        )

    model_summaries = []
    for model in MODELS:
        baseline_rows = _combine_refined_source(
            root, model, percentage_tag(None)
        )
        fault_aware_rows = {
            percentage: _combine_refined_source(
                root, model, percentage_tag(percentage)
            )
            for percentage in TRAINING_PERCENTAGES
        }
        output_stem = root / (
            f"{model.alias}_input_bit_flip_robustness_linear_"
            "refined_0to10pct"
        )
        plot_training_percentage_sweep_2d(
            baseline_rows,
            fault_aware_rows,
            output_stem,
            model.alias,
            title=(
                f"{model.display_name}: random input bit flips "
                "(refined from 0 to 10%)"
            ),
            x_limit=max(EVALUATION_PERCENTAGES),
            x_label="Random input bit flips per event [%]",
            linear_x_axis=True,
        )
        counts = tuple(
            sorted(
                (
                    *evaluation_flip_counts(model.input_fault_sites),
                    *refinement_flip_counts(model.input_fault_sites),
                )
            )
        )
        model_summaries.append(
            {
                **asdict(model),
                "additional_target_percentages": list(REFINEMENT_PERCENTAGES),
                "additional_flip_counts": list(
                    refinement_flip_counts(model.input_fault_sites)
                ),
                "refined_flip_counts": list(counts),
                "realized_refined_percentages": [
                    100 * count / model.input_fault_sites for count in counts
                ],
                "plot_png": str(output_stem.with_suffix(".png")),
                "plot_pdf": str(output_stem.with_suffix(".pdf")),
            }
        )

    summary = {
        "study": "selected_models_input_bit_flip_0to10pct_refinement",
        "base_campaign_summary": str(
            root / "input_bit_flip_campaign_summary.json"
        ),
        "fault_scope": "input",
        "training_reused": True,
        "additional_target_percentages": list(REFINEMENT_PERCENTAGES),
        "repeats_per_nonzero_evaluation_point": REPEATS,
        "x_axis_scale": "linear",
        "models": model_summaries,
    }
    summary_path = (
        root / "input_bit_flip_campaign_refined_0to10pct_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] Wrote low-percentage refinement under {root}")


def combine_paper_refined_campaign(root: Path) -> None:
    """Merge all refinement points and write the 0--50% paper plots."""
    from lgnStudies.bitFlipStudy.plots import plot_training_percentage_sweep_2d

    prerequisite = (
        root / "input_bit_flip_campaign_refined_0to10pct_summary.json"
    )
    if not root.is_dir() or not prerequisite.is_file():
        raise ValueError(
            "The completed 0--10% refinement is required before the paper "
            f"refinement: {prerequisite}"
        )

    model_summaries = []
    for model in MODELS:
        baseline_rows = _combine_paper_refined_source(
            root, model, percentage_tag(None)
        )
        fault_aware_rows = {
            percentage: _combine_paper_refined_source(
                root, model, percentage_tag(percentage)
            )
            for percentage in TRAINING_PERCENTAGES
        }
        output_stem = root / (
            f"{model.alias}_input_bit_flip_robustness_paper_0to50pct"
        )
        plot_training_percentage_sweep_2d(
            baseline_rows,
            fault_aware_rows,
            output_stem,
            model.alias,
            title=f"{model.display_name}: input-bit-flip robustness",
            x_limit=50.0,
            x_label="Random input bit flips per event [%]",
            linear_x_axis=True,
            large_paper_style=True,
        )
        counts = tuple(
            sorted(
                (
                    *evaluation_flip_counts(model.input_fault_sites),
                    *refinement_flip_counts(model.input_fault_sites),
                    *paper_refinement_flip_counts(model.input_fault_sites),
                )
            )
        )
        model_summaries.append(
            {
                **asdict(model),
                "additional_target_percentages": list(
                    PAPER_REFINEMENT_PERCENTAGES
                ),
                "additional_flip_counts": list(
                    paper_refinement_flip_counts(model.input_fault_sites)
                ),
                "paper_flip_counts": list(counts),
                "realized_paper_percentages": [
                    100 * count / model.input_fault_sites for count in counts
                ],
                "plot_x_range_percent": [0, 50],
                "plot_png": str(output_stem.with_suffix(".png")),
                "plot_pdf": str(output_stem.with_suffix(".pdf")),
            }
        )

    summary = {
        "study": "selected_models_input_bit_flip_paper_refinement",
        "prerequisite_summary": str(prerequisite),
        "fault_scope": "input",
        "training_reused": True,
        "additional_target_percentages": list(PAPER_REFINEMENT_PERCENTAGES),
        "all_target_percentages": list(PAPER_REFINED_EVALUATION_PERCENTAGES),
        "repeats_per_nonzero_evaluation_point": REPEATS,
        "x_axis_scale": "linear",
        "plot_x_range_percent": [0, 50],
        "models": model_summaries,
    }
    summary_path = root / "input_bit_flip_campaign_paper_0to50pct_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DONE] Wrote paper-ready 0--50% plots under {root}")


def write_campaign_config(root: Path, git_commit: str) -> None:
    root.mkdir(parents=True, exist_ok=False)
    model_grids = []
    for model in MODELS:
        counts = evaluation_flip_counts(model.input_fault_sites)
        model_grids.append(
            {
                **asdict(model),
                "evaluation_flip_counts": list(counts),
                "realized_evaluation_percentages": [
                    100 * count / model.input_fault_sites for count in counts
                ],
            }
        )
    config = {
        "study": "selected_models_random_input_bit_flip_training_sweep",
        "git_commit": git_commit,
        "fault_scope": "input",
        "source_point_provenance": SOURCE_POINT_PROVENANCE,
        "training_fault_percentages": list(TRAINING_PERCENTAGES),
        "training_conditions_per_model": 1 + len(TRAINING_PERCENTAGES),
        "evaluation_percentage_grid": list(EVALUATION_PERCENTAGES),
        "evaluation_repeats_per_nonzero_point": REPEATS,
        "evaluation_counts_per_chunk": COUNTS_PER_CHUNK,
        "training_task_count": len(training_tasks()),
        "evaluation_task_count": len(evaluation_tasks()),
        "x_axis_scale": "linear",
        "models": model_grids,
    }
    (root / "campaign_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _format_percentage(value: float | None) -> str:
    return "none" if value is None else f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--training-task-id", type=int)
    actions.add_argument("--evaluation-task-id", type=int)
    actions.add_argument("--refinement-task-id", type=int)
    actions.add_argument("--paper-refinement-task-id", type=int)
    actions.add_argument("--print-training-task-count", action="store_true")
    actions.add_argument("--print-evaluation-task-count", action="store_true")
    actions.add_argument("--print-refinement-task-count", action="store_true")
    actions.add_argument(
        "--print-paper-refinement-task-count", action="store_true"
    )
    actions.add_argument("--print-model-grids", action="store_true")
    actions.add_argument("--combine", action="store_true")
    actions.add_argument("--combine-refinement", action="store_true")
    actions.add_argument("--combine-paper-refinement", action="store_true")
    actions.add_argument("--write-campaign-config", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    try:
        if args.training_task_id is not None:
            task = training_configuration(args.training_task_id)
            print(
                "\t".join(
                    (
                        task["family"],
                        task["condition"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        f"{task['probability']:.12g}",
                        task["model_alias"],
                        task["registry_name"],
                        f"{task['minimum_training_hours']:g}",
                        str(task["minimum_training_steps"]),
                    )
                )
            )
        elif args.evaluation_task_id is not None:
            task = evaluation_configuration(args.evaluation_task_id)
            print(
                "\t".join(
                    (
                        task["family"],
                        task["condition"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        task["model_alias"],
                        task["registry_name"],
                        f"{task['chunk_index']:02d}",
                        " ".join(map(str, task["flip_counts"])),
                        f"{task['minimum_training_hours']:g}",
                        str(task["minimum_training_steps"]),
                    )
                )
            )
        elif args.refinement_task_id is not None:
            task = refinement_configuration(args.refinement_task_id)
            print(
                "\t".join(
                    (
                        task["family"],
                        task["condition"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        task["model_alias"],
                        task["registry_name"],
                        " ".join(map(str, task["flip_counts"])),
                        f"{task['minimum_training_hours']:g}",
                        str(task["minimum_training_steps"]),
                    )
                )
            )
        elif args.paper_refinement_task_id is not None:
            task = paper_refinement_configuration(
                args.paper_refinement_task_id
            )
            print(
                "\t".join(
                    (
                        task["family"],
                        task["condition"],
                        task["tag"],
                        _format_percentage(task["percentage"]),
                        task["model_alias"],
                        task["registry_name"],
                        " ".join(map(str, task["flip_counts"])),
                        f"{task['minimum_training_hours']:g}",
                        str(task["minimum_training_steps"]),
                    )
                )
            )
        elif args.print_training_task_count:
            print(len(training_tasks()))
        elif args.print_evaluation_task_count:
            print(len(evaluation_tasks()))
        elif args.print_refinement_task_count:
            print(len(refinement_tasks()))
        elif args.print_paper_refinement_task_count:
            print(len(paper_refinement_tasks()))
        elif args.print_model_grids:
            for model in MODELS:
                print(
                    "\t".join(
                        (
                            model.family,
                            model.alias,
                            str(model.input_fault_sites),
                            " ".join(map(str, evaluation_flip_counts(model.input_fault_sites))),
                        )
                    )
                )
        elif args.combine:
            if args.root is None:
                parser.error("--root is required with --combine.")
            combine_campaign(args.root)
        elif args.combine_refinement:
            if args.root is None:
                parser.error("--root is required with --combine-refinement.")
            combine_refined_campaign(args.root)
        elif args.combine_paper_refinement:
            if args.root is None:
                parser.error(
                    "--root is required with --combine-paper-refinement."
                )
            combine_paper_refined_campaign(args.root)
        elif args.write_campaign_config:
            if args.root is None or args.git_commit is None:
                parser.error(
                    "--root and --git-commit are required with "
                    "--write-campaign-config."
                )
            write_campaign_config(args.root, args.git_commit)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
