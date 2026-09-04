#!/usr/bin/env python3
"""Task mapping, validation, and plotting for internal random overwrites."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil

import numpy as np

from models.registry import REGISTRY
from qkerasStudies.fault_injection import stage_metadata
from qkerasStudies.models import _import_qkeras_layers, build_model, get_config


DEFAULT_TRAINING_PERCENTAGES = (0.0, 1.0, 3.0, 10.0)
EVALUATION_PERCENTAGES = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
)
REPEATS = 5
COUNTS_PER_CHUNK = 3
FAULT_SCOPE = "internal"
FAULT_OPERATION = "random_set"


def parse_training_percentages(text: str | None = None) -> tuple[float, ...]:
    """Parse a configurable training grid while requiring a clean baseline."""
    if text is None:
        text = os.environ.get("INTERNAL_OVERWRITE_TRAINING_PERCENTAGES", "")
    if not text.strip():
        return DEFAULT_TRAINING_PERCENTAGES
    tokens = text.replace(",", " ").split()
    try:
        percentages = tuple(sorted({float(token) for token in tokens}))
    except ValueError as error:
        raise ValueError(f"Invalid training-percentage list: {text!r}.") from error
    if not percentages or percentages[0] != 0.0:
        raise ValueError("Training percentages must include the 0% clean baseline.")
    if any(not 0.0 <= value <= 100.0 for value in percentages):
        raise ValueError("Training percentages must lie in [0, 100].")
    return percentages


TRAINING_PERCENTAGES = parse_training_percentages()


@dataclass(frozen=True)
class ModelSpec:
    family: str
    alias: str
    registry_name: str
    display_name: str
    internal_fault_sites: int
    internal_stage_widths: tuple[int, ...]
    activation_bits: int
    minimum_training_hours: float
    minimum_training_steps: int


QKERAS_CONFIG = get_config("qkeras_y256_64_w4a8")
QKERAS_INTERNAL_METADATA = stage_metadata(QKERAS_CONFIG, FAULT_SCOPE)
LNN_REGISTRY_NAME = "lgn-dense-2-dense-100_128_-model2lgnFull_s02_p163M_20"
LNN_INTERNAL_WIDTHS = tuple(REGISTRY[LNN_REGISTRY_NAME]["class"].hidden_dims)
MODELS = (
    ModelSpec(
        family="lnn",
        alias="s02_p163M_tau20",
        registry_name=LNN_REGISTRY_NAME,
        display_name="Dense LNN: s02 p0.163M, tau=20",
        internal_fault_sites=sum(LNN_INTERNAL_WIDTHS),
        internal_stage_widths=LNN_INTERNAL_WIDTHS,
        activation_bits=1,
        minimum_training_hours=5.0,
        minimum_training_steps=20_000,
    ),
    ModelSpec(
        family="qkeras",
        alias=QKERAS_CONFIG.alias,
        registry_name=QKERAS_CONFIG.registry_name,
        display_name="Dense QKeras: Y256/64 W4/A8",
        internal_fault_sites=int(
            QKERAS_INTERNAL_METADATA["total_fault_sites_per_event"]
        ),
        internal_stage_widths=tuple(QKERAS_CONFIG.hidden_units),
        activation_bits=QKERAS_CONFIG.activation_bits,
        minimum_training_hours=3.0,
        minimum_training_steps=6_000,
    ),
)


def percentage_tag(percentage: float) -> str:
    if percentage == 0:
        return "baseline_clean"
    return f"train_{percentage:g}pct".replace(".", "p")


def evaluation_counts(total_sites: int) -> tuple[int, ...]:
    counts = tuple(
        sorted(
            {
                min(total_sites, max(0, round(total_sites * value / 100.0)))
                for value in EVALUATION_PERCENTAGES
            }
        )
    )
    if counts[0] != 0 or len(counts) != len(EVALUATION_PERCENTAGES):
        raise ValueError(
            f"Evaluation percentages are not uniquely representable for {total_sites} sites."
        )
    return counts


def evaluation_chunks(total_sites: int) -> tuple[tuple[int, ...], ...]:
    nonzero = evaluation_counts(total_sites)[1:]
    return tuple(
        tuple(nonzero[index : index + COUNTS_PER_CHUNK])
        for index in range(0, len(nonzero), COUNTS_PER_CHUNK)
    )


def training_tasks() -> list[dict]:
    return [
        {
            **asdict(model),
            "training_percentage": percentage,
            "training_probability": percentage / 100.0,
            "tag": percentage_tag(percentage),
        }
        for model in MODELS
        for percentage in TRAINING_PERCENTAGES
    ]


def evaluation_tasks() -> list[dict]:
    tasks = []
    for model in MODELS:
        for percentage in TRAINING_PERCENTAGES:
            for chunk_index, counts in enumerate(
                evaluation_chunks(model.internal_fault_sites)
            ):
                tasks.append(
                    {
                        **asdict(model),
                        "training_percentage": percentage,
                        "tag": percentage_tag(percentage),
                        "chunk_index": chunk_index,
                        "selected_site_counts": counts,
                    }
                )
    return tasks


def _task_at(tasks: list[dict], task_id: int, kind: str) -> dict:
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"{kind} task ID must lie in [0, {len(tasks) - 1}].")
    return tasks[task_id]


def training_configuration(task_id: int) -> dict:
    return _task_at(training_tasks(), task_id, "Training")


def evaluation_configuration(task_id: int) -> dict:
    return _task_at(evaluation_tasks(), task_id, "Evaluation")


def _condition_root(root: Path, model: ModelSpec, percentage: float) -> Path:
    return root / model.family / percentage_tag(percentage) / model.alias


def write_campaign_config(root: Path, git_commit: str) -> None:
    root.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1,
        "study": "selected_models_internal_activation_random_overwrite",
        "git_commit": git_commit,
        "fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "fault_operation_definition": (
            "Selected sites are independently assigned uniformly random 0/1 "
            "targets. A selected write leaves the stored bit unchanged with "
            "probability 1/2."
        ),
        "training_percentages": list(TRAINING_PERCENTAGES),
        "evaluation_target_percentages": list(EVALUATION_PERCENTAGES),
        "plot_x_range_percent": [0.0, 10.0],
        "evaluation_repeats_per_nonzero_point": REPEATS,
        "counts_per_evaluation_chunk": COUNTS_PER_CHUNK,
        "training_task_count": len(training_tasks()),
        "evaluation_task_count": len(evaluation_tasks()),
        "models": [
            {
                **asdict(model),
                "evaluation_selected_site_counts": list(
                    evaluation_counts(model.internal_fault_sites)
                ),
                "realized_evaluation_percentages": [
                    100 * count / model.internal_fault_sites
                    for count in evaluation_counts(model.internal_fault_sites)
                ],
            }
            for model in MODELS
        ],
    }
    (root / "campaign_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _validate_training(
    root: Path, model: ModelSpec, training_percentage: float
) -> dict:
    condition_root = _condition_root(root, model, training_percentage)
    status = _load_json(condition_root / "training" / "training_complete.json")
    expected = {
        "complete": True,
        "model_alias": model.alias,
        "registry_name": model.registry_name,
        "training_fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "total_fault_sites_per_event": model.internal_fault_sites,
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise ValueError(
                f"Training metadata mismatch for {model.family}/{percentage_tag(training_percentage)}: "
                f"{key}={status.get(key)!r}, expected {value!r}."
            )
    if not np.isclose(
        float(status["training_fault_percentage"]),
        training_percentage,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Wrong training percentage in {condition_root}.")
    checkpoint_name = "final_model.pth" if model.family == "lnn" else "final_model.keras"
    if not (condition_root / "training" / checkpoint_name).exists():
        raise ValueError(f"Missing completed checkpoint in {condition_root / 'training'}.")
    return status


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing evaluation table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty evaluation table: {path}")
    return rows


def _validate_chunk(
    rows: list[dict],
    metadata: dict,
    model: ModelSpec,
    expected_nonzero_counts: tuple[int, ...],
    source: Path,
) -> None:
    expected_counts = (0, *expected_nonzero_counts)
    expected_repetitions = {
        0: 1,
        **{count: REPEATS for count in expected_nonzero_counts},
    }
    observed = Counter(int(row["flip_count_per_event"]) for row in rows)
    if dict(observed) != expected_repetitions:
        raise ValueError(
            f"Incomplete count/repeat grid in {source}: {dict(observed)}; "
            f"expected {expected_repetitions}."
        )
    if {row["model_alias"] for row in rows} != {model.alias}:
        raise ValueError(f"Wrong model alias in {source}.")
    if {row["registry_name"] for row in rows} != {model.registry_name}:
        raise ValueError(f"Wrong registry in {source}.")
    if {int(row["total_fault_sites_per_event"]) for row in rows} != {
        model.internal_fault_sites
    }:
        raise ValueError(f"Wrong internal-site total in {source}.")
    metadata_expected = {
        "complete": True,
        "model_alias": model.alias,
        "registry_name": model.registry_name,
        "fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "total_fault_sites_per_event": model.internal_fault_sites,
        "flip_counts_per_event": list(expected_counts),
        "repeats_per_nonzero_count": REPEATS,
    }
    for key, value in metadata_expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Evaluation metadata mismatch in {source}: "
                f"{key}={metadata.get(key)!r}, expected {value!r}."
            )


def _combine_condition(
    root: Path,
    model: ModelSpec,
    training_percentage: float,
) -> list[dict]:
    status = _validate_training(root, model, training_percentage)
    condition_root = _condition_root(root, model, training_percentage)
    clean_rows = []
    nonzero_rows = []
    reference_predictions = None
    prediction_source = None
    expected_nonzero_counts = set(evaluation_counts(model.internal_fault_sites)[1:])
    source_chunks = []
    for chunk_root in sorted((condition_root / "chunks").glob("chunk_*")):
        metadata_path = chunk_root / "bit_flip_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = _load_json(metadata_path)
        if not metadata.get("complete"):
            continue
        chunk_counts = tuple(
            int(count)
            for count in metadata.get("flip_counts_per_event", ())
            if int(count) > 0
        )
        if not chunk_counts or not set(chunk_counts).issubset(expected_nonzero_counts):
            raise ValueError(f"Unexpected evaluation counts in {metadata_path}.")
        csv_path = chunk_root / "bit_flip_results.csv"
        rows = _read_rows(csv_path)
        _validate_chunk(rows, metadata, model, chunk_counts, csv_path)
        clean_rows.extend(
            row for row in rows if int(row["flip_count_per_event"]) == 0
        )
        nonzero_rows.extend(
            row for row in rows if int(row["flip_count_per_event"]) > 0
        )
        predictions_path = chunk_root / "clean_predictions.npy"
        if not predictions_path.is_file():
            raise ValueError(f"Missing clean predictions: {predictions_path}")
        predictions = np.load(predictions_path)
        if reference_predictions is None:
            reference_predictions = predictions
            prediction_source = predictions_path
        elif not np.array_equal(reference_predictions, predictions):
            raise ValueError(
                f"Clean predictions differ between chunks in {condition_root}."
            )
        source_chunks.append(str(chunk_root.relative_to(condition_root)))

    if not source_chunks:
        raise ValueError(f"No complete evaluation chunks in {condition_root}.")
    if not clean_rows or any(row != clean_rows[0] for row in clean_rows[1:]):
        raise ValueError(f"Clean evaluation differs between chunks in {condition_root}.")
    keys = [
        (int(row["flip_count_per_event"]), int(row["repeat"]))
        for row in nonzero_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate evaluation rows in {condition_root}.")
    nonzero_rows.sort(
        key=lambda row: (
            int(row["flip_count_per_event"]),
            int(row["repeat"]),
        )
    )
    combined = [clean_rows[0], *nonzero_rows]
    expected_grid = {
        count: 1 if count == 0 else REPEATS
        for count in evaluation_counts(model.internal_fault_sites)
    }
    if dict(Counter(int(row["flip_count_per_event"]) for row in combined)) != expected_grid:
        raise ValueError(f"Combined grid is incomplete in {condition_root}.")

    translated = [
        _translate_raw_row(row, model, training_percentage) for row in combined
    ]
    _write_dict_rows(condition_root / "random_overwrite_results.csv", translated)
    if prediction_source is None:
        raise ValueError(f"No clean prediction source in {condition_root}.")
    shutil.copyfile(prediction_source, condition_root / "clean_predictions.npy")
    metadata = {
        "complete": True,
        "fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "model": asdict(model),
        "training_percentage": training_percentage,
        "training_status": status,
        "evaluation_selected_site_counts": list(
            evaluation_counts(model.internal_fault_sites)
        ),
        "source_chunks": [
            source for source in source_chunks
        ],
        "percentage_definition": (
            "100 * selected internal sites / total eligible internal bit sites."
        ),
        "effective_change_note": (
            "Each selected site receives a uniform random 0/1 target; therefore "
            "the expected number of bits whose value actually changes is K/2."
        ),
    }
    (condition_root / "random_overwrite_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return translated


def _validate_preliminary_rows(
    rows: list[dict],
    model: ModelSpec,
    source: Path,
) -> None:
    """Validate the fields available even when a timeout prevented metadata."""
    if {row["model_alias"] for row in rows} != {model.alias}:
        raise ValueError(f"Wrong model alias in {source}.")
    if {row["registry_name"] for row in rows} != {model.registry_name}:
        raise ValueError(f"Wrong registry in {source}.")
    if {int(row["total_fault_sites_per_event"]) for row in rows} != {
        model.internal_fault_sites
    }:
        raise ValueError(f"Wrong internal-site total in {source}.")
    expected_counts = set(evaluation_counts(model.internal_fault_sites))
    observed_counts = {int(row["flip_count_per_event"]) for row in rows}
    if not observed_counts.issubset(expected_counts):
        raise ValueError(f"Unexpected evaluation count in {source}.")


def _preliminary_condition(
    root: Path,
    model: ModelSpec,
    training_percentage: float,
) -> list[dict]:
    """Collect verified complete points, including those in timed-out chunks."""
    _validate_training(root, model, training_percentage)
    condition_root = _condition_root(root, model, training_percentage)
    clean_row = None
    reference_predictions = None
    accepted: dict[int, tuple[list[dict], str, bool]] = {}

    for chunk_root in sorted((condition_root / "chunks").glob("chunk_*")):
        csv_path = chunk_root / "bit_flip_results.csv"
        predictions_path = chunk_root / "clean_predictions.npy"
        if not csv_path.is_file() or not predictions_path.is_file():
            continue
        rows = _read_rows(csv_path)
        _validate_preliminary_rows(rows, model, csv_path)

        predictions = np.load(predictions_path)
        if reference_predictions is None:
            reference_predictions = predictions
        elif not np.array_equal(reference_predictions, predictions):
            raise ValueError(
                f"Clean predictions differ between chunks in {condition_root}."
            )

        chunk_clean = [
            row for row in rows if int(row["flip_count_per_event"]) == 0
        ]
        if len(chunk_clean) != 1:
            raise ValueError(f"Expected one clean row in {csv_path}.")
        if clean_row is None:
            clean_row = chunk_clean[0]
        elif clean_row != chunk_clean[0]:
            raise ValueError(
                f"Clean evaluation differs between chunks in {condition_root}."
            )

        metadata_path = chunk_root / "bit_flip_metadata.json"
        metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
        artifact_complete = metadata.get("complete") is True
        if artifact_complete:
            chunk_counts = tuple(
                int(count)
                for count in metadata.get("flip_counts_per_event", ())
                if int(count) > 0
            )
            _validate_chunk(rows, metadata, model, chunk_counts, csv_path)

        grouped = defaultdict(list)
        for row in rows:
            selected = int(row["flip_count_per_event"])
            if selected > 0:
                grouped[selected].append(row)
        for selected, group in grouped.items():
            # A preliminary marker is admitted only after all five independent
            # full-test realizations for that exact point have been persisted.
            repeats = {int(row["repeat"]) for row in group}
            expected_seeds = {
                42 + 1_000_003 * selected + repeat for repeat in range(REPEATS)
            }
            observed_seeds = {int(row["fault_seed"]) for row in group}
            if (
                len(group) != REPEATS
                or repeats != set(range(REPEATS))
                or (not artifact_complete and observed_seeds != expected_seeds)
            ):
                continue
            relative_source = str(chunk_root.relative_to(condition_root))
            previous = accepted.get(selected)
            if previous is not None:
                previous_rows, previous_source, previous_complete = previous
                if previous_complete and artifact_complete:
                    raise ValueError(
                        f"Duplicate complete point K={selected} in "
                        f"{previous_source} and {relative_source}."
                    )
                if previous_complete:
                    continue
                if not artifact_complete and previous_rows != group:
                    raise ValueError(
                        f"Conflicting unfinished point K={selected} in "
                        f"{condition_root}."
                    )
            accepted[selected] = (group, relative_source, artifact_complete)

    if clean_row is None or reference_predictions is None:
        raise ValueError(f"No usable evaluation output in {condition_root}.")

    translated_clean = _translate_raw_row(clean_row, model, training_percentage)
    translated_clean.update(
        {
            "source_chunk": "clean reference (verified across available chunks)",
            "source_artifact_complete": True,
            "preliminary_point_complete": True,
        }
    )
    translated = [translated_clean]
    for selected in sorted(accepted):
        group, source, artifact_complete = accepted[selected]
        for row in sorted(group, key=lambda value: int(value["repeat"])):
            output = _translate_raw_row(row, model, training_percentage)
            output.update(
                {
                    "source_chunk": source,
                    "source_artifact_complete": artifact_complete,
                    "preliminary_point_complete": True,
                }
            )
            translated.append(output)
    return translated


def _translate_raw_row(
    row: dict, model: ModelSpec, training_percentage: float
) -> dict:
    selected = int(row["flip_count_per_event"])
    total = int(row["total_fault_sites_per_event"])
    output = {
        "family": model.family,
        "display_name": model.display_name,
        "model_alias": row["model_alias"],
        "registry_name": row["registry_name"],
        "fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "training_overwrite_percentage": training_percentage,
        "selected_sites_per_event": selected,
        "total_internal_bit_sites": total,
        "selected_site_fraction": selected / total,
        "selected_site_percentage": 100 * selected / total,
        "expected_changed_sites_per_event": selected / 2,
        "expected_changed_site_count_std": float(np.sqrt(selected) / 2),
        "expected_changed_fraction": selected / (2 * total),
        "expected_changed_percentage": 50 * selected / total,
        "repeat": int(row["repeat"]),
        "fault_seed": int(row["fault_seed"]),
    }
    for key, value in row.items():
        if key in {
            "model_alias",
            "registry_name",
            "flip_count_per_event",
            "total_fault_sites_per_event",
            "equivalent_fault_fraction",
            "equivalent_fault_percentage",
            "repeat",
            "fault_seed",
        }:
            continue
        output[key] = value
    return output


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t" if path.suffix == ".tsv" else ",",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    critical = 2.776 if len(values) == 5 else 1.96
    return float(critical * values.std(ddof=1) / np.sqrt(len(values)))


def _curve_points(
    rows: list[dict], model: ModelSpec, training_percentage: float, style: dict
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["selected_sites_per_event"])].append(row)
    points = []
    for selected in sorted(grouped):
        group = grouped[selected]
        accuracy = np.asarray(
            [float(row["balanced_accuracy"]) for row in group], dtype=float
        )
        point = {
            "family": model.family,
            "display_name": model.display_name,
            "model_alias": model.alias,
            "registry_name": model.registry_name,
            "training_overwrite_percentage": training_percentage,
            "training_condition": (
                "clean_baseline" if training_percentage == 0 else "fault_aware"
            ),
            "fault_scope": FAULT_SCOPE,
            "fault_operation": FAULT_OPERATION,
            "total_internal_bit_sites": model.internal_fault_sites,
            "selected_sites_per_event": selected,
            "selected_site_percentage": 100 * selected / model.internal_fault_sites,
            "expected_changed_sites_per_event": selected / 2,
            "expected_changed_site_count_std": float(np.sqrt(selected) / 2),
            "expected_changed_percentage": 50 * selected / model.internal_fault_sites,
            "repeat_count": len(group),
            "balanced_accuracy_mean": float(accuracy.mean()),
            "balanced_accuracy_percent": float(100 * accuracy.mean()),
            "balanced_accuracy_std": (
                float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0
            ),
            "balanced_accuracy_ci95": _ci95(accuracy),
            "balanced_accuracy_ci95_percent": 100 * _ci95(accuracy),
            **style,
        }
        if all("source_chunk" in row for row in group):
            point.update(
                {
                    "source_chunks": ";".join(
                        sorted({str(row["source_chunk"]) for row in group})
                    ),
                    "source_artifact_complete": all(
                        bool(row["source_artifact_complete"]) for row in group
                    ),
                    "preliminary_point_complete": all(
                        bool(row["preliminary_point_complete"]) for row in group
                    ),
                }
            )
        points.append(point)
    return points


def _styles() -> dict[float, dict]:
    colors = ["#222222", "#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00"]
    styles = {}
    for index, percentage in enumerate(TRAINING_PERCENTAGES):
        styles[percentage] = {
            "color": colors[index % len(colors)],
            "linestyle": "-" if percentage == 0 else "--",
            "marker": "o" if percentage == 0 else ("s", "^", "D", "v", "P")[
                (index - 1) % 5
            ],
        }
    return styles


def _plot(points: list[dict], output_stem: Path, preliminary: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = _styles()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        for percentage in TRAINING_PERCENTAGES:
            curve = sorted(
                (
                    row
                    for row in points
                    if row["family"] == model.family
                    and float(row["training_overwrite_percentage"]) == percentage
                ),
                key=lambda row: float(row["selected_site_percentage"]),
            )
            style = styles[percentage]
            x = np.asarray([row["selected_site_percentage"] for row in curve])
            y = np.asarray([row["balanced_accuracy_percent"] for row in curve])
            ci = np.asarray([row["balanced_accuracy_ci95_percent"] for row in curve])
            label = (
                "Clean training"
                if percentage == 0
                else f"{percentage:g}% overwrite training"
            )
            axis.plot(
                x,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.5,
                markersize=6.5,
            )
            axis.fill_between(x, y - ci, y + ci, color=style["color"], alpha=0.10)
        axis.set_title("Dense LNN" if model.family == "lnn" else "Dense QKeras", fontsize=19)
        axis.set_xlim(0, 10)
        axis.set_xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        axis.set_xlabel("Selected internal bits overwritten [%]", fontsize=16)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Balanced accuracy [%]", fontsize=16)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        frameon=False,
        fontsize=13,
        handlelength=2.5,
    )
    if preliminary:
        fig.suptitle(
            "PRELIMINARY — complete five-realization points available so far",
            fontsize=16,
            fontweight="bold",
        )
    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.14,
        top=0.86 if preliminary else 0.91,
        wspace=0.08,
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)


def combine_campaign(root: Path, preliminary: bool = False) -> None:
    if not root.is_dir():
        raise ValueError(f"Campaign root does not exist: {root}")
    config = _load_json(root / "campaign_config.json")
    if config.get("fault_scope") != FAULT_SCOPE or config.get("fault_operation") != FAULT_OPERATION:
        raise ValueError("Campaign configuration does not describe this fault protocol.")
    if tuple(config.get("training_percentages", ())) != TRAINING_PERCENTAGES:
        raise ValueError(
            "Training-percentage environment differs from campaign_config.json."
        )

    all_raw_rows = []
    all_points = []
    styles = _styles()
    for model in MODELS:
        for percentage in TRAINING_PERCENTAGES:
            rows = (
                _preliminary_condition(root, model, percentage)
                if preliminary
                else _combine_condition(root, model, percentage)
            )
            all_raw_rows.extend(rows)
            all_points.extend(_curve_points(rows, model, percentage, styles[percentage]))

    prefix = (
        "internal_random_overwrite_preliminary"
        if preliminary
        else "internal_random_overwrite"
    )
    raw_path = root / f"{prefix}_raw_results.csv"
    points_path = root / f"{prefix}_plot_points.csv"
    output_stem = root / (
        "lnn_vs_qkeras_internal_random_overwrite_preliminary"
        if preliminary
        else "lnn_vs_qkeras_internal_random_overwrite"
    )
    _write_dict_rows(raw_path, all_raw_rows)
    _write_dict_rows(points_path, all_points)
    _plot(all_points, output_stem, preliminary=preliminary)
    expected_point_count = (
        len(MODELS) * len(TRAINING_PERCENTAGES) * len(EVALUATION_PERCENTAGES)
    )
    summary = {
        "schema_version": 1,
        "complete": not preliminary,
        "preliminary": preliminary,
        "study": "selected_models_internal_activation_random_overwrite",
        "fault_scope": FAULT_SCOPE,
        "fault_operation": FAULT_OPERATION,
        "training_percentages": list(TRAINING_PERCENTAGES),
        "evaluation_target_percentages": list(EVALUATION_PERCENTAGES),
        "plot_x_range_percent": [0.0, 10.0],
        "repeats_per_nonzero_point": REPEATS,
        "x_axis": (
            "Exact percentage of eligible internal bit sites selected for a "
            "random 0/1 overwrite; expected effective changed-bit rate is half."
        ),
        "models": [asdict(model) for model in MODELS],
        "available_plot_point_count": len(all_points),
        "expected_final_plot_point_count": expected_point_count,
        "raw_results_csv": str(raw_path),
        "plot_points_csv": str(points_path),
        "plot_png": str(output_stem.with_suffix(".png")),
        "plot_pdf": str(output_stem.with_suffix(".pdf")),
    }
    summary_name = (
        "internal_random_overwrite_preliminary_summary.json"
        if preliminary
        else "internal_random_overwrite_campaign_summary.json"
    )
    (root / summary_name).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    label = "preliminary available-point plot" if preliminary else "complete campaign"
    print(f"[DONE] Validated and plotted {label} under {root}")


def status(root: Path) -> None:
    training_expected = len(training_tasks())
    training_found = len(list(root.glob("*/*/*/training/training_complete.json")))
    evaluation_points_expected = len(MODELS) * len(TRAINING_PERCENTAGES) * (
        len(EVALUATION_PERCENTAGES) - 1
    )
    evaluation_points_found = 0
    complete_artifacts = 0
    for metadata_path in root.glob("*/*/*/chunks/chunk_*/bit_flip_metadata.json"):
        metadata = _load_json(metadata_path)
        if not metadata.get("complete"):
            continue
        complete_artifacts += 1
        evaluation_points_found += sum(
            int(count) > 0 for count in metadata.get("flip_counts_per_event", ())
        )
    print(f"Training completions:  {training_found} / {training_expected}")
    print(
        f"Evaluation points:     {evaluation_points_found} / "
        f"{evaluation_points_expected} ({complete_artifacts} complete artifacts)"
    )
    final_plot = root / "lnn_vs_qkeras_internal_random_overwrite.png"
    print(f"Final plot:            {'READY' if final_plot.is_file() else 'not yet produced'}")


def _completed_evaluation_counts(
    root: Path,
    model: ModelSpec,
    training_percentage: float,
) -> set[int]:
    """Return validated nonzero counts already completed for one condition."""
    condition_root = _condition_root(root, model, training_percentage)
    completed = set()
    for chunk_root in sorted((condition_root / "chunks").glob("chunk_*")):
        metadata_path = chunk_root / "bit_flip_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = _load_json(metadata_path)
        if not metadata.get("complete"):
            continue
        counts = tuple(
            int(count)
            for count in metadata.get("flip_counts_per_event", ())
            if int(count) > 0
        )
        rows = _read_rows(chunk_root / "bit_flip_results.csv")
        _validate_chunk(
            rows,
            metadata,
            model,
            counts,
            chunk_root / "bit_flip_results.csv",
        )
        overlap = completed.intersection(counts)
        if overlap:
            raise ValueError(
                f"Duplicate completed counts {sorted(overlap)} in {condition_root}."
            )
        completed.update(counts)
    return completed


def qkeras_retry_tasks(root: Path) -> list[dict]:
    """Build one-point retry tasks for missing QKeras evaluations only."""
    model = next(model for model in MODELS if model.family == "qkeras")
    tasks = []
    expected = set(evaluation_counts(model.internal_fault_sites)[1:])
    for training_percentage in TRAINING_PERCENTAGES:
        completed = _completed_evaluation_counts(
            root,
            model,
            training_percentage,
        )
        for count in sorted(expected - completed):
            tasks.append(
                {
                    **asdict(model),
                    "training_percentage": training_percentage,
                    "tag": percentage_tag(training_percentage),
                    "selected_site_counts": (count,),
                }
            )
    return tasks


def write_qkeras_retry_manifest(root: Path) -> Path:
    tasks = qkeras_retry_tasks(root)
    path = root / "qkeras_evaluation_retry_manifest.tsv"
    rows = [
        {
            "task_index": index,
            "family": task["family"],
            "tag": task["tag"],
            "training_percentage": task["training_percentage"],
            "model_alias": task["alias"],
            "registry_name": task["registry_name"],
            "selected_site_count": task["selected_site_counts"][0],
            "minimum_training_hours": task["minimum_training_hours"],
            "minimum_training_steps": task["minimum_training_steps"],
        }
        for index, task in enumerate(tasks)
    ]
    if rows:
        _write_dict_rows(path, rows)
    else:
        path.write_text(
            "task_index\tfamily\ttag\ttraining_percentage\tmodel_alias\t"
            "registry_name\tselected_site_count\tminimum_training_hours\t"
            "minimum_training_steps\n"
        )
    print(f"[RETRY] Missing QKeras evaluation points: {len(tasks)}")
    print(f"[RETRY] Manifest: {path}")
    return path


def _read_qkeras_retry_manifest(root: Path) -> list[dict]:
    path = root / "qkeras_evaluation_retry_manifest.tsv"
    if not path.is_file():
        raise ValueError(f"Missing retry manifest: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def qkeras_retry_task_text(root: Path, task_id: int) -> str:
    task = _task_at(_read_qkeras_retry_manifest(root), task_id, "QKeras retry")
    return "\t".join(
        (
            task["family"],
            task["tag"],
            task["training_percentage"],
            task["model_alias"],
            task["registry_name"],
            task["selected_site_count"],
            task["minimum_training_hours"],
            task["minimum_training_steps"],
        )
    )


def qkeras_preflight() -> None:
    """Fail before submission if this environment cannot construct QKeras."""
    QActivation, QBatchNormalization, QDense, quantized_bits = (
        _import_qkeras_layers()
    )
    model = build_model(QKERAS_CONFIG.registry_name)
    imported = {
        "QActivation": QActivation,
        "QBatchNormalization": QBatchNormalization,
        "QDense": QDense,
        "quantized_bits": quantized_bits,
    }
    for expected_name, imported_object in imported.items():
        if imported_object.__name__ != expected_name:
            raise ValueError(
                f"QKeras import mismatch: expected {expected_name}, "
                f"got {imported_object.__name__}."
            )
    if tuple(model.input_shape) != (None, QKERAS_CONFIG.input_dim):
        raise ValueError(f"Unexpected QKeras input shape: {model.input_shape}.")
    if tuple(model.output_shape) != (None, 3):
        raise ValueError(f"Unexpected QKeras output shape: {model.output_shape}.")
    print(
        "QKeras preflight passed: "
        f"{QKERAS_CONFIG.registry_name} {model.input_shape} -> {model.output_shape}"
    )


def _task_text(task: dict, evaluation: bool = False) -> str:
    fields = [
        task["family"],
        task["tag"],
        f"{task['training_percentage']:g}",
        task["alias"],
        task["registry_name"],
    ]
    if evaluation:
        fields.extend(
            (
                f"{task['chunk_index']:02d}",
                " ".join(map(str, task["selected_site_counts"])),
            )
        )
    fields.extend(
        (
            f"{task['minimum_training_hours']:g}",
            str(task["minimum_training_steps"]),
        )
    )
    if not evaluation:
        fields.insert(3, f"{task['training_probability']:.12g}")
    return "\t".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--training-task-id", type=int)
    actions.add_argument("--evaluation-task-id", type=int)
    actions.add_argument("--print-training-task-count", action="store_true")
    actions.add_argument("--print-evaluation-task-count", action="store_true")
    actions.add_argument("--print-model-grids", action="store_true")
    actions.add_argument("--write-campaign-config", action="store_true")
    actions.add_argument("--combine", action="store_true")
    actions.add_argument("--combine-partial", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--qkeras-preflight", action="store_true")
    actions.add_argument("--write-qkeras-retry-manifest", action="store_true")
    actions.add_argument("--print-qkeras-retry-task-count", action="store_true")
    actions.add_argument("--qkeras-retry-task-id", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    try:
        if args.training_task_id is not None:
            print(_task_text(training_configuration(args.training_task_id)))
        elif args.evaluation_task_id is not None:
            print(_task_text(evaluation_configuration(args.evaluation_task_id), True))
        elif args.print_training_task_count:
            print(len(training_tasks()))
        elif args.print_evaluation_task_count:
            print(len(evaluation_tasks()))
        elif args.print_model_grids:
            for model in MODELS:
                print(
                    "\t".join(
                        (
                            model.family,
                            model.alias,
                            str(model.internal_fault_sites),
                            " ".join(map(str, evaluation_counts(model.internal_fault_sites))),
                        )
                    )
                )
        elif args.write_campaign_config:
            if args.root is None or args.git_commit is None:
                parser.error("--root and --git-commit are required.")
            write_campaign_config(args.root, args.git_commit)
        elif args.combine:
            if args.root is None:
                parser.error("--root is required.")
            combine_campaign(args.root)
        elif args.combine_partial:
            if args.root is None:
                parser.error("--root is required.")
            combine_campaign(args.root, preliminary=True)
        elif args.status:
            if args.root is None:
                parser.error("--root is required.")
            status(args.root)
        elif args.qkeras_preflight:
            qkeras_preflight()
        elif args.write_qkeras_retry_manifest:
            if args.root is None:
                parser.error("--root is required.")
            write_qkeras_retry_manifest(args.root)
        elif args.print_qkeras_retry_task_count:
            if args.root is None:
                parser.error("--root is required.")
            print(len(_read_qkeras_retry_manifest(args.root)))
        elif args.qkeras_retry_task_id is not None:
            if args.root is None:
                parser.error("--root is required.")
            print(qkeras_retry_task_text(args.root, args.qkeras_retry_task_id))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
