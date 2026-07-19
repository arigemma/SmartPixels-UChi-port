#!/usr/bin/env python3
"""Validate and combine the chunked exact-count bit-flip campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from lgnStudies.bitFlipStudy.campaign import (
    EXPECTED_FLIP_COUNTS,
    FLIP_COUNT_CHUNKS,
    MODELS,
    REPEATS,
)
from lgnStudies.bitFlipStudy.plots import (
    plot_combined_models,
    plot_single_model,
)
from lgnStudies.bitFlipStudy.study import CSV_FIELDS, write_rows


MODEL_ORDER = tuple(model_alias for model_alias, _ in MODELS)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result CSV is empty: {path}")
    if tuple(rows[0]) != CSV_FIELDS:
        raise ValueError(f"Unexpected CSV schema in {path}.")
    return rows


def _validate_chunk_rows(
    rows: list[dict],
    model_alias: str,
    expected_nonzero_counts: tuple[int, ...],
    path: Path,
) -> None:
    if {row["model_alias"] for row in rows} != {model_alias}:
        raise ValueError(f"Wrong model alias in {path}.")
    expected_counts = {0, *expected_nonzero_counts}
    observed_counts = {int(row["flip_count_per_event"]) for row in rows}
    if observed_counts != expected_counts:
        raise ValueError(
            f"Wrong counts in {path}: expected {sorted(expected_counts)}, "
            f"observed {sorted(observed_counts)}."
        )
    observed_rows = Counter(int(row["flip_count_per_event"]) for row in rows)
    expected_rows = {0: 1, **{count: REPEATS for count in expected_nonzero_counts}}
    if dict(observed_rows) != expected_rows:
        raise ValueError(
            f"Wrong repeat counts in {path}: expected {expected_rows}, "
            f"observed {dict(observed_rows)}."
        )


def _metadata_from_complete_rows(
    rows: list[dict],
    expected_counts: tuple[int, ...],
    chunk_root: Path,
) -> dict:
    """Recover minimal chunk metadata when Slurm ended after writing the CSV."""
    first = rows[0]
    confusion_matrices = {}
    for row in rows:
        count = int(row["flip_count_per_event"])
        repeat = int(row["repeat"])
        key = f"K={count}:repeat={repeat}"
        confusion_matrices[key] = [
            [
                int(row[f"confusion_{truth}_{prediction}"])
                for prediction in range(3)
            ]
            for truth in range(3)
        ]
    return {
        "model_alias": first["model_alias"],
        "registry_name": first["registry_name"],
        "flip_counts_per_event": [0, *expected_counts],
        "total_fault_sites_per_event": int(first["total_fault_sites_per_event"]),
        "repeats_per_nonzero_count": REPEATS,
        "number_test_events": sum(
            int(first[f"n_class_{index}"]) for index in range(3)
        ),
        "complete": True,
        "metadata_recovered_from_complete_csv": True,
        "metadata_recovery_source": str(chunk_root / "bit_flip_results.csv"),
        "confusion_matrices": confusion_matrices,
    }


def load_and_combine_model_chunks(
    model_root: Path,
    model_alias: str,
    flip_count_chunks: tuple[tuple[int, ...], ...] = FLIP_COUNT_CHUNKS,
    expected_flip_counts: tuple[int, ...] = EXPECTED_FLIP_COUNTS,
) -> tuple[list[dict], dict, Path]:
    clean_rows: list[dict] = []
    clean_predictions: np.ndarray | None = None
    clean_prediction_source: Path | None = None
    nonzero_rows: list[dict] = []
    metadata_by_chunk: list[dict] = []
    registry_names: set[str] = set()
    total_fault_sites: set[int] = set()

    for chunk_index, expected_counts in enumerate(flip_count_chunks):
        chunk_root = model_root / "chunks" / f"chunk_{chunk_index:02d}"
        csv_path = chunk_root / "bit_flip_results.csv"
        metadata_path = chunk_root / "bit_flip_metadata.json"
        predictions_path = chunk_root / "clean_predictions.npy"
        for required in (csv_path, predictions_path):
            if not required.is_file():
                raise ValueError(f"Missing completed chunk artifact: {required}")

        rows = read_csv(csv_path)
        _validate_chunk_rows(rows, model_alias, expected_counts, csv_path)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = _metadata_from_complete_rows(
                rows,
                expected_counts,
                chunk_root,
            )
        if not metadata.get("complete"):
            raise ValueError(f"Study is marked incomplete: {metadata_path}")
        if tuple(metadata.get("flip_counts_per_event", ())) != (
            0,
            *expected_counts,
        ):
            raise ValueError(f"Metadata count mismatch: {metadata_path}")

        chunk_predictions = np.load(predictions_path)
        if clean_predictions is None:
            clean_predictions = chunk_predictions
            clean_prediction_source = predictions_path
        elif not np.array_equal(clean_predictions, chunk_predictions):
            raise ValueError(
                f"K=0 predictions differ between chunks for {model_alias}: "
                f"{predictions_path}"
            )

        chunk_clean = [row for row in rows if int(row["flip_count_per_event"]) == 0]
        clean_rows.extend(chunk_clean)
        nonzero_rows.extend(
            row for row in rows if int(row["flip_count_per_event"]) > 0
        )
        metadata_by_chunk.append(metadata)
        registry_names.update(row["registry_name"] for row in rows)
        total_fault_sites.update(int(row["total_fault_sites_per_event"]) for row in rows)

    if len(registry_names) != 1:
        raise ValueError(f"Registry names differ between chunks: {registry_names}")
    if len(total_fault_sites) != 1:
        raise ValueError(
            f"Fault-site totals differ between chunks: {total_fault_sites}"
        )
    if any(row != clean_rows[0] for row in clean_rows[1:]):
        raise ValueError(f"K=0 metrics differ between chunks for {model_alias}.")

    keys = [
        (int(row["flip_count_per_event"]), int(row["repeat"]))
        for row in nonzero_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate nonzero count/repeat rows for {model_alias}.")
    nonzero_rows.sort(
        key=lambda row: (
            int(row["flip_count_per_event"]),
            int(row["repeat"]),
        )
    )
    combined_rows = [clean_rows[0], *nonzero_rows]
    observed_counts = tuple(
        sorted({int(row["flip_count_per_event"]) for row in combined_rows})
    )
    if observed_counts != expected_flip_counts:
        raise ValueError(
            f"Incomplete combined sweep for {model_alias}: {observed_counts}"
        )

    combined_confusions = {}
    for metadata in metadata_by_chunk:
        combined_confusions.update(metadata.get("confusion_matrices", {}))
    combined_metadata = {
        **metadata_by_chunk[0],
        "flip_counts_per_event": list(expected_flip_counts),
        "repeats_per_nonzero_count": REPEATS,
        "chunk_count": len(flip_count_chunks),
        "source_chunks": [
            f"chunks/chunk_{index:02d}" for index in range(len(flip_count_chunks))
        ],
        "complete": True,
        "confusion_matrices": combined_confusions,
    }
    if clean_prediction_source is None:
        raise ValueError(f"No clean predictions found for {model_alias}.")
    return combined_rows, combined_metadata, clean_prediction_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--title-context",
        help="Optional provenance line appended to plot titles.",
    )
    args = parser.parse_args()

    rows_by_model = {}
    for model_alias in MODEL_ORDER:
        model_root = args.root / model_alias
        try:
            rows, metadata, prediction_source = load_and_combine_model_chunks(
                model_root, model_alias
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

        model_root.mkdir(parents=True, exist_ok=True)
        write_rows(model_root / "bit_flip_results.csv", rows)
        (model_root / "bit_flip_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copyfile(prediction_source, model_root / "clean_predictions.npy")
        plot_single_model(
            rows,
            model_alias,
            model_root / "bit_flips_per_event",
            x_mode="count",
            title_context=args.title_context,
        )
        plot_single_model(
            rows,
            model_alias,
            model_root / "bit_flip_percentage",
            x_mode="percentage",
            title_context=args.title_context,
        )
        rows_by_model[model_alias] = rows

    plot_combined_models(
        rows_by_model,
        args.root / "bit_flips_per_event_all_models",
        x_mode="count",
        title_context=args.title_context,
    )
    plot_combined_models(
        rows_by_model,
        args.root / "bit_flip_percentage_all_models",
        x_mode="percentage",
        title_context=args.title_context,
    )
    print(f"[DONE] Validated and combined all chunks under {args.root}")


if __name__ == "__main__":
    main()
