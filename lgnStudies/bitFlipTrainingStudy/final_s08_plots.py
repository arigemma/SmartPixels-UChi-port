#!/usr/bin/env python3
"""Write the two final S08 fault-scope plots into a simple temp directory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/smartpixels-matplotlib")

import numpy as np

from lgnStudies.bitFlipStudy.campaign import REPEATS
from lgnStudies.bitFlipStudy.combine_results import read_csv
from lgnStudies.bitFlipStudy.plots import (
    _aggregate,
    plot_training_percentage_sweep_2d,
)
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MODEL_ALIAS,
    parse_training_percentages,
)


TAG_PATTERN = re.compile(r"^train_(?P<percentage>.+)pct$")


@dataclass
class Candidate:
    percentage: float | None
    model_root: Path
    rows: list[dict]
    source: str
    maximum_inference_percentage: float


def parse_tag_percentage(tag: str) -> float | None:
    if tag == "inference_only":
        return None
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"Not an S08 percentage tag: {tag}")
    return float(match.group("percentage").replace("p", "."))


def _chunk_index(chunk_root: Path) -> int:
    match = re.fullmatch(r"chunk_(\d+)", chunk_root.name)
    if match is None:
        raise ValueError(f"Unexpected chunk directory name: {chunk_root}")
    return int(match.group(1))


def _validate_rows_for_chunk(rows: list[dict], chunk_root: Path, model_alias: str) -> None:
    aliases = {row["model_alias"] for row in rows}
    if aliases != {model_alias}:
        raise ValueError(f"Wrong model alias in {chunk_root}: {aliases}")

    observed = Counter(int(row["flip_count_per_event"]) for row in rows)
    if observed.get(0) != 1:
        raise ValueError(f"Expected one K=0 row in {chunk_root}, observed {observed}")
    for count, row_count in observed.items():
        if count > 0 and row_count != REPEATS:
            raise ValueError(
                f"Expected {REPEATS} repeats for K={count} in {chunk_root}, "
                f"observed {row_count}"
            )


def _load_rows_from_chunks(
    model_root: Path,
    model_alias: str,
    expected_chunks: int | None,
) -> tuple[list[dict], str]:
    chunk_parent = model_root / "chunks"
    chunk_roots = sorted(
        (path for path in chunk_parent.glob("chunk_*") if path.is_dir()),
        key=_chunk_index,
    )
    if not chunk_roots:
        raise ValueError(f"No chunk directories under {chunk_parent}")

    chunk_indices = [_chunk_index(path) for path in chunk_roots]
    if chunk_indices != list(range(len(chunk_indices))):
        raise ValueError(f"Non-contiguous chunk set under {chunk_parent}: {chunk_indices}")
    if expected_chunks is not None and len(chunk_roots) != expected_chunks:
        raise ValueError(
            f"Expected {expected_chunks} chunks under {chunk_parent}, "
            f"observed {len(chunk_roots)}"
        )

    clean_rows: list[dict] = []
    clean_predictions: np.ndarray | None = None
    nonzero_rows: list[dict] = []
    for chunk_root in chunk_roots:
        csv_path = chunk_root / "bit_flip_results.csv"
        predictions_path = chunk_root / "clean_predictions.npy"
        metadata_path = chunk_root / "bit_flip_metadata.json"
        for required in (csv_path, predictions_path):
            if not required.is_file():
                raise ValueError(f"Missing completed chunk artifact: {required}")

        rows = read_csv(csv_path)
        _validate_rows_for_chunk(rows, chunk_root, model_alias)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("complete") is not True:
                raise ValueError(f"Chunk metadata is not complete: {metadata_path}")

        predictions = np.load(predictions_path)
        if clean_predictions is None:
            clean_predictions = predictions
        elif not np.array_equal(clean_predictions, predictions):
            raise ValueError(f"K=0 predictions differ in {predictions_path}")

        clean_rows.extend(row for row in rows if int(row["flip_count_per_event"]) == 0)
        nonzero_rows.extend(
            row for row in rows if int(row["flip_count_per_event"]) > 0
        )

    if any(row != clean_rows[0] for row in clean_rows[1:]):
        raise ValueError(f"K=0 rows differ between chunks under {model_root}")

    keys = [
        (int(row["flip_count_per_event"]), int(row["repeat"]))
        for row in nonzero_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate nonzero K/repeat rows under {model_root}")
    nonzero_rows.sort(
        key=lambda row: (
            int(row["flip_count_per_event"]),
            int(row["repeat"]),
        )
    )
    return [clean_rows[0], *nonzero_rows], "chunks"


def _load_rows(
    model_root: Path,
    model_alias: str,
    expected_chunks: int | None,
) -> tuple[list[dict], str]:
    try:
        return _load_rows_from_chunks(model_root, model_alias, expected_chunks)
    except ValueError:
        csv_path = model_root / "bit_flip_results.csv"
        if csv_path.is_file():
            return read_csv(csv_path), "combined_csv"
        raise


def _filter_rows(rows: list[dict], max_inference_percentage: float) -> list[dict]:
    filtered = [
        row
        for row in rows
        if float(row["equivalent_fault_percentage"]) <= max_inference_percentage + 1e-9
    ]
    if len({int(row["flip_count_per_event"]) for row in filtered}) < 2:
        raise ValueError("No nonzero inference-fault points remain after filtering.")
    return filtered


def _maximum_inference_percentage(rows: list[dict]) -> float:
    return max(float(row["equivalent_fault_percentage"]) for row in rows)


def _percentage_is_requested(
    percentage: float | None,
    requested_percentages: tuple[float, ...] | None,
) -> bool:
    if percentage is None or requested_percentages is None:
        return True
    return any(
        np.isclose(percentage, requested, rtol=1e-9, atol=1e-12)
        for requested in requested_percentages
    )


def _candidate_patterns(scope: str, model_alias: str) -> tuple[str, ...]:
    if scope == "whole":
        return (
            f"*/{model_alias}",
            f"*/*/{model_alias}",
            f"whole_scope/*/{model_alias}",
            f"*/whole_scope/*/{model_alias}",
        )
    if scope == "input":
        return (
            f"input_scope/*/{model_alias}",
            f"*/input_scope/*/{model_alias}",
        )
    raise ValueError(f"Unknown scope: {scope}")


def _is_scope_candidate(scope: str, model_root: Path) -> bool:
    parent_names = {parent.name for parent in model_root.parents}
    if scope == "input":
        return "input_scope" in parent_names
    if scope == "whole":
        return "input_scope" not in parent_names
    raise ValueError(f"Unknown scope: {scope}")


def discover_candidates(
    search_roots: tuple[Path, ...],
    scope: str,
    model_alias: str,
    requested_training_percentages: tuple[float, ...] | None,
    max_training_percentage: float,
    max_inference_percentage: float,
    expected_chunks: int | None,
) -> tuple[dict[float | None, Candidate], list[str]]:
    candidates_by_percentage: dict[float | None, list[Candidate]] = {}
    skipped: list[str] = []
    seen: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            skipped.append(f"{search_root}: does not exist")
            continue
        for pattern in _candidate_patterns(scope, model_alias):
            for model_root in search_root.glob(pattern):
                if not model_root.is_dir() or model_root in seen:
                    continue
                seen.add(model_root)
                if not _is_scope_candidate(scope, model_root):
                    continue

                tag = model_root.parent.name
                try:
                    percentage = parse_tag_percentage(tag)
                except ValueError:
                    continue
                if not _percentage_is_requested(
                    percentage,
                    requested_training_percentages,
                ):
                    continue
                if percentage is not None and percentage > max_training_percentage:
                    continue

                try:
                    rows, source = _load_rows(
                        model_root,
                        model_alias,
                        expected_chunks,
                    )
                    rows = _filter_rows(rows, max_inference_percentage)
                except ValueError as error:
                    skipped.append(f"{model_root}: {error}")
                    continue

                candidate = Candidate(
                    percentage=percentage,
                    model_root=model_root,
                    rows=rows,
                    source=source,
                    maximum_inference_percentage=_maximum_inference_percentage(rows),
                )
                candidates_by_percentage.setdefault(percentage, []).append(candidate)

    selected = {
        percentage: max(
            candidates,
            key=lambda candidate: (
                candidate.maximum_inference_percentage,
                str(candidate.model_root),
            ),
        )
        for percentage, candidates in candidates_by_percentage.items()
    }
    return selected, skipped


def _require_plot_inputs(scope: str, selected: dict[float | None, Candidate]) -> None:
    if None not in selected:
        raise SystemExit(
            f"No completed {scope} baseline found. Expected an inference_only/"
            f"{MODEL_ALIAS} result with chunks or bit_flip_results.csv."
        )
    percentages = [percentage for percentage in selected if percentage is not None]
    if not percentages:
        raise SystemExit(
            f"No completed {scope} fault-aware training percentages found."
        )


def _point_rows_for_candidate(
    scope: str,
    curve_label: str,
    training_percentage: float | None,
    candidate: Candidate,
) -> list[dict]:
    values = _aggregate(candidate.rows)
    sample_counts = Counter(
        int(row["flip_count_per_event"]) for row in candidate.rows
    )
    rows = []
    for index, count in enumerate(values["count"]):
        count_int = int(count)
        rows.append(
            {
                "scope": scope,
                "curve_label": curve_label,
                "training_percentage": ""
                if training_percentage is None
                else f"{training_percentage:.12g}",
                "flip_count_per_event": count_int,
                "equivalent_fault_percentage": f"{values['percentage'][index]:.12g}",
                "balanced_accuracy_percent": (
                    f"{100 * values['balanced_accuracy_mean'][index]:.12g}"
                ),
                "balanced_accuracy_ci95_percent": (
                    f"{100 * values['balanced_accuracy_ci95'][index]:.12g}"
                ),
                "sample_count": sample_counts[count_int],
                "source": str(candidate.model_root),
            }
        )
    return rows


def _write_points_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No point rows supplied for {path}.")
    fieldnames = (
        "scope",
        "curve_label",
        "training_percentage",
        "flip_count_per_event",
        "equivalent_fault_percentage",
        "balanced_accuracy_percent",
        "balanced_accuracy_ci95_percent",
        "sample_count",
        "source",
    )
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _scope_point_rows(
    scope: str,
    selected: dict[float | None, Candidate],
    training_candidates: list[tuple[float, Candidate]],
) -> list[dict]:
    rows = _point_rows_for_candidate(
        scope,
        "Baseline clean training",
        None,
        selected[None],
    )
    for percentage, candidate in training_candidates:
        rows.extend(
            _point_rows_for_candidate(
                scope,
                f"Fault-aware training {percentage:g}%",
                percentage,
                candidate,
            )
        )
    return rows


def write_scope_plot(
    scope: str,
    selected: dict[float | None, Candidate],
    output_stem: Path,
    title: str | None,
    model_alias: str,
    x_limit: float,
    fit_exponential_decay: bool,
    require_x_coverage: bool,
    x_label: str,
    show_title: bool,
    paper_style: bool,
) -> dict:
    _require_plot_inputs(scope, selected)
    inference_rows = selected[None].rows
    training_candidates = sorted(
        (
            (float(percentage), candidate)
            for percentage, candidate in selected.items()
            if percentage is not None
        ),
        key=lambda item: item[0],
    )
    if require_x_coverage:
        coverage_items = [("baseline", selected[None])]
        coverage_items.extend(
            (f"{percentage:g}%", candidate)
            for percentage, candidate in training_candidates
        )
        shortfalls = [
            (
                f"{label} reaches {candidate.maximum_inference_percentage:g}% "
                f"from {candidate.model_root}"
            )
            for label, candidate in coverage_items
            if candidate.maximum_inference_percentage + 1e-6 < x_limit
        ]
        if shortfalls:
            raise SystemExit(
                f"{scope} plot cannot cover requested x-axis limit {x_limit:g}%:\n"
                + "\n".join(f"  - {item}" for item in shortfalls)
            )
    fault_aware_rows = {
        percentage: candidate.rows for percentage, candidate in training_candidates
    }
    plot_training_percentage_sweep_2d(
        inference_rows,
        fault_aware_rows,
        output_stem,
        model_alias,
        title=title,
        x_limit=x_limit,
        fit_exponential_decay=fit_exponential_decay,
        x_label=x_label,
        show_title=show_title,
        paper_style=paper_style,
    )
    point_rows = _scope_point_rows(scope, selected, training_candidates)
    points_csv = output_stem.with_name(output_stem.name + "_points.csv")
    _write_points_csv(points_csv, point_rows)
    return {
        "baseline_source": str(selected[None].model_root),
        "training_percentages": sorted(fault_aware_rows),
        "points_csv": str(points_csv),
        "training_sources": {
            f"{percentage:g}": str(candidate.model_root)
            for percentage, candidate in training_candidates
        },
        "maximum_inference_percentage_in_plot": {
            "baseline": selected[None].maximum_inference_percentage,
            **{
                f"{percentage:g}": candidate.maximum_inference_percentage
                for percentage, candidate in training_candidates
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover completed S08 percentage-sweep chunks and write two final "
            "2D plots into temp/."
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        help="Root to scan. Defaults to results/SLURM.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("temp"))
    parser.add_argument("--model-alias", default=MODEL_ALIAS)
    parser.add_argument(
        "--training-percentages",
        help=(
            "Optional exact fault-aware training percentages to include. "
            "Defaults to all discovered percentages up to "
            "--max-training-percentage."
        ),
    )
    parser.add_argument("--max-training-percentage", type=float, default=7.0)
    parser.add_argument("--max-inference-percentage", type=float, default=7.0)
    parser.add_argument(
        "--expected-chunks",
        type=int,
        default=7,
        help="Expected chunks per completed S08 tag. Use 0 to accept any count.",
    )
    parser.add_argument("--fit-exponential-decay", action="store_true")
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help=(
            "Use larger text, scope-specific x-axis labels, and no in-figure "
            "titles for paper figures."
        ),
    )
    parser.add_argument(
        "--allow-partial-x-coverage",
        action="store_true",
        help=(
            "Write plots even when selected curves do not reach the requested "
            "--max-inference-percentage. By default this is treated as an error."
        ),
    )
    args = parser.parse_args()

    if args.max_training_percentage <= 0 or args.max_inference_percentage <= 0:
        parser.error("Maximum percentages must be positive.")
    requested_training_percentages = (
        parse_training_percentages(args.training_percentages)
        if args.training_percentages
        else None
    )
    expected_chunks = args.expected_chunks if args.expected_chunks > 0 else None
    search_roots = tuple(args.search_root or (Path("results/SLURM"),))

    whole, whole_skipped = discover_candidates(
        search_roots,
        "whole",
        args.model_alias,
        requested_training_percentages,
        args.max_training_percentage,
        args.max_inference_percentage,
        expected_chunks,
    )
    input_scope, input_skipped = discover_candidates(
        search_roots,
        "input",
        args.model_alias,
        requested_training_percentages,
        args.max_training_percentage,
        args.max_inference_percentage,
        expected_chunks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    whole_summary = write_scope_plot(
        "whole",
        whole,
        args.output_dir / "s08_internal_bit_flips_training_percentage_sweep",
        None
        if args.paper_style
        else (
            "S08 internal/whole-model bit flips: baseline vs fault-aware "
            "training percentages"
        ),
        args.model_alias,
        args.max_inference_percentage,
        args.fit_exponential_decay,
        not args.allow_partial_x_coverage,
        "Flipped internal bits per event (%)",
        not args.paper_style,
        args.paper_style,
    )
    input_summary = write_scope_plot(
        "input",
        input_scope,
        args.output_dir / "s08_input_bit_flips_training_percentage_sweep",
        None
        if args.paper_style
        else "S08 input bit flips: baseline vs fault-aware training percentages",
        args.model_alias,
        args.max_inference_percentage,
        args.fit_exponential_decay,
        not args.allow_partial_x_coverage,
        "Flipped input bits per event (%)",
        not args.paper_style,
        args.paper_style,
    )
    combined_points_csv = args.output_dir / "s08_final_plot_points.csv"
    combined_point_rows = []
    for summary_key in (whole_summary, input_summary):
        with Path(summary_key["points_csv"]).open(newline="") as handle:
            combined_point_rows.extend(csv.DictReader(handle))
    _write_points_csv(combined_points_csv, combined_point_rows)

    summary = {
        "model_alias": args.model_alias,
        "search_roots": [str(path) for path in search_roots],
        "output_dir": str(args.output_dir),
        "requested_training_percentages": (
            list(requested_training_percentages)
            if requested_training_percentages is not None
            else None
        ),
        "max_training_percentage": args.max_training_percentage,
        "max_inference_percentage": args.max_inference_percentage,
        "paper_style": args.paper_style,
        "combined_points_csv": str(combined_points_csv),
        "whole_model_plot": whole_summary,
        "input_scope_plot": input_summary,
        "skipped_candidates": {
            "whole": whole_skipped[:50],
            "input": input_skipped[:50],
        },
    }
    (args.output_dir / "s08_final_plot_sources.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("[DONE] Wrote final S08 plots:")
    print(args.output_dir / "s08_internal_bit_flips_training_percentage_sweep.png")
    print(args.output_dir / "s08_input_bit_flips_training_percentage_sweep.png")
    print(combined_points_csv)
    print(args.output_dir / "s08_final_plot_sources.json")


if __name__ == "__main__":
    main()
