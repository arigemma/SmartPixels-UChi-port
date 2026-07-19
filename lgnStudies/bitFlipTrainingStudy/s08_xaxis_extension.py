#!/usr/bin/env python3
"""Audit missing S08 final chunks for x-axis extension runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from lgnStudies.bitFlipStudy.campaign import REPEATS


@dataclass(frozen=True)
class ScopeLayout:
    scope: str
    root_prefix: Path
    model_alias: str
    chunks: tuple[tuple[int, ...], ...]
    percentages: tuple[float, ...]


@dataclass(frozen=True)
class ChunkTask:
    task_id: int
    condition: str
    tag: str
    training_percentage: float | None
    model_root: Path
    chunk_index: int
    flip_counts: tuple[int, ...]
    complete: bool
    reason: str

    def as_report_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "tag": self.tag,
            "training_percentage": self.training_percentage,
            "model_root": str(self.model_root),
            "chunk_index": self.chunk_index,
            "flip_counts": list(self.flip_counts),
            "complete": self.complete,
            "reason": self.reason,
        }


def _set_max_evaluation_percentage(value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError("Maximum evaluation percentage must lie in (0, 100].")
    os.environ["S08_MAX_EVALUATION_PERCENTAGE"] = f"{value:.12g}"


def _scope_layout(
    scope: str,
    root: Path,
    training_percentages: str,
) -> ScopeLayout:
    if scope == "whole":
        from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
            MODEL_ALIAS,
            evaluation_flip_count_chunks,
            parse_training_percentages,
        )

        return ScopeLayout(
            scope=scope,
            root_prefix=root,
            model_alias=MODEL_ALIAS,
            chunks=evaluation_flip_count_chunks(),
            percentages=parse_training_percentages(training_percentages),
        )
    if scope == "input":
        from lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep import (
            INPUT_SCOPE_DIR,
            input_evaluation_flip_count_chunks,
        )
        from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
            MODEL_ALIAS,
            parse_training_percentages,
        )

        return ScopeLayout(
            scope=scope,
            root_prefix=root / INPUT_SCOPE_DIR,
            model_alias=MODEL_ALIAS,
            chunks=input_evaluation_flip_count_chunks(),
            percentages=parse_training_percentages(training_percentages),
        )
    raise ValueError(f"Unknown S08 fault scope {scope!r}; use 'whole' or 'input'.")


def _percentage_tag(percentage: float | None) -> str:
    from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import percentage_tag

    return percentage_tag(percentage)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result CSV is empty: {path}")
    return rows


def _chunk_completion(
    model_root: Path,
    model_alias: str,
    chunk_index: int,
    expected_nonzero_counts: tuple[int, ...],
) -> tuple[bool, str]:
    chunk_root = model_root / "chunks" / f"chunk_{chunk_index:02d}"
    csv_path = chunk_root / "bit_flip_results.csv"
    predictions_path = chunk_root / "clean_predictions.npy"
    metadata_path = chunk_root / "bit_flip_metadata.json"
    if not csv_path.is_file():
        return False, f"missing {csv_path}"
    if not predictions_path.is_file():
        return False, f"missing {predictions_path}"

    rows = _read_rows(csv_path)
    aliases = {row.get("model_alias") for row in rows}
    if aliases != {model_alias}:
        return False, f"wrong model aliases in {csv_path}: {sorted(aliases)}"

    observed_rows = Counter(int(row["flip_count_per_event"]) for row in rows)
    expected_rows = {0: 1, **{count: REPEATS for count in expected_nonzero_counts}}
    if dict(observed_rows) != expected_rows:
        return (
            False,
            "wrong flip-count/repeat grid in "
            f"{csv_path}: expected {expected_rows}, observed {dict(observed_rows)}",
        )

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        expected_metadata_counts = (0, *expected_nonzero_counts)
        if metadata.get("complete") is not True:
            return False, f"incomplete metadata in {metadata_path}"
        if tuple(metadata.get("flip_counts_per_event", ())) != expected_metadata_counts:
            return False, f"metadata count mismatch in {metadata_path}"

    return True, "complete"


def audit_final_chunk_tasks(
    scope: str,
    root: Path,
    training_percentages: str,
    include_baseline: bool = True,
) -> tuple[ScopeLayout, list[ChunkTask]]:
    if not root.is_dir():
        raise FileNotFoundError(f"S08 sweep root does not exist: {root}")

    layout = _scope_layout(scope, root, training_percentages)
    final_chunk_index = len(layout.chunks) - 1
    final_counts = layout.chunks[final_chunk_index]
    tasks: list[ChunkTask] = []

    sources: list[tuple[int, float | None]] = []
    if include_baseline:
        sources.append((0, None))
    sources.extend(
        (percentage_index, percentage)
        for percentage_index, percentage in enumerate(layout.percentages, start=1)
    )

    for source_index, percentage in sources:
        tag = _percentage_tag(percentage)
        model_root = layout.root_prefix / tag / layout.model_alias
        complete, reason = _chunk_completion(
            model_root,
            layout.model_alias,
            final_chunk_index,
            final_counts,
        )
        tasks.append(
            ChunkTask(
                task_id=source_index * len(layout.chunks) + final_chunk_index,
                condition=(
                    "inference_only"
                    if percentage is None
                    else "fault_aware_training"
                ),
                tag=tag,
                training_percentage=percentage,
                model_root=model_root,
                chunk_index=final_chunk_index,
                flip_counts=final_counts,
                complete=complete,
                reason=reason,
            )
        )

    return layout, tasks


def _write_report(
    path: Path,
    scope: str,
    root: Path,
    max_evaluation_percentage: float,
    layout: ScopeLayout,
    tasks: list[ChunkTask],
) -> None:
    missing = [task for task in tasks if not task.complete]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scope": scope,
                "root": str(root),
                "maximum_evaluation_percentage": max_evaluation_percentage,
                "model_alias": layout.model_alias,
                "training_percentages": list(layout.percentages),
                "final_chunk_index": len(layout.chunks) - 1,
                "final_chunk_flip_counts": list(layout.chunks[-1]),
                "missing_task_ids": [task.task_id for task in missing],
                "complete_task_ids": [
                    task.task_id for task in tasks if task.complete
                ],
                "tasks": [task.as_report_dict() for task in tasks],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find S08 final evaluation chunks that must be rerun when extending "
            "the inference x-axis to 7% or 10%."
        )
    )
    parser.add_argument("--scope", choices=("whole", "input"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-percentages", required=True)
    parser.add_argument("--max-evaluation-percentage", type=float, required=True)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument(
        "--print-task-ids",
        action="store_true",
        help="Print only the comma-separated missing Slurm array task IDs.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Do not audit or submit the inference_only source. Use this when "
            "another searched root already provides the baseline at the target "
            "x-axis percentage."
        ),
    )
    args = parser.parse_args()

    try:
        _set_max_evaluation_percentage(args.max_evaluation_percentage)
        layout, tasks = audit_final_chunk_tasks(
            args.scope,
            args.root,
            args.training_percentages,
            include_baseline=not args.skip_baseline,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.write_report is not None:
        _write_report(
            args.write_report,
            args.scope,
            args.root,
            args.max_evaluation_percentage,
            layout,
            tasks,
        )

    missing = [task for task in tasks if not task.complete]
    if args.print_task_ids:
        print(",".join(str(task.task_id) for task in missing))
        return

    print(
        f"{args.scope} scope final chunk {len(layout.chunks) - 1}: "
        f"{' '.join(map(str, layout.chunks[-1]))}"
    )
    print(f"Missing tasks: {','.join(str(task.task_id) for task in missing) or 'none'}")
    for task in tasks:
        state = "ok" if task.complete else "missing"
        percentage = "baseline" if task.training_percentage is None else (
            f"{task.training_percentage:g}%"
        )
        print(f"{state}\ttask={task.task_id}\t{percentage}\t{task.reason}")


if __name__ == "__main__":
    main()
