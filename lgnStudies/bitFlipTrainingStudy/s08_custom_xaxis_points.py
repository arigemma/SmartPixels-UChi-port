#!/usr/bin/env python3
"""Submit/audit arbitrary extra S08 inference x-axis points."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

from lgnStudies.bitFlipStudy.campaign import REPEATS
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MODEL_ALIAS,
    REGISTRY_NAME,
    S08_TOTAL_FAULT_SITES,
    parse_training_percentages,
    percentage_tag,
)
from lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep import (
    INPUT_SCOPE_DIR,
    INPUT_SCOPE_NAME,
    S08_INPUT_FAULT_SITES,
    WHOLE_SCOPE_NAME,
)


COUNTS_PER_CUSTOM_CHUNK = 3
DEFAULT_CHUNK_OFFSET = 7


@dataclass(frozen=True)
class CustomTask:
    task_id: int
    condition: str
    fault_scope: str
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
            "fault_scope": self.fault_scope,
            "tag": self.tag,
            "training_percentage": self.training_percentage,
            "model_root": str(self.model_root),
            "chunk_index": self.chunk_index,
            "flip_counts": list(self.flip_counts),
            "complete": self.complete,
            "reason": self.reason,
        }


def _parse_percentages(text: str) -> tuple[float, ...]:
    values = []
    for raw in text.replace(",", " ").split():
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"Percentage must be numeric: {raw!r}") from error
        if not math.isfinite(value) or not 0.0 < value <= 100.0:
            raise ValueError(f"Percentages must lie in (0, 100], got {raw!r}.")
        values.append(value)
    if not values:
        raise ValueError("At least one x-axis percentage is required.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate x-axis percentages are not allowed.")
    return tuple(sorted(values))


def _scope_properties(scope: str, root: Path) -> tuple[str, Path, int]:
    if scope == "whole":
        return "all", root, S08_TOTAL_FAULT_SITES
    if scope == "input":
        return "input", root / INPUT_SCOPE_DIR, S08_INPUT_FAULT_SITES
    raise ValueError(f"Unknown S08 fault scope {scope!r}; use 'whole' or 'input'.")


def _custom_count_chunks(
    scope: str,
    x_percentages: str,
) -> tuple[tuple[int, ...], ...]:
    _fault_scope, _root_prefix, total_sites = _scope_properties(scope, Path("."))
    percentages = _parse_percentages(x_percentages)
    counts = tuple(int(round(total_sites * percentage / 100.0)) for percentage in percentages)
    if any(count <= 0 for count in counts):
        raise ValueError(f"X-axis percentages map to non-positive counts: {counts}.")
    if len(set(counts)) != len(counts):
        raise ValueError(
            "Some x-axis percentages map to the same integer flip count: "
            f"{dict(zip(percentages, counts))}"
        )
    return tuple(
        tuple(counts[index : index + COUNTS_PER_CUSTOM_CHUNK])
        for index in range(0, len(counts), COUNTS_PER_CUSTOM_CHUNK)
    )


def _source_entries(
    training_percentages: tuple[float, ...],
    include_baseline: bool,
) -> list[tuple[int, float | None]]:
    sources: list[tuple[int, float | None]] = []
    if include_baseline:
        sources.append((0, None))
    sources.extend(
        (percentage_index, percentage)
        for percentage_index, percentage in enumerate(training_percentages, start=1)
    )
    return sources


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Result CSV is empty: {path}")
    return rows


def _chunk_completion(
    model_root: Path,
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
    if {row.get("model_alias") for row in rows} != {MODEL_ALIAS}:
        return False, f"wrong model alias in {csv_path}"

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


def audit_tasks(
    scope: str,
    root: Path,
    training_percentages: str,
    x_percentages: str,
    chunk_offset: int = DEFAULT_CHUNK_OFFSET,
    include_baseline: bool = True,
) -> tuple[tuple[tuple[int, ...], ...], list[CustomTask]]:
    if not root.is_dir():
        raise FileNotFoundError(f"S08 sweep root does not exist: {root}")
    if chunk_offset < 0:
        raise ValueError("--chunk-offset must be non-negative.")

    fault_scope, root_prefix, _total_sites = _scope_properties(scope, root)
    percentages = parse_training_percentages(training_percentages)
    custom_chunks = _custom_count_chunks(scope, x_percentages)
    tasks: list[CustomTask] = []
    for source_index, percentage in _source_entries(percentages, include_baseline):
        tag = percentage_tag(percentage)
        model_root = root_prefix / tag / MODEL_ALIAS
        condition = "inference_only" if percentage is None else "fault_aware_training"
        for custom_chunk_index, counts in enumerate(custom_chunks):
            chunk_index = chunk_offset + custom_chunk_index
            complete, reason = _chunk_completion(model_root, chunk_index, counts)
            tasks.append(
                CustomTask(
                    task_id=source_index * len(custom_chunks) + custom_chunk_index,
                    condition=condition,
                    fault_scope=fault_scope,
                    tag=tag,
                    training_percentage=percentage,
                    model_root=model_root,
                    chunk_index=chunk_index,
                    flip_counts=counts,
                    complete=complete,
                    reason=reason,
                )
            )
    return custom_chunks, tasks


def _write_report(
    path: Path,
    scope: str,
    root: Path,
    training_percentages: str,
    x_percentages: str,
    custom_chunks: tuple[tuple[int, ...], ...],
    tasks: list[CustomTask],
) -> None:
    missing = [task for task in tasks if not task.complete]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scope": scope,
                "root": str(root),
                "training_percentages": list(
                    parse_training_percentages(training_percentages)
                ),
                "x_axis_percentages": list(_parse_percentages(x_percentages)),
                "custom_flip_count_chunks": [list(chunk) for chunk in custom_chunks],
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


def _task_for_id(tasks: list[CustomTask], task_id: int) -> CustomTask:
    by_id = {task.task_id: task for task in tasks}
    try:
        return by_id[task_id]
    except KeyError as error:
        valid_ids = ",".join(str(task_id) for task_id in sorted(by_id))
        raise ValueError(f"Invalid task ID {task_id}; valid IDs are {valid_ids}.") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("whole", "input"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-percentages", required=True)
    parser.add_argument("--x-percentages", required=True)
    parser.add_argument("--chunk-offset", type=int, default=DEFAULT_CHUNK_OFFSET)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--write-report", type=Path)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--print-task-ids", action="store_true")
    task_group.add_argument("--evaluation-task-id", type=int)
    args = parser.parse_args()

    try:
        custom_chunks, tasks = audit_tasks(
            args.scope,
            args.root,
            args.training_percentages,
            args.x_percentages,
            chunk_offset=args.chunk_offset,
            include_baseline=not args.skip_baseline,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.write_report is not None:
        _write_report(
            args.write_report,
            args.scope,
            args.root,
            args.training_percentages,
            args.x_percentages,
            custom_chunks,
            tasks,
        )

    missing = [task for task in tasks if not task.complete]
    if args.print_task_ids:
        print(",".join(str(task.task_id) for task in missing))
        return
    if args.evaluation_task_id is not None:
        try:
            task = _task_for_id(tasks, args.evaluation_task_id)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(
            "\t".join(
                (
                    task.condition,
                    task.fault_scope,
                    task.tag,
                    "none"
                    if task.training_percentage is None
                    else f"{task.training_percentage:.12g}",
                    MODEL_ALIAS,
                    REGISTRY_NAME,
                    f"{task.chunk_index:02d}",
                    " ".join(map(str, task.flip_counts)),
                )
            )
        )
        return

    print(f"{args.scope} custom chunks: {custom_chunks}")
    print(f"Missing tasks: {','.join(str(task.task_id) for task in missing) or 'none'}")
    for task in tasks:
        state = "ok" if task.complete else "missing"
        percentage = "baseline" if task.training_percentage is None else (
            f"{task.training_percentage:g}%"
        )
        print(
            f"{state}\ttask={task.task_id}\tchunk={task.chunk_index:02d}\t"
            f"{percentage}\t{task.reason}"
        )


if __name__ == "__main__":
    main()
