#!/usr/bin/env python3
"""Select and stage the best discrete-evaluation tau for each model size."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


TAU_DIRECTORY_PATTERN = re.compile(r"tau_(\d+)$")


@dataclass(frozen=True)
class EvaluationRow:
    size: str
    tau: int
    balanced_accuracy: float
    label: str
    input_type: str
    source_checkpoint: str
    resolved_checkpoint: Path


def _checkpoint_identity(checkpoint: Path) -> tuple[str, int]:
    match = TAU_DIRECTORY_PATTERN.fullmatch(checkpoint.parent.name)
    if match is None or not checkpoint.parent.parent.name:
        raise ValueError(
            "Checkpoint paths must end in <size>/tau_<integer>/best_model.pth; "
            f"got {checkpoint}"
        )
    return checkpoint.parent.parent.name, int(match.group(1))


def read_evaluation_rows(summary: Path, repository_root: Path) -> list[EvaluationRow]:
    if not summary.is_file():
        raise ValueError(f"Missing balanced-accuracy summary: {summary}")

    with summary.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"label", "input_type", "balanced_acc", "path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{summary} is missing required columns: {sorted(missing)}"
            )

        rows: list[EvaluationRow] = []
        seen: set[tuple[str, int]] = set()
        for line_number, raw in enumerate(reader, start=2):
            source_checkpoint = raw["path"].strip()
            checkpoint = Path(source_checkpoint)
            size, tau = _checkpoint_identity(checkpoint)
            identity = (size, tau)
            if identity in seen:
                raise ValueError(
                    f"Duplicate evaluation for {size} tau={tau} at line {line_number}"
                )
            seen.add(identity)

            try:
                balanced_accuracy = float(raw["balanced_acc"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid balanced_acc at line {line_number}: "
                    f"{raw['balanced_acc']!r}"
                ) from error
            if not math.isfinite(balanced_accuracy) or not 0.0 <= balanced_accuracy <= 1.0:
                raise ValueError(
                    f"balanced_acc must be finite and within [0, 1] at line "
                    f"{line_number}; got {balanced_accuracy}"
                )

            resolved = checkpoint if checkpoint.is_absolute() else repository_root / checkpoint
            resolved = resolved.resolve()
            if not resolved.is_file():
                raise ValueError(
                    f"Evaluated checkpoint is no longer present: {resolved}"
                )

            rows.append(
                EvaluationRow(
                    size=size,
                    tau=tau,
                    balanced_accuracy=balanced_accuracy,
                    label=raw["label"].strip(),
                    input_type=raw["input_type"].strip(),
                    source_checkpoint=source_checkpoint,
                    resolved_checkpoint=resolved,
                )
            )
    return rows


def select_best_rows(
    rows: Iterable[EvaluationRow],
    expected_sizes: Sequence[str],
    expected_taus: Sequence[int],
) -> list[EvaluationRow]:
    sizes = list(expected_sizes)
    taus = list(expected_taus)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Expected sizes must be a non-empty list without duplicates.")
    if not taus or len(set(taus)) != len(taus):
        raise ValueError("Expected taus must be a non-empty list without duplicates.")

    by_identity = {(row.size, row.tau): row for row in rows}
    expected = {(size, tau) for size in sizes for tau in taus}
    observed = set(by_identity)
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unexpected:
            parts.append(f"unexpected={unexpected}")
        raise ValueError("Evaluation matrix is not the expected size/tau grid: " + "; ".join(parts))

    selected = []
    for size in sizes:
        candidates = [by_identity[(size, tau)] for tau in taus]
        # Tau is a hyperparameter. Select solely by balanced accuracy; use the
        # smaller tau only as a deterministic tie-break for exactly equal values.
        selected.append(
            max(candidates, key=lambda row: (row.balanced_accuracy, -row.tau))
        )
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_best_table(path: Path, selected: Sequence[EvaluationRow]) -> None:
    fieldnames = [
        "size",
        "tau",
        "balanced_accuracy",
        "input_type",
        "label",
        "source_checkpoint",
        "staged_checkpoint",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in selected:
            flat_name = f"{row.size}_tau{row.tau}"
            writer.writerow(
                {
                    "size": row.size,
                    "tau": row.tau,
                    "balanced_accuracy": f"{row.balanced_accuracy:.12g}",
                    "input_type": row.input_type,
                    "label": row.label,
                    "source_checkpoint": row.source_checkpoint,
                    "staged_checkpoint": f"{flat_name}/best_model.pth",
                }
            )


def stage_selected_rows(
    selected: Sequence[EvaluationRow],
    summary: Path,
    stage_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    stage_dir = stage_dir.resolve()
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    if stage_dir.exists() and not overwrite:
        raise ValueError(
            f"Transfer directory already exists: {stage_dir}; pass --overwrite-stage"
        )
    if stage_dir.exists() and not stage_dir.is_dir():
        raise ValueError(f"Transfer path exists and is not a directory: {stage_dir}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{stage_dir.name}.", dir=stage_dir.parent)
    )
    manifest_rows = []
    try:
        for row in selected:
            flat_name = f"{row.size}_tau{row.tau}"
            model_dir = temporary / flat_name
            model_dir.mkdir()
            staged_checkpoint = model_dir / "best_model.pth"
            # Use an independent copy rather than a symlink/hardlink so no action
            # in the transfer directory can mutate the original trained model.
            shutil.copy2(row.resolved_checkpoint, staged_checkpoint)
            checksum = _sha256(staged_checkpoint)
            (model_dir / "SHA256SUMS").write_text(
                f"{checksum}  {staged_checkpoint.name}\n"
            )
            selection = {
                "size": row.size,
                "tau": row.tau,
                "balanced_accuracy": row.balanced_accuracy,
                "label": row.label,
                "input_type": row.input_type,
                "source_checkpoint": row.source_checkpoint,
                "staged_checkpoint": f"{flat_name}/best_model.pth",
                "staging_method": "copy",
                "sha256": checksum,
            }
            (model_dir / "selection.json").write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n"
            )
            manifest_rows.append(selection)

        _write_best_table(temporary / "best_by_size.tsv", selected)
        shutil.copy2(summary, temporary / "balanced_accuracy_summary.tsv")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_metric": "maximum exact discrete balanced accuracy per size",
            "selected_count": len(selected),
            "models": manifest_rows,
        }
        (temporary / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        temporary.replace(stage_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return stage_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete size/tau evaluation grid, select the highest "
            "balanced-accuracy tau per size, and stage transfer-ready checkpoints."
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--best-table", type=Path, required=True)
    parser.add_argument("--expected-size", action="append", required=True)
    parser.add_argument("--expected-tau", action="append", type=int, required=True)
    parser.add_argument("--overwrite-stage", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = read_evaluation_rows(args.summary, args.repository_root.resolve())
        selected = select_best_rows(rows, args.expected_size, args.expected_tau)
        stage_dir = stage_selected_rows(
            selected,
            args.summary,
            args.stage_dir,
            overwrite=args.overwrite_stage,
        )
        args.best_table.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_dir / "best_by_size.tsv", args.best_table)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print("Selected best tau per model size:")
    for row in selected:
        print(
            f"  {row.size}: tau={row.tau} "
            f"balanced_accuracy={row.balanced_accuracy:.6f} "
            f"checkpoint={row.source_checkpoint}"
        )
    print(f"[DONE] Best-tau table: {args.best_table}")
    print(f"[DONE] Transfer-ready checkpoints: {stage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
