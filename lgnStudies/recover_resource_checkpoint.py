#!/usr/bin/env python3
"""Recover a registered final LGN from the latest complete synthesis export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lgnStudies.resource_precision_study import MODEL_ALIASES, save_model
from models.registry import REGISTRY


def rebuild_registered_dense_lgn(
    portable: torch.nn.Sequential,
    model_class: type[torch.nn.Module],
) -> torch.nn.Module:
    if not isinstance(portable, torch.nn.Sequential) or len(portable) < 3:
        raise ValueError("Expected a sequential binarizer/layers/GroupSum export.")
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.bin = portable[0]
    model.layers = torch.nn.ModuleList(list(portable.children())[1:-1])
    model.group_sum = portable[-1]
    return model.eval()


def latest_complete_export(
    output: Path,
    model_alias: str,
) -> tuple[int, Path, dict]:
    export_root = output / "synthesis_export" / "model2" / "LGN"
    pattern = re.compile(rf"^{re.escape(model_alias)}_step(\d+)$")
    candidates = []
    for directory in export_root.iterdir() if export_root.is_dir() else ():
        match = pattern.match(directory.name)
        checkpoint = directory / "best_model.importable.pth"
        metadata = directory / "model.json"
        if match and checkpoint.is_file() and metadata.is_file():
            candidates.append((int(match.group(1)), checkpoint, metadata))
    if not candidates:
        raise ValueError(f"No complete synthesis exports found under {export_root}.")
    step, checkpoint, metadata_path = max(candidates)
    return step, checkpoint, json.loads(metadata_path.read_text())


def resource_row_for_step(output: Path, step: int) -> dict:
    path = output / "resource_precision.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["step"]) == step]
    if len(matches) != 1:
        raise ValueError(f"Expected one resource row for step {step} in {path}.")
    return matches[0]


def recover_checkpoint(
    output: Path,
    model_alias: str,
    minimum_steps: int = 20_000,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    registry_name = MODEL_ALIASES[model_alias]
    final_path = output / "final_model.pth"
    status_path = output / "training_complete.json"
    if not overwrite and (final_path.exists() or status_path.exists()):
        raise ValueError(
            "Final checkpoint or completion metadata already exists; "
            "use --overwrite only after inspecting it."
        )

    step, portable_path, export_metadata = latest_complete_export(
        output, model_alias
    )
    if step < minimum_steps:
        raise ValueError(
            f"Latest complete stage is only step {step}; require {minimum_steps}."
        )
    training_metadata = export_metadata.get("training_metadata", {})
    if training_metadata.get("model_alias") != model_alias:
        raise ValueError("Export model alias does not match the requested model.")
    if training_metadata.get("registry_name") != registry_name:
        raise ValueError("Export registry name does not match the requested model.")
    if int(training_metadata.get("training_step", -1)) != step:
        raise ValueError("Export training step does not match its directory name.")

    row = resource_row_for_step(output, step)
    balanced_accuracy = float(row["balanced_accuracy"])
    if not np.isclose(
        balanced_accuracy,
        float(training_metadata["balanced_accuracy"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Export and resource CSV balanced accuracies differ.")

    portable = torch.load(
        portable_path, map_location="cpu", weights_only=False
    ).eval()
    model = rebuild_registered_dense_lgn(
        portable, REGISTRY[registry_name]["class"]
    )
    sample = torch.linspace(-1.0, 1.0, 14).reshape(1, 14)
    with torch.no_grad():
        portable_output = portable(sample)
        recovered_output = model(sample)
    if not torch.equal(portable_output, recovered_output):
        raise ValueError("Recovered registered model does not match its export.")

    save_model(model, final_path)
    status = {
        "complete": True,
        "completion_basis": "recovered_latest_recorded_stage",
        "model_alias": model_alias,
        "registry_name": registry_name,
        "minimum_steps": minimum_steps,
        "completed_steps": step,
        "minimum_training_hours": None,
        "elapsed_training_hours": None,
        "training_loop_wall_hours": None,
        "final_recorded_step": step,
        "final_balanced_accuracy": balanced_accuracy,
        "stop_requested": False,
        "recovered_from": str(portable_path),
        "recovery_note": (
            "The original Slurm job reached this fully evaluated/exported "
            "stage but ended before its final checkpoint block."
        ),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return final_path, status_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--minimum-steps", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        final_path, status_path = recover_checkpoint(
            args.output,
            args.model,
            minimum_steps=args.minimum_steps,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error)) from error
    print(f"[RECOVERED] {final_path}")
    print(f"[RECOVERED] {status_path}")


if __name__ == "__main__":
    main()
