#!/usr/bin/env python3
"""Validate, summarize, and stage all RTL handoff campaign artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from models.rtl_handoff import RTL_HANDOFF_CONFIGS
from rtlHandoffStudy.portable import estimated_group_sum_luts


SUMMARY_FIELDS = (
    "task_index",
    "alias",
    "family",
    "base_architecture",
    "thresholds_per_feature",
    "binarization_training",
    "group_width_per_class",
    "tau",
    "compact_code_bits_per_feature",
    "compact_input_bits",
    "eflx_input_margin",
    "thermometer_input_bits",
    "estimated_group_sum_luts_xilinx_style",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "categorical_accuracy",
    "cross_entropy",
    "checkpoint",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.result_root.resolve()
    rows = []
    required = (
        "best_model.importable.pth",
        "discrete_netlist.pth",
        "thresholds.pth",
        "model.json",
        "test_metrics.json",
        "training_complete.json",
    )
    for task_index, config in enumerate(RTL_HANDOFF_CONFIGS):
        alias = str(config["alias"])
        directory = root / alias
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{alias} is incomplete; missing {missing}.")
        model = json.loads((directory / "model.json").read_text())
        test = json.loads((directory / "test_metrics.json").read_text())
        training = json.loads((directory / "training_complete.json").read_text())
        kwargs = model["model_kwargs"]
        if not training.get("complete"):
            raise RuntimeError(f"{alias} training marker is not complete.")
        rows.append(
            {
                "task_index": task_index,
                "alias": alias,
                "family": kwargs["family"],
                "base_architecture": kwargs["base_architecture"],
                "thresholds_per_feature": kwargs["thresholds_per_feature"],
                "binarization_training": kwargs["binarization_training"],
                "group_width_per_class": kwargs["group_width_per_class"],
                "tau": kwargs["tau"],
                "compact_code_bits_per_feature": kwargs["compact_code_bits_per_feature"],
                "compact_input_bits": kwargs["compact_input_bits"],
                "eflx_input_margin": 632 - int(kwargs["compact_input_bits"]),
                "thermometer_input_bits": kwargs["thermometer_input_bits"],
                "estimated_group_sum_luts_xilinx_style": estimated_group_sum_luts(
                    int(kwargs["group_width_per_class"])
                ),
                "validation_balanced_accuracy": training[
                    "best_validation_balanced_accuracy"
                ],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "categorical_accuracy": test["categorical_accuracy"],
                "cross_entropy": test["cross_entropy"],
                "checkpoint": str((directory / "best_model.importable.pth").resolve()),
            }
        )

    summary = root / "balanced_accuracy_summary.tsv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    stage = root / "transfer_ready"
    if stage.exists() and not args.overwrite:
        raise FileExistsError(f"Transfer directory already exists: {stage}; pass --overwrite.")
    temporary = Path(tempfile.mkdtemp(prefix=".transfer_ready.", dir=root))
    try:
        for row in rows:
            source = root / row["alias"]
            model_set = "model2" if row["family"] == "dense" else "model3"
            destination = temporary / model_set / "LGN" / row["alias"]
            destination.mkdir(parents=True, exist_ok=True)
            for name in required:
                _link_or_copy(source / name, destination / name)
            checksummed = [destination / name for name in required]
            (destination / "SHA256SUMS").write_text(
                "".join(f"{_sha256(path)}  {path.name}\n" for path in checksummed)
            )
        _link_or_copy(summary, temporary / summary.name)
        if stage.exists():
            shutil.rmtree(stage)
        temporary.replace(stage)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    manifest = {
        "format": "smartpixels_lgn_rtl_handoff_campaign_v1",
        "models": len(rows),
        "dense_models": sum(row["family"] == "dense" for row in rows),
        "conv_models": sum(row["family"] == "conv" for row in rows),
        "summary": str(summary),
        "transfer_ready": str(stage),
        "fit_warning": (
            "No EFLX4K fit is claimed. Current resource diagnostics are Xilinx-oriented; "
            "run target-specific synthesis including the compact decoder."
        ),
    }
    (root / "campaign_complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[DONE] Summary: {summary}")
    print(f"[DONE] Transfer-ready artifacts: {stage}")


if __name__ == "__main__":
    main()
