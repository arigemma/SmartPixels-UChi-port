"""Strictly summarize calibrated HGQ2 checkpoints that pass acceptance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hgqStudies.campaign import MODEL_CONFIGS
from hgqStudies.common import sha256_file


FIELDS = (
    "task_index",
    "alias",
    "beta0",
    "balanced_accuracy",
    "categorical_accuracy",
    "cross_entropy",
    "calibrated_ebops",
    "selected_source_checkpoint",
    "validation_balanced_accuracy",
    "validation_cross_entropy",
    "checkpoint",
)


def collect(result_root: Path) -> list[dict]:
    rows = []
    problems = []
    for config in MODEL_CONFIGS:
        calibration_dir = result_root / config.alias / "calibration"
        completion_path = calibration_dir / "calibration_complete.json"
        metrics_path = calibration_dir / "calibrated_test_metrics.json"
        checkpoint_path = calibration_dir / "best_model.calibrated.keras"
        missing = [
            str(path)
            for path in (completion_path, metrics_path, checkpoint_path)
            if not path.is_file()
        ]
        if missing:
            problems.append(f"{config.alias}: missing {', '.join(missing)}")
            continue
        completion = json.loads(completion_path.read_text())
        result = json.loads(metrics_path.read_text())
        if not completion.get("complete") or not result.get("passes_acceptance"):
            problems.append(f"{config.alias}: calibration did not pass acceptance")
            continue
        if completion.get("alias") != config.alias or result.get("alias") != config.alias:
            problems.append(f"{config.alias}: metadata alias mismatch")
            continue
        if result.get("task_index") != config.index:
            problems.append(f"{config.alias}: metadata task-index mismatch")
            continue
        digest = sha256_file(checkpoint_path)
        if (
            digest != completion.get("checkpoint_sha256")
            or digest != result.get("checkpoint_sha256")
        ):
            problems.append(f"{config.alias}: calibrated checkpoint SHA256 mismatch")
            continue
        test = result["test_metrics"]
        validation = result["selected_validation_metrics"]
        rows.append(
            {
                "task_index": config.index,
                "alias": config.alias,
                "beta0": config.beta0,
                "balanced_accuracy": test["balanced_accuracy"],
                "categorical_accuracy": test["categorical_accuracy"],
                "cross_entropy": test["cross_entropy"],
                "calibrated_ebops": result["calibrated_ebops"],
                "selected_source_checkpoint": result["selected_source_checkpoint"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_cross_entropy": validation["cross_entropy"],
                "checkpoint": str(checkpoint_path.resolve()),
            }
        )
    if problems:
        raise RuntimeError(
            f"Calibrated campaign has {len(problems)} problem(s):\n  "
            + "\n  ".join(problems)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    rows = collect(result_root)
    output_dir = result_root / "calibrated_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "balanced_accuracy_summary.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "balanced_accuracy_summary.json").write_text(
        json.dumps({"schema_version": 1, "results": rows}, indent=2, sort_keys=True)
        + "\n"
    )
    print("alias\tbalanced_accuracy\tcalibrated_ebops\tcheckpoint")
    for row in rows:
        print(
            f"{row['alias']}\t{float(row['balanced_accuracy']):.6f}\t"
            f"{float(row['calibrated_ebops']):.1f}\t{row['checkpoint']}"
        )
    print(f"[DONE] {table_path}")


if __name__ == "__main__":
    main()
