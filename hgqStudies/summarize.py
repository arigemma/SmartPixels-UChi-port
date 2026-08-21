"""Strictly validate and summarize all 21 HGQ2 Model-2 campaign outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hgqStudies.campaign import MODEL_CONFIGS, manifest
from hgqStudies.common import sha256_file


FIELDS = (
    "task_index",
    "alias",
    "qkeras_peer",
    "input_type",
    "hidden_units",
    "beta0",
    "balanced_accuracy",
    "categorical_accuracy",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "hgq_ebops",
    "keras_parameter_count",
    "reference_lut",
    "reference_ff",
    "reference_balanced_accuracy",
    "reference_resources_lut_plus_ff",
    "checkpoint",
)


def collect(result_root: Path) -> list[dict]:
    rows = []
    errors = []
    for config in MODEL_CONFIGS:
        model_dir = result_root / config.alias
        completion_path = model_dir / "training_complete.json"
        metrics_path = model_dir / "test_metrics.json"
        checkpoint_path = model_dir / "best_model.keras"
        missing = [str(path) for path in (completion_path, metrics_path, checkpoint_path) if not path.is_file()]
        if missing:
            errors.append(f"{config.alias}: missing {', '.join(missing)}")
            continue
        completion = json.loads(completion_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        if not completion.get("complete"):
            errors.append(f"{config.alias}: completion marker is not complete")
            continue
        if completion.get("alias") != config.alias or metrics.get("alias") != config.alias:
            errors.append(f"{config.alias}: metadata alias mismatch")
            continue
        digest = sha256_file(checkpoint_path)
        if digest != completion.get("best_model_sha256"):
            errors.append(f"{config.alias}: checkpoint SHA256 mismatch")
            continue
        recalls = metrics["recalls"]
        rows.append(
            {
                "task_index": config.index,
                "alias": config.alias,
                "qkeras_peer": config.qkeras_peer,
                "input_type": config.input_type,
                "hidden_units": "x".join(str(width) for width in config.hidden_units),
                "beta0": config.beta0,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "categorical_accuracy": metrics["categorical_accuracy"],
                "recall_class_0": recalls[0],
                "recall_class_1": recalls[1],
                "recall_class_2": recalls[2],
                "hgq_ebops": metrics["hgq_ebops"],
                "keras_parameter_count": metrics["keras_parameter_count"],
                "reference_lut": config.reference_lut,
                "reference_ff": config.reference_ff,
                "reference_balanced_accuracy": config.reference_balanced_accuracy,
                "reference_resources_lut_plus_ff": config.reference_resources,
                "checkpoint": str(checkpoint_path.resolve()),
            }
        )
    if errors:
        raise RuntimeError(
            f"HGQ2 campaign is incomplete or invalid ({len(errors)} problem(s)):\n  "
            + "\n  ".join(errors)
        )
    if len(rows) != len(MODEL_CONFIGS):
        raise RuntimeError(f"Expected {len(MODEL_CONFIGS)} rows, collected {len(rows)}.")
    return rows


def write_summary(result_root: Path, output_dir: Path) -> list[dict]:
    rows = collect(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "balanced_accuracy_summary.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    payload = manifest()
    payload["results"] = rows
    (output_dir / "balanced_accuracy_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print("\t".join(("alias", "balanced_accuracy", "hgq_ebops", "qkeras_peer", "checkpoint")))
    for row in rows:
        print(
            f"{row['alias']}\t{float(row['balanced_accuracy']):.6f}\t"
            f"{float(row['hgq_ebops']):.1f}\t{row['qkeras_peer']}\t{row['checkpoint']}"
        )
    print(f"[DONE] {table_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.result_root / "summary")
    write_summary(args.result_root.resolve(), output_dir.resolve())


if __name__ == "__main__":
    main()
