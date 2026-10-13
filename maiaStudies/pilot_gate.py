"""Refuse a full MAIA campaign unless corrected-input pilot models learn."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .campaign import get_config
from .data import INPUT_SCHEMA


PILOT_TASKS = (7, 22)  # middle-size tau=20 Model 2 and Model 3


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def check_pilot(
    result_root: Path,
    minimum_balanced_accuracy: float = 0.55,
    minimum_roc_auc: float = 0.60,
) -> dict:
    rows = []
    failures = []
    for task_index in PILOT_TASKS:
        config = get_config(task_index)
        path = result_root / config.alias / "evaluation_complete.json"
        if not path.is_file():
            failures.append(f"{config.alias}: missing {path}")
            continue
        with path.open() as handle:
            result = json.load(handle)
        balanced_accuracy = float(result["full_test"]["argmax"]["balanced_accuracy"])
        roc_auc = float(result["full_test"]["roc_auc"])
        row = {
            "task_index": task_index,
            "alias": config.alias,
            "family": config.family,
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
            "input_schema": result.get("input_schema"),
        }
        rows.append(row)
        if result.get("input_schema") != INPUT_SCHEMA:
            failures.append(
                f"{config.alias}: schema {result.get('input_schema')!r} != {INPUT_SCHEMA!r}"
            )
        if balanced_accuracy < minimum_balanced_accuracy:
            failures.append(
                f"{config.alias}: balanced accuracy {balanced_accuracy:.4f} < "
                f"{minimum_balanced_accuracy:.4f}"
            )
        if roc_auc < minimum_roc_auc:
            failures.append(
                f"{config.alias}: ROC AUC {roc_auc:.4f} < {minimum_roc_auc:.4f}"
            )
    payload = {
        "passed": not failures and len(rows) == len(PILOT_TASKS),
        "input_schema": INPUT_SCHEMA,
        "minimum_balanced_accuracy": minimum_balanced_accuracy,
        "minimum_roc_auc": minimum_roc_auc,
        "models": rows,
        "failures": failures,
    }
    output = result_root / (
        "pilot_gate_complete.json" if payload["passed"] else "pilot_gate_failed.json"
    )
    _atomic_json(output, payload)
    if not payload["passed"]:
        raise RuntimeError("MAIA pilot gate failed:\n- " + "\n- ".join(failures))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.55)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.60)
    args = parser.parse_args()
    result = check_pilot(
        args.result_root,
        minimum_balanced_accuracy=args.minimum_balanced_accuracy,
        minimum_roc_auc=args.minimum_roc_auc,
    )
    for row in result["models"]:
        print(
            f"[PASS] {row['alias']}: balanced_accuracy="
            f"{row['balanced_accuracy']:.5f}, roc_auc={row['roc_auc']:.5f}"
        )
    print(f"[DONE] Pilot gate: {args.result_root / 'pilot_gate_complete.json'}")


if __name__ == "__main__":
    main()
