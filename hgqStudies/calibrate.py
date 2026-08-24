"""Calibrate HGQ WRAP ranges and reevaluate an existing campaign checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from dataloader import create_dataloaders
from hgqStudies.campaign import config_for_alias, config_for_task
from hgqStudies.common import sha256_file, to_numpy
from hgqStudies.models import load_model, model_ebops, quantizer_statistics
from hgqStudies.train import configure_backend, evaluate, write_json


def calibrate_ranges(model, loader, maximum_batches: int = 0) -> dict:
    """Trace integer ranges over sequential training batches using HGQ's public API."""
    from hgq.utils import trace_minmax

    batch_count = 0
    sample_count = 0
    for inputs, _labels in loader:
        if maximum_batches and batch_count >= maximum_batches:
            break
        values = to_numpy(inputs).astype(np.float32, copy=False)
        trace_minmax(
            model,
            values,
            reset=(batch_count == 0),
            batch_size=len(values),
            verbose=0,
        )
        batch_count += 1
        sample_count += len(values)
    if batch_count == 0:
        raise RuntimeError("Calibration loader produced no batches.")
    return {
        "calibration_batches": batch_count,
        "calibration_samples": sample_count,
        "calibrated_ebops": model_ebops(model),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", type=int)
    selection.add_argument("--alias")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-config", default="config/10783560/v3.yml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="Maximum sequential training batches; zero uses the complete training split.",
    )
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_batches < 0:
        raise ValueError("--calibration-batches cannot be negative.")
    config = config_for_task(args.task) if args.task is not None else config_for_alias(args.alias)
    model_dir = args.result_root.resolve() / config.alias
    output_dir = model_dir / "calibration"
    completion_path = output_dir / "calibration_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text())
        if not completion.get("passes_acceptance"):
            raise RuntimeError(
                f"{config.alias} was calibrated but did not pass acceptance: "
                f"balanced_accuracy={float(completion['balanced_accuracy']):.6f}"
            )
        print(
            f"[SKIP] {config.alias} already calibrated: "
            f"balanced_accuracy={float(completion['balanced_accuracy']):.6f}",
            flush=True,
        )
        return
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse incomplete calibration directory: {output_dir}"
        )

    _device_scope = configure_backend(args.device)
    train_loader, val_loader, test_loader = create_dataloaders(
        config_path=args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        input_type=config.input_type,
        target_type="classification",
        balance_classes=False,
        val_size=0.01,
        label_format="one-hot",
    )

    candidate_paths = (
        model_dir / "best_model.keras",
        model_dir / "final_model.keras",
    )
    missing = [str(path) for path in candidate_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source checkpoint(s): {', '.join(missing)}")

    candidates = []
    calibrated_models = []
    for source_path in candidate_paths:
        print(f"[CALIBRATE] {config.alias} source={source_path.name}", flush=True)
        model = load_model(source_path)
        calibration = calibrate_ranges(
            model,
            train_loader,
            maximum_batches=args.calibration_batches,
        )
        validation = evaluate(model, val_loader)
        record = {
            "source_checkpoint": source_path.name,
            **calibration,
            "validation": validation,
            "quantization": quantizer_statistics(model),
        }
        candidates.append(record)
        calibrated_models.append(model)
        print(
            f"[CANDIDATE] source={source_path.name} "
            f"val_cross_entropy={validation['cross_entropy']:.6f} "
            f"val_balanced_accuracy={validation['balanced_accuracy']:.6f} "
            f"EBOPs={calibration['calibrated_ebops']:.1f}",
            flush=True,
        )

    selected_index = min(
        range(len(candidates)),
        key=lambda index: candidates[index]["validation"]["cross_entropy"],
    )
    selected_record = candidates[selected_index]
    selected_model = calibrated_models[selected_index]
    output_dir.mkdir(parents=True)
    calibrated_path = output_dir / "best_model.calibrated.keras"
    selected_model.save(calibrated_path, overwrite=True)

    test_metrics = evaluate(selected_model, test_loader)
    passes_acceptance = (
        test_metrics["balanced_accuracy"] >= args.minimum_balanced_accuracy
    )
    result = {
        "schema_version": 1,
        "alias": config.alias,
        "task_index": config.index,
        "input_type": config.input_type,
        "hidden_units": list(config.hidden_units),
        "beta0": config.beta0,
        "selected_source_checkpoint": selected_record["source_checkpoint"],
        "selection_metric": "minimum calibrated validation cross-entropy",
        "selected_validation_metrics": selected_record["validation"],
        "test_metrics": test_metrics,
        "calibrated_ebops": selected_record["calibrated_ebops"],
        "minimum_balanced_accuracy": args.minimum_balanced_accuracy,
        "passes_acceptance": passes_acceptance,
        "calibration_batches": selected_record["calibration_batches"],
        "calibration_samples": selected_record["calibration_samples"],
        "checkpoint": str(calibrated_path.resolve()),
        "checkpoint_sha256": sha256_file(calibrated_path),
    }
    write_json(output_dir / "calibration_candidates.json", {"candidates": candidates})
    write_json(output_dir / "calibrated_test_metrics.json", result)
    write_json(
        completion_path,
        {
            "complete": True,
            "alias": config.alias,
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "passes_acceptance": passes_acceptance,
            "checkpoint": calibrated_path.name,
            "checkpoint_sha256": result["checkpoint_sha256"],
        },
    )
    print(
        f"[DONE] {config.alias} calibrated_test_balanced_accuracy="
        f"{test_metrics['balanced_accuracy']:.6f} passes_acceptance={passes_acceptance} "
        f"checkpoint={calibrated_path}",
        flush=True,
    )
    if not passes_acceptance:
        raise RuntimeError(
            f"{config.alias} calibrated balanced accuracy "
            f"{test_metrics['balanced_accuracy']:.6f} is below "
            f"{args.minimum_balanced_accuracy:.6f}."
        )


if __name__ == "__main__":
    main()
