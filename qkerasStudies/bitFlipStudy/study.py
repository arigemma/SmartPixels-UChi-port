#!/usr/bin/env python3
"""Measure QKeras robustness to exact-count transient activation-bit faults."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataloader import create_dataloaders
from utils import CreateFolder

from qkerasStudies.common import (
    balanced_accuracy_from_confusion,
    configure_device,
    confusion_from_predictions,
    sha256_path,
    to_numpy,
)
from qkerasStudies.fault_injection import (
    FAULT_SCOPES,
    QuantizationRange,
    forward_with_faults,
    sample_unique_fault_indices,
    stage_metadata,
    validate_flip_count,
)
from qkerasStudies.models import get_config, load_model
from qkerasStudies.bitFlipStudy.schema import write_rows


def set_fault_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    tf.keras.utils.set_random_seed(seed)
    return np.random.default_rng(seed)


def validate_clean_checkpoint_metrics(
    training_status: dict,
    model_alias: str,
    registry_name: str,
    clean_metrics: dict,
) -> None:
    if training_status.get("complete") is not True:
        raise RuntimeError("Training status is not marked complete.")
    if training_status.get("model_alias") != model_alias:
        raise RuntimeError("Training status model alias does not match this study.")
    if training_status.get("registry_name") != registry_name:
        raise RuntimeError("Training status registry name does not match this study.")
    expected = float(training_status["final_balanced_accuracy"])
    observed = float(clean_metrics["balanced_accuracy"])
    if not np.isclose(observed, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "K=0 balanced accuracy does not match the final training evaluation: "
            f"training={expected:.12f}, bit_flip={observed:.12f}. "
            "Refusing to analyze a stale or different checkpoint."
        )


def evaluate_complete_test_split(
    model,
    loader,
    config,
    fault_scope: str,
    input_quantization_range: QuantizationRange,
    flip_count: int,
    rng: np.random.Generator,
    reference_predictions: np.ndarray | None = None,
    collect_predictions: bool = False,
) -> tuple[dict, np.ndarray | None]:
    fault_metadata = stage_metadata(config, fault_scope)
    total_fault_sites = int(fault_metadata["total_fault_sites_per_event"])
    validate_flip_count(flip_count, total_fault_sites)

    confusion = np.zeros((3, 3), dtype=np.int64)
    predictions_for_output: list[np.ndarray] = []
    changed = 0
    repairs = np.zeros(3, dtype=np.int64)
    damages = np.zeros(3, dtype=np.int64)
    number_seen = 0

    for inputs, labels in tqdm(
        loader,
        desc="Full test evaluation",
        unit="batch",
        leave=False,
    ):
        x_np = to_numpy(inputs).astype(np.float32, copy=False)
        y_np = to_numpy(labels)
        fault_indices = (
            None
            if flip_count == 0
            else sample_unique_fault_indices(
                x_np.shape[0],
                total_fault_sites,
                flip_count,
                rng,
            )
        )
        logits = forward_with_faults(
            model,
            x_np,
            config,
            training=False,
            fault_scope=fault_scope,
            input_quantization_range=input_quantization_range,
            exact_fault_indices=fault_indices,
        )
        predictions = np.argmax(to_numpy(logits), axis=-1).astype(np.int64)
        truth = np.argmax(y_np, axis=-1).astype(np.int64)
        confusion += confusion_from_predictions(truth, predictions)

        if reference_predictions is not None:
            reference_batch = reference_predictions[
                number_seen : number_seen + len(predictions)
            ]
            if len(reference_batch) != len(predictions):
                raise RuntimeError("Reference predictions do not match test split.")
            changed += int(np.count_nonzero(predictions != reference_batch))
            reference_correct = reference_batch == truth
            fault_correct = predictions == truth
            repairs += np.bincount(
                truth[(~reference_correct) & fault_correct],
                minlength=3,
            )
            damages += np.bincount(
                truth[reference_correct & (~fault_correct)],
                minlength=3,
            )
        if collect_predictions:
            predictions_for_output.append(predictions.astype(np.int8, copy=True))
        number_seen += len(predictions)

    if reference_predictions is not None and number_seen != len(reference_predictions):
        raise RuntimeError("Reference predictions do not match test split length.")

    metrics = balanced_accuracy_from_confusion(confusion)
    totals = np.asarray(metrics["class_totals"], dtype=np.float64)
    repair_rates = repairs.astype(np.float64) / totals
    damage_rates = damages.astype(np.float64) / totals
    metrics.update(
        {
            "prediction_change_fraction": (
                float(changed / number_seen)
                if reference_predictions is not None
                else 0.0
            ),
            "balanced_repair_gain": float(repair_rates.mean()),
            "balanced_damage_loss": float(damage_rates.mean()),
            "repair_fraction": int(repairs.sum()) / number_seen,
            "damage_fraction": int(damages.sum()) / number_seen,
            "repairs": [int(value) for value in repairs],
            "damages": [int(value) for value in damages],
            "number_test_events": number_seen,
        }
    )
    collected = (
        np.concatenate(predictions_for_output)
        if collect_predictions
        else None
    )
    return metrics, collected


def validate_transition_accounting(clean_metrics: dict, fault_metrics: dict) -> None:
    if clean_metrics["class_totals"] != fault_metrics["class_totals"]:
        raise RuntimeError("Clean and fault evaluations have different class totals.")
    observed_delta = (
        fault_metrics["balanced_accuracy"] - clean_metrics["balanced_accuracy"]
    )
    transition_delta = (
        fault_metrics["balanced_repair_gain"]
        - fault_metrics["balanced_damage_loss"]
    )
    if not np.isclose(observed_delta, transition_delta, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Repair/damage accounting does not reproduce the balanced-accuracy "
            f"change: observed={observed_delta}, transitions={transition_delta}."
        )


def make_result_row(
    model_alias: str,
    registry_name: str,
    flip_count: int,
    total_fault_sites: int,
    repeat: int,
    fault_seed: int,
    metrics: dict,
) -> dict:
    fraction = flip_count / total_fault_sites
    confusion = metrics["confusion_matrix"]
    row = {
        "model_alias": model_alias,
        "registry_name": registry_name,
        "flip_count_per_event": flip_count,
        "total_fault_sites_per_event": total_fault_sites,
        "equivalent_fault_fraction": fraction,
        "equivalent_fault_percentage": 100 * fraction,
        "repeat": repeat,
        "fault_seed": fault_seed,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "prediction_change_fraction": metrics["prediction_change_fraction"],
        "balanced_repair_gain": metrics["balanced_repair_gain"],
        "balanced_damage_loss": metrics["balanced_damage_loss"],
        "repair_fraction": metrics["repair_fraction"],
        "damage_fraction": metrics["damage_fraction"],
        **{f"recall_class_{index}": metrics["recalls"][index] for index in range(3)},
        **{f"n_class_{index}": metrics["class_totals"][index] for index in range(3)},
        **{f"repairs_class_{index}": metrics["repairs"][index] for index in range(3)},
        **{f"damages_class_{index}": metrics["damages"][index] for index in range(3)},
    }
    row.update(
        {
            f"confusion_{truth}_{prediction}": confusion[truth][prediction]
            for truth in range(3)
            for prediction in range(3)
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "QKeras robustness to exactly K random transient activation-bit "
            "faults per test event"
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--registry-name", required=True)
    parser.add_argument("--clean-model", type=Path, required=True)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("-c", "--data-config", default="config/10783560/v3.yml")
    parser.add_argument("-o", "--output", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024)
    parser.add_argument("--flip-counts", type=int, nargs="+", required=True)
    parser.add_argument("--fault-scope", choices=FAULT_SCOPES, default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    flip_counts = sorted(set(args.flip_counts))
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to estimate uncertainty.")
    if not flip_counts or flip_counts[0] < 0:
        parser.error("--flip-counts must contain non-negative integers.")
    if 0 not in flip_counts:
        flip_counts.insert(0, 0)
    if not args.clean_model.exists():
        parser.error(f"Clean checkpoint does not exist: {args.clean_model}")
    if not args.training_status.is_file():
        parser.error(
            f"Training completion metadata does not exist: {args.training_status}"
        )

    configure_device(args.device)
    set_fault_seed(args.seed)
    output = Path(args.output)
    config = get_config(args.registry_name)
    if config.alias != args.model_alias:
        parser.error(
            f"Registry {args.registry_name} belongs to {config.alias}, "
            f"not {args.model_alias}."
        )
    training_status = json.loads(args.training_status.read_text())
    input_range = QuantizationRange(**training_status["input_quantization_range"])

    _train_loader, _val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=0.0,
        balance_classes=False,
        input_type=config.input_type,
        target_type=config.task_type,
        drop_dominant_features=False,
    )

    clean_model = load_model(args.clean_model, registry_name=args.registry_name)
    fault_metadata = stage_metadata(config, args.fault_scope)
    total_fault_sites = int(fault_metadata["total_fault_sites_per_event"])
    if flip_counts[-1] > total_fault_sites:
        parser.error(
            f"Requested {flip_counts[-1]} flips, but this model/scope has only "
            f"{total_fault_sites} eligible activation bits per event."
        )

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        print(f"[SIGNAL] Received {signum}; finishing current evaluation.", flush=True)
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    rows: list[dict] = []
    latest_confusions: dict[str, list[list[int]]] = {}

    rng = set_fault_seed(args.seed)
    clean_metrics, clean_predictions = evaluate_complete_test_split(
        clean_model,
        test_loader,
        config,
        args.fault_scope,
        input_range,
        flip_count=0,
        rng=rng,
        collect_predictions=True,
    )
    if clean_predictions is None:
        raise RuntimeError("Clean reference predictions were not collected.")
    validate_clean_checkpoint_metrics(
        training_status,
        args.model_alias,
        args.registry_name,
        clean_metrics,
    )
    np.save(output / "clean_predictions.npy", clean_predictions)
    rows.append(
        make_result_row(
            args.model_alias,
            args.registry_name,
            0,
            total_fault_sites,
            0,
            args.seed,
            clean_metrics,
        )
    )
    latest_confusions["K=0:repeat=0"] = clean_metrics["confusion_matrix"]
    write_rows(output / "bit_flip_results.csv", rows)

    for flip_count in (value for value in flip_counts if value > 0):
        if stop:
            break
        for repeat in range(args.repeats):
            if stop:
                break
            fault_seed = args.seed + 1_000_003 * flip_count + repeat
            rng = set_fault_seed(fault_seed)
            metrics, _ = evaluate_complete_test_split(
                clean_model,
                test_loader,
                config,
                args.fault_scope,
                input_range,
                flip_count=flip_count,
                rng=rng,
                reference_predictions=clean_predictions,
            )
            validate_transition_accounting(clean_metrics, metrics)
            rows.append(
                make_result_row(
                    args.model_alias,
                    args.registry_name,
                    flip_count,
                    total_fault_sites,
                    repeat,
                    fault_seed,
                    metrics,
                )
            )
            latest_confusions[
                f"K={flip_count}:repeat={repeat}"
            ] = metrics["confusion_matrix"]
            write_rows(output / "bit_flip_results.csv", rows)
            print(
                f"[POINT] K={flip_count} repeat={repeat + 1}/{args.repeats} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.6f}",
                flush=True,
            )

    complete = (
        not stop
        and len(rows) == 1 + args.repeats * (len(flip_counts) - 1)
    )
    metadata = {
        "model_alias": args.model_alias,
        "registry_name": args.registry_name,
        "clean_checkpoint": str(args.clean_model),
        "clean_checkpoint_sha256": sha256_path(args.clean_model),
        "training_status": str(args.training_status),
        "training_final_balanced_accuracy": float(
            training_status["final_balanced_accuracy"]
        ),
        "flip_counts_per_event": flip_counts,
        "fault_scope": args.fault_scope,
        **fault_metadata,
        "input_quantization_range": input_range.to_dict(),
        "repeats_per_nonzero_count": args.repeats,
        "number_test_events": clean_metrics["number_test_events"],
        "uses_complete_configured_test_split": True,
        "balanced_accuracy": "Unweighted mean of all three aggregate class recalls.",
        "prediction_change_fraction": (
            "Fraction of test events whose predicted class differs from the "
            "same event with K=0."
        ),
        "repair_and_damage": (
            "A repair is an event wrong at K=0 and correct with faults; damage "
            "is correct at K=0 and wrong with faults. balanced_repair_gain and "
            "balanced_damage_loss are the macro averages of their per-class "
            "rates, so balanced_accuracy(K)-balanced_accuracy(0) equals gain-loss."
        ),
        "fault_model": (
            "For every test event, uniformly sample exactly K distinct activation "
            "bits without replacement from the eligible QKeras activation-bit "
            "space. Scope 'input' includes the 14 y-profile input features after "
            "training-set calibrated activation-bit quantization. Scope 'all' "
            "adds every hidden QActivation output bit. Logits, weights, biases, "
            "and batch-normalization state are unchanged."
        ),
        "percentage_definition": (
            "100 * K / total_fault_sites_per_event; derived from the exact-K "
            "experiment, not independent Bernoulli fault injection."
        ),
        "complete": complete,
        "confusion_matrices": latest_confusions,
    }
    (output / "bit_flip_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    if not complete:
        raise SystemExit(
            "Study stopped before all requested fault realizations completed; "
            "partial CSV and metadata were retained."
        )
    print(f"[DONE] Outputs written to {output}", flush=True)


if __name__ == "__main__":
    main()
