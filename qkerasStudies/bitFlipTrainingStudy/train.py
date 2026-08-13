#!/usr/bin/env python3
"""Train one QKeras model with optional transient activation-bit faults."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smartpixels-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataloader import create_dataloaders
from utils import CreateFolder

from qkerasStudies.common import (
    CLASS_NAMES,
    balanced_accuracy_from_confusion,
    compute_input_quantization_range,
    configure_device,
    confusion_from_predictions,
    get_class_weights,
    to_numpy,
)
from qkerasStudies.fault_injection import (
    FAULT_SCOPES,
    QuantizationRange,
    forward_with_faults,
    stage_metadata,
    validate_fault_probability,
)
from qkerasStudies.models import (
    CONFIG_BY_ALIAS,
    build_model,
    count_trainable_params,
    get_config,
    write_model_metadata,
)


CSV_FIELDS = (
    "model_alias",
    "registry_name",
    "step",
    "training_fault_probability",
    "training_fault_percentage",
    "training_fault_scope",
    "total_fault_sites_per_event",
    "expected_training_flips_per_event",
    "fault_count_standard_deviation",
    "clean_balanced_accuracy",
    "clean_recall_class_0",
    "clean_recall_class_1",
    "clean_recall_class_2",
    "n_class_0",
    "n_class_1",
    "n_class_2",
    *(f"clean_confusion_{truth}_{prediction}" for truth in range(3) for prediction in range(3)),
)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_training_history(
    rows: list[dict],
    model_alias: str,
    output_stem: Path,
) -> None:
    if not rows:
        return
    steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
    accuracy = 100 * np.asarray(
        [float(row["clean_balanced_accuracy"]) for row in rows],
        dtype=float,
    )

    fig, (accuracy_axis, recall_axis) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    accuracy_axis.plot(steps, accuracy, color="#3366a6", marker="o")
    probability = 100 * float(rows[0]["training_fault_probability"])
    scope = rows[0].get("training_fault_scope", "all")
    accuracy_axis.set_ylabel("Clean balanced accuracy [%]")
    if probability == 0.0:
        title = f"{model_alias}: QKeras clean baseline training"
    else:
        title = (
            f"{model_alias}: QKeras fault-aware training "
            f"({probability:g}% per activation bit, {scope})"
        )
    accuracy_axis.set_title(title)
    accuracy_axis.grid(True, alpha=0.25)

    colors = ("#3366a6", "#d05a3a", "#2f8a62")
    for class_index, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        recall_axis.plot(
            steps,
            100
            * np.asarray(
                [
                    float(row[f"clean_recall_class_{class_index}"])
                    for row in rows
                ],
                dtype=float,
            ),
            color=color,
            marker="o",
            label=class_name,
        )
    recall_axis.set_xlabel("Optimizer step")
    recall_axis.set_ylabel("Clean recall by true class [%]")
    recall_axis.legend(loc="best", fontsize=8)
    recall_axis.grid(True, alpha=0.25)

    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def evaluate_complete_test_split(
    model,
    loader,
    config,
    input_quantization_range: QuantizationRange,
) -> dict:
    confusion = np.zeros((3, 3), dtype=np.int64)
    number_seen = 0
    for inputs, labels in tqdm(
        loader,
        desc="Full clean test evaluation",
        unit="batch",
        leave=False,
    ):
        x_np = to_numpy(inputs).astype(np.float32, copy=False)
        y_np = to_numpy(labels)
        logits = forward_with_faults(
            model,
            x_np,
            config,
            training=False,
            fault_scope="all",
            input_quantization_range=input_quantization_range,
        )
        predictions = np.argmax(to_numpy(logits), axis=-1)
        truth = np.argmax(y_np, axis=-1)
        confusion += confusion_from_predictions(truth, predictions)
        number_seen += len(predictions)

    metrics = balanced_accuracy_from_confusion(confusion)
    metrics["number_test_events"] = number_seen
    return metrics


def record_clean_stage(
    model,
    config,
    model_alias: str,
    step: int,
    test_loader,
    input_quantization_range: QuantizationRange,
    training_fault_probability: float,
    fault_scope: str,
    output: Path,
    rows: list[dict],
) -> dict:
    metrics = evaluate_complete_test_split(
        model,
        test_loader,
        config,
        input_quantization_range,
    )
    fault_metadata = stage_metadata(config, fault_scope)
    total_fault_sites = int(fault_metadata["total_fault_sites_per_event"])
    confusion = metrics["confusion_matrix"]
    row = {
        "model_alias": model_alias,
        "registry_name": config.registry_name,
        "step": step,
        "training_fault_probability": training_fault_probability,
        "training_fault_percentage": 100 * training_fault_probability,
        "training_fault_scope": fault_scope,
        "total_fault_sites_per_event": total_fault_sites,
        "expected_training_flips_per_event": (
            training_fault_probability * total_fault_sites
        ),
        "fault_count_standard_deviation": math.sqrt(
            total_fault_sites
            * training_fault_probability
            * (1 - training_fault_probability)
        ),
        "clean_balanced_accuracy": metrics["balanced_accuracy"],
        **{
            f"clean_recall_class_{index}": metrics["recalls"][index]
            for index in range(3)
        },
        **{
            f"n_class_{index}": metrics["class_totals"][index]
            for index in range(3)
        },
        **{
            f"clean_confusion_{truth}_{prediction}": confusion[truth][prediction]
            for truth in range(3)
            for prediction in range(3)
        },
    }
    rows.append(row)
    write_rows(output / "fault_aware_training_history.csv", rows)
    plot_training_history(
        rows,
        model_alias,
        output / "clean_accuracy_during_fault_aware_training",
    )
    print(
        f"[STAGE] step={step} clean_balanced_acc="
        f"{metrics['balanced_accuracy']:.6f}",
        flush=True,
    )
    return metrics


def training_requirements_met(
    step: int,
    minimum_steps: int,
    elapsed_seconds: float,
    minimum_training_hours: float,
) -> bool:
    return (
        step >= minimum_steps
        and elapsed_seconds >= minimum_training_hours * 3600
    )


def save_model(model, output_path: Path, metadata: dict) -> None:
    temporary_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    if temporary_path.exists():
        if temporary_path.is_dir():
            import shutil

            shutil.rmtree(temporary_path)
        else:
            temporary_path.unlink()
    model.save(temporary_path)
    if output_path.exists():
        if output_path.is_dir():
            import shutil

            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    temporary_path.replace(output_path)
    write_model_metadata(output_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a selected y-profile QKeras model with activation-bit faults"
    )
    parser.add_argument(
        "--model-alias",
        "--model",
        choices=sorted(CONFIG_BY_ALIAS),
        required=True,
    )
    parser.add_argument(
        "--training-fault-probability",
        type=float,
        required=True,
        help="Independent per-activation-bit training fault probability in [0, 1].",
    )
    parser.add_argument("--fault-scope", choices=FAULT_SCOPES, default="all")
    parser.add_argument("--data-config", "-c", default="config/10783560/v3.yml")
    parser.add_argument("--output", "-o", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", "-bs", type=int, default=1024)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--minimum-training-hours", type=float, default=3.0)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--val-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-classes", action="store_true")
    args = parser.parse_args()

    try:
        fault_probability = validate_fault_probability(
            args.training_fault_probability
        )
    except ValueError as error:
        parser.error(str(error))
    if args.steps < 1 or args.eval_freq < 1 or args.minimum_training_hours < 0:
        parser.error(
            "--steps and --eval-freq must be positive, and "
            "--minimum-training-hours must be non-negative."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    configure_device(args.device)

    output = Path(args.output)
    completion_path = output / "training_complete.json"
    incomplete_path = output / "training_incomplete.json"
    completion_path.unlink(missing_ok=True)
    incomplete_path.unlink(missing_ok=True)

    config = get_config(args.model_alias)
    train_loader, _val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=args.val_size,
        balance_classes=args.balance_classes,
        input_type=config.input_type,
        target_type=config.task_type,
        drop_dominant_features=False,
    )
    input_range_dict = compute_input_quantization_range(train_loader)
    input_quantization_range = QuantizationRange(**input_range_dict)
    class_weights = get_class_weights(train_loader)

    model = build_model(config.registry_name)
    n_params = count_trainable_params(model)
    print(
        f"Selected model: {args.model_alias} ({config.registry_name}); "
        f"trainable parameters={n_params}",
        flush=True,
    )
    print(
        f"Input activation-bit quantization range: {input_range_dict}",
        flush=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    class_weights_tf = tf.constant(class_weights.astype(np.float32))

    rows: list[dict] = []
    step = 0
    epoch = 0
    stop_requested = False
    start_time = time.monotonic()
    latest_metrics: dict | None = None

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[SIGNAL] Received {signum}; stopping after current batch.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while True:
            if hasattr(train_loader.dataset, "shuffle_indices"):
                train_loader.dataset.shuffle_indices()
            for inputs, labels in tqdm(
                train_loader,
                desc=f"Training (epoch {epoch})",
                unit="batch",
                leave=True,
            ):
                step += 1
                x_np = to_numpy(inputs).astype(np.float32, copy=False)
                y_np = to_numpy(labels).astype(np.float32, copy=False)
                with tf.GradientTape() as tape:
                    logits = forward_with_faults(
                        model,
                        x_np,
                        config,
                        training=True,
                        fault_scope=args.fault_scope,
                        input_quantization_range=input_quantization_range,
                        training_fault_probability=fault_probability,
                    )
                    per_event_loss = loss_fn(y_np, logits)
                    truth = tf.argmax(y_np, axis=-1, output_type=tf.int32)
                    sample_weights = tf.gather(class_weights_tf, truth)
                    loss = tf.reduce_mean(per_event_loss * sample_weights)

                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                if step % 100 == 0:
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.reduce_mean(
                        tf.cast(predictions == truth, tf.float32)
                    )
                    print(
                        f"[TRAIN] step={step} loss={float(loss):.6f} "
                        f"batch_acc={float(accuracy):.4f}",
                        flush=True,
                    )

                should_record = step % args.eval_freq == 0
                if should_record:
                    latest_metrics = record_clean_stage(
                        model,
                        config,
                        args.model_alias,
                        step,
                        test_loader,
                        input_quantization_range,
                        fault_probability,
                        args.fault_scope,
                        output,
                        rows,
                    )
                    model_metadata = {
                        "model": config.registry_name,
                        "registry_name": config.registry_name,
                        "model_alias": args.model_alias,
                        "input_type": config.input_type,
                        "task_type": config.task_type,
                        "weight_bits": config.weight_bits,
                        "activation_bits": config.activation_bits,
                        "hidden_units": list(config.hidden_units),
                    }
                    save_model(model, output / "latest_model.keras", model_metadata)

                elapsed_seconds = time.monotonic() - start_time
                if stop_requested:
                    raise KeyboardInterrupt
                if training_requirements_met(
                    step,
                    args.steps,
                    elapsed_seconds,
                    args.minimum_training_hours,
                ):
                    raise StopIteration
            epoch += 1
    except StopIteration:
        pass
    except KeyboardInterrupt:
        elapsed_seconds = time.monotonic() - start_time
        model_metadata = {
            "model": config.registry_name,
            "registry_name": config.registry_name,
            "model_alias": args.model_alias,
            "input_type": config.input_type,
            "task_type": config.task_type,
            "weight_bits": config.weight_bits,
            "activation_bits": config.activation_bits,
            "hidden_units": list(config.hidden_units),
        }
        save_model(model, output / "interrupted_model.keras", model_metadata)
        incomplete = {
            "complete": False,
            "stop_requested": True,
            "model_alias": args.model_alias,
            "registry_name": config.registry_name,
            "completed_steps": step,
            "minimum_steps": args.steps,
            "elapsed_training_hours": elapsed_seconds / 3600,
            "minimum_training_hours": args.minimum_training_hours,
        }
        incomplete_path.write_text(json.dumps(incomplete, indent=2) + "\n")
        raise SystemExit("Training stopped before completion requirements were met.")

    if latest_metrics is None or (not rows) or int(rows[-1]["step"]) != step:
        latest_metrics = record_clean_stage(
            model,
            config,
            args.model_alias,
            step,
            test_loader,
            input_quantization_range,
            fault_probability,
            args.fault_scope,
            output,
            rows,
        )

    elapsed_seconds = time.monotonic() - start_time
    fault_metadata = stage_metadata(config, args.fault_scope)
    model_metadata = {
        "model": config.registry_name,
        "registry_name": config.registry_name,
        "model_alias": args.model_alias,
        "input_type": config.input_type,
        "task_type": config.task_type,
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "hidden_units": list(config.hidden_units),
        "input_quantization_range": input_range_dict,
    }
    save_model(model, output / "final_model.keras", model_metadata)

    completion = {
        "complete": True,
        "completion_basis": "qkeras_fault_aware_training_loop",
        "training_condition": (
            "clean_baseline" if fault_probability == 0.0 else "fault_aware"
        ),
        "stop_requested": False,
        "model_alias": args.model_alias,
        "registry_name": config.registry_name,
        "training_fault_probability": fault_probability,
        "training_fault_percentage": 100 * fault_probability,
        "training_fault_scope": args.fault_scope,
        "total_fault_sites_per_event": fault_metadata["total_fault_sites_per_event"],
        "stage_widths": fault_metadata["stage_widths"],
        "stage_bit_widths": fault_metadata["stage_bit_widths"],
        "model_stage_widths": fault_metadata["model_stage_widths"],
        "eligible_model_stage_indices": fault_metadata[
            "eligible_model_stage_indices"
        ],
        "input_quantization_range": input_range_dict,
        "minimum_training_hours": args.minimum_training_hours,
        "elapsed_training_hours": elapsed_seconds / 3600,
        "minimum_steps": args.steps,
        "completed_steps": step,
        "final_recorded_step": step,
        "final_balanced_accuracy": latest_metrics["balanced_accuracy"],
        "final_recalls": latest_metrics["recalls"],
        "final_class_totals": latest_metrics["class_totals"],
        "final_confusion_matrix": latest_metrics["confusion_matrix"],
        "number_test_events": latest_metrics["number_test_events"],
        "trainable_parameters": n_params,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    print(
        f"[DONE] steps={step} hours={elapsed_seconds / 3600:.3f} "
        f"clean_balanced_acc={latest_metrics['balanced_accuracy']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
