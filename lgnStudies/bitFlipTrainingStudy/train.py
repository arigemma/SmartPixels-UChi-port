#!/usr/bin/env python3
"""Train one model-2 LGN with transient activation faults."""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from models.registry import REGISTRY
from utils import CreateFolder, get_device

from lgnStudies.bitFlipTrainingStudy.fault_injection import (
    FAULT_SCOPES,
    FaultAwareTrainingModel,
    validate_fault_probability,
)
from lgnStudies.resource_precision_study import (
    CLASS_NAMES,
    MODEL_ALIASES,
    derive_thresholds_from_data,
    exact_balanced_accuracy,
    get_class_weights,
    save_model,
    training_requirements_met,
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
    """Plot clean balanced accuracy and per-class recall during training."""
    steps = np.asarray([int(row["step"]) for row in rows], dtype=float)
    accuracy = 100 * np.asarray(
        [float(row["clean_balanced_accuracy"]) for row in rows]
    )

    fig, (accuracy_axis, recall_axis) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    accuracy_axis.plot(steps, accuracy, color="#3366a6", marker="o")
    accuracy_axis.set_ylabel("Clean balanced accuracy [%]")
    probability = 100 * float(rows[0]["training_fault_probability"])
    scope = rows[0].get("training_fault_scope", "all")
    accuracy_axis.set_title(
        f"{model_alias}: fault-aware training ({probability:g}% per site, {scope})"
    )
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
                ]
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


def record_clean_stage(
    base_model: torch.nn.Module,
    fault_model: FaultAwareTrainingModel,
    model_alias: str,
    registry_name: str,
    step: int,
    test_loader,
    device: torch.device,
    output: Path,
    rows: list[dict],
) -> None:
    """Evaluate the uncorrupted discrete model over the complete test split."""
    metrics = exact_balanced_accuracy(base_model, test_loader, device)
    confusion = metrics["confusion_matrix"]
    row = {
        "model_alias": model_alias,
        "registry_name": registry_name,
        "step": step,
        "training_fault_probability": fault_model.fault_probability,
        "training_fault_percentage": 100 * fault_model.fault_probability,
        "training_fault_scope": fault_model.fault_scope,
        "total_fault_sites_per_event": fault_model.total_fault_sites,
        "expected_training_flips_per_event": fault_model.expected_flips_per_event,
        "fault_count_standard_deviation": fault_model.fault_count_standard_deviation,
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
        f"[STAGE] step={step} "
        f"clean_balanced_acc={metrics['balanced_accuracy']:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model-2 LGN while independently flipping each eligible "
            "activation with probability p"
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument(
        "--training-fault-probability",
        type=float,
        required=True,
        help="Independent per-site training fault probability in [0, 1].",
    )
    parser.add_argument(
        "--fault-scope",
        choices=FAULT_SCOPES,
        default="all",
        help=(
            "Eligible activation space for training faults: all keeps the "
            "FixedBinarization plus every LogicDense output; input uses only "
            "the FixedBinarization output."
        ),
    )
    parser.add_argument("--data-config", "-c", default="config/10783560/v3.yml")
    parser.add_argument("--output", "-o", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", "-bs", type=int, default=1024)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--minimum-training-hours", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()

    try:
        fault_probability = validate_fault_probability(
            args.training_fault_probability
        )
    except ValueError as error:
        parser.error(str(error))
    if fault_probability == 0:
        parser.error(
            "--training-fault-probability must be nonzero for this study."
        )
    if (
        args.steps < 1
        or args.eval_freq < 1
        or args.minimum_training_hours < 0
    ):
        parser.error(
            "--steps and --eval-freq must be positive, and "
            "--minimum-training-hours must be non-negative."
        )

    output = Path(args.output)
    completion_path = output / "training_complete.json"
    incomplete_path = output / "training_incomplete.json"
    completion_path.unlink(missing_ok=True)
    incomplete_path.unlink(missing_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    registry_name = MODEL_ALIASES[args.model]
    model_cfg = REGISTRY[registry_name]
    if model_cfg["input_type"] != "y-profile":
        raise RuntimeError("This study is restricted to model-2 y-profile input.")

    train_loader, _, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=0.01,
        balance_classes=True,
        input_type="y-profile",
        target_type="classification",
        drop_dominant_features=False,
    )
    model_class = model_cfg["class"]
    thresholds = derive_thresholds_from_data(
        train_loader, num_bits=model_class.n_bits
    )
    base_model = model_class(thresholds=thresholds).to(device)
    fault_model = FaultAwareTrainingModel(
        base_model,
        fault_probability=fault_probability,
        fault_scope=args.fault_scope,
    ).to(device)

    class_weights = get_class_weights(train_loader).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        fault_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.compile_model:
        print("Compiling the fault-aware training model...", flush=True)
        fault_model.compile(dynamic=True)

    print(
        "[FAULT MODEL] "
        f"scope={fault_model.fault_scope} "
        f"sites={fault_model.total_fault_sites} "
        f"p={fault_model.fault_probability:.8f} "
        f"expected_K={fault_model.expected_flips_per_event:.3f} "
        f"std_K={fault_model.fault_count_standard_deviation:.3f}",
        flush=True,
    )

    rows: list[dict] = []
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(
            f"[SIGNAL] Received {signum}; stopping after the current batch.",
            flush=True,
        )
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    record_clean_stage(
        base_model,
        fault_model,
        args.model,
        registry_name,
        0,
        test_loader,
        device,
        output,
        rows,
    )
    save_model(base_model, output / "latest_model.pth")

    step = 0
    epoch = 0
    training_started = time.monotonic()

    def elapsed_training_seconds() -> float:
        return time.monotonic() - training_started

    def requirements_met() -> bool:
        return training_requirements_met(
            step,
            args.steps,
            elapsed_training_seconds(),
            args.minimum_training_hours,
        )

    fault_model.train()
    while not requirements_met() and not stop_requested:
        train_loader.dataset.shuffle_indices()
        progress = tqdm(train_loader, desc=f"Training epoch {epoch}", unit="batch")
        for inputs, labels in progress:
            if requirements_met() or stop_requested:
                break
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = fault_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                progress.set_postfix(
                    step=step,
                    loss=f"{loss.item():.4f}",
                    hours=f"{elapsed_training_seconds() / 3600:.2f}",
                )

            if step % args.eval_freq == 0:
                record_clean_stage(
                    base_model,
                    fault_model,
                    args.model,
                    registry_name,
                    step,
                    test_loader,
                    device,
                    output,
                    rows,
                )
                save_model(base_model, output / "latest_model.pth")
                fault_model.train()
        epoch += 1

    complete = requirements_met() and not stop_requested
    if rows[-1]["step"] != step and complete:
        record_clean_stage(
            base_model,
            fault_model,
            args.model,
            registry_name,
            step,
            test_loader,
            device,
            output,
            rows,
        )
        save_model(base_model, output / "latest_model.pth")

    elapsed_hours = elapsed_training_seconds() / 3600
    status = {
        "complete": complete,
        "study": "fault_aware_activation_bit_flip_training",
        "model_alias": args.model,
        "registry_name": registry_name,
        "minimum_steps": args.steps,
        "completed_steps": step,
        "minimum_training_hours": args.minimum_training_hours,
        "elapsed_training_hours": elapsed_hours,
        "training_loop_wall_hours": elapsed_hours,
        "final_recorded_step": int(rows[-1]["step"]),
        "final_balanced_accuracy": float(
            rows[-1]["clean_balanced_accuracy"]
        ),
        "final_metric_fault_condition": "clean inference, p=0",
        "training_fault_probability": fault_model.fault_probability,
        "training_fault_percentage": 100 * fault_model.fault_probability,
        "training_fault_scope": fault_model.fault_scope,
        "total_fault_sites_per_event": fault_model.total_fault_sites,
        "stage_widths": fault_model.stage_widths,
        "model_stage_widths": fault_model.model_stage_widths,
        "eligible_model_stage_indices": fault_model._eligible_stage_indices,
        "expected_training_flips_per_event": (
            fault_model.expected_flips_per_event
        ),
        "fault_count_standard_deviation": (
            fault_model.fault_count_standard_deviation
        ),
        "training_fault_definition": (
            "On every training forward pass, independently select every "
            "eligible activation with probability p and apply x -> 1-x. "
            "Scope 'all' includes FixedBinarization and every LogicDense "
            "output; scope 'input' includes only FixedBinarization. GroupSum "
            "is excluded. Faults are disabled for clean stage evaluation."
        ),
        "stop_requested": stop_requested,
    }
    if complete:
        save_model(base_model, output / "final_model.pth")
        completion_path.write_text(json.dumps(status, indent=2) + "\n")
        print(
            f"[DONE] step={step} training_hours={elapsed_hours:.3f} "
            f"clean_balanced_acc={rows[-1]['clean_balanced_accuracy']:.6f} "
            f"output={output}",
            flush=True,
        )
    else:
        save_model(base_model, output / "interrupted_model.pth")
        incomplete_path.write_text(json.dumps(status, indent=2) + "\n")
        raise SystemExit(
            "Training stopped before all completion requirements were met; "
            "wrote interrupted_model.pth and did not publish final_model.pth."
        )


if __name__ == "__main__":
    main()
