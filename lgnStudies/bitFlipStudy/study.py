#!/usr/bin/env python3
"""Fine-tune and evaluate model-2 LGNs under random activation bit flips."""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from models.registry import class_to_config, load_registered_model
from utils import CreateFolder, get_device

from lgnStudies.bitFlipStudy.fault_injection import ActivationBitFlipModel
from lgnStudies.bitFlipStudy.plots import plot_single_model


DEFAULT_INFERENCE_PROBABILITIES = (
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)
CSV_FIELDS = (
    "model_alias",
    "registry_name",
    "variant",
    "training_flip_probability",
    "inference_flip_probability",
    "repeat",
    "fault_seed",
    "balanced_accuracy",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "n_class_0",
    "n_class_1",
    "n_class_2",
)


def get_class_weights(data_loader) -> torch.Tensor:
    labels = data_loader.dataset.get_all_class_labels()
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError(f"Invalid training class counts: {counts.tolist()}.")
    return torch.as_tensor(
        len(labels) / (len(counts) * counts), dtype=torch.float32
    )


def validate_model(model, expected_registry_name: str) -> None:
    registry_name, config = class_to_config(type(model))
    if registry_name != expected_registry_name:
        raise ValueError(
            f"Checkpoint contains {registry_name}, expected {expected_registry_name}."
        )
    if config["input_type"] != "y-profile":
        raise ValueError("Bit-flip study is restricted to model-2 y-profile LGNs.")


def evaluate_complete_test_split(
    model: ActivationBitFlipModel,
    loader,
    device: torch.device,
) -> dict:
    original_mode = model.training
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in tqdm(
            loader, desc="Full test evaluation", unit="batch", leave=False
        ):
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            truth = labels.argmax(dim=1).to(torch.int64)
            confusion += torch.bincount(
                3 * truth + predictions, minlength=9
            ).reshape(3, 3)
    model.train(original_mode)

    totals = confusion.sum(dim=1)
    if torch.any(totals == 0):
        raise RuntimeError(f"Missing test class; totals={totals.tolist()}.")
    recalls = confusion.diag().to(torch.float64) / totals
    return {
        "balanced_accuracy": float(recalls.mean()),
        "recalls": [float(value) for value in recalls],
        "class_totals": [int(value) for value in totals],
        "confusion_matrix": confusion.tolist(),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fine_tune_with_faults(
    model,
    train_loader,
    device: torch.device,
    flip_probability: float,
    learning_rate: float,
    weight_decay: float,
    steps: int,
    compile_model: bool,
    stop_requested,
) -> int:
    wrapped = ActivationBitFlipModel(model, flip_probability).to(device)
    optimizer = torch.optim.AdamW(
        wrapped.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=get_class_weights(train_loader).to(device)
    )
    if compile_model:
        print("Compiling the fault-aware training model...", flush=True)
        wrapped.compile(dynamic=True)

    step = 0
    epoch = 0
    wrapped.train()
    while step < steps and not stop_requested():
        train_loader.dataset.shuffle_indices()
        progress = tqdm(
            train_loader, desc=f"Fault-aware fine-tuning epoch {epoch}", unit="batch"
        )
        for inputs, labels in progress:
            if step >= steps or stop_requested():
                break
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = wrapped(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                progress.set_postfix(step=step, loss=f"{loss.item():.4f}")
        epoch += 1
    return step


def make_result_row(
    model_alias: str,
    registry_name: str,
    variant: str,
    training_probability: float,
    inference_probability: float,
    repeat: int,
    fault_seed: int,
    metrics: dict,
) -> dict:
    return {
        "model_alias": model_alias,
        "registry_name": registry_name,
        "variant": variant,
        "training_flip_probability": (
            training_probability if variant == "robust" else 0.0
        ),
        "inference_flip_probability": inference_probability,
        "repeat": repeat,
        "fault_seed": fault_seed,
        "balanced_accuracy": metrics["balanced_accuracy"],
        **{f"recall_class_{i}": metrics["recalls"][i] for i in range(3)},
        **{f"n_class_{i}": metrics["class_totals"][i] for i in range(3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-2 LGN random bit-flip robustness study"
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--registry-name", required=True)
    parser.add_argument("--clean-model", type=Path, required=True)
    parser.add_argument("--robust-model", type=Path)
    parser.add_argument("-c", "--data-config", default="config/10783560/v3.yml")
    parser.add_argument("-o", "--output", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--robust-steps", type=int, default=500)
    parser.add_argument("--train-flip-probability", type=float, default=1e-3)
    parser.add_argument(
        "--inference-flip-probabilities",
        type=float,
        nargs="+",
        default=DEFAULT_INFERENCE_PROBABILITIES,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()

    probabilities = sorted(set(args.inference_flip_probabilities))
    if (
        args.robust_steps < 0
        or args.repeats < 1
        or not 0.0 <= args.train_flip_probability <= 1.0
        or any(probability < 0 or probability > 1 for probability in probabilities)
    ):
        parser.error("Invalid steps, repeats, or flip probability.")
    if not args.clean_model.is_file():
        parser.error(f"Clean checkpoint does not exist: {args.clean_model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)
    output = Path(args.output)

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

    clean_model = load_registered_model(
        str(args.clean_model), map_location=device
    ).to(device)
    validate_model(clean_model, args.registry_name)

    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        print(f"[SIGNAL] Received {signum}; finishing current operation.", flush=True)
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.robust_model is not None:
        if not args.robust_model.is_file():
            parser.error(f"Robust checkpoint does not exist: {args.robust_model}")
        robust_model = load_registered_model(
            str(args.robust_model), map_location=device
        ).to(device)
        validate_model(robust_model, args.registry_name)
        completed_steps = 0
    else:
        robust_model = load_registered_model(
            str(args.clean_model), map_location=device
        ).to(device)
        validate_model(robust_model, args.registry_name)
        completed_steps = fine_tune_with_faults(
            robust_model,
            train_loader,
            device=device,
            flip_probability=args.train_flip_probability,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            steps=args.robust_steps,
            compile_model=args.compile_model,
            stop_requested=lambda: stop,
        )

    robust_path = output / "bitflip_robust_model.pth"
    torch.save(robust_model, robust_path)
    print(f"Saved fault-aware model to {robust_path}", flush=True)

    clean_fault_model = ActivationBitFlipModel(clean_model).to(device)
    robust_fault_model = ActivationBitFlipModel(robust_model).to(device)
    rows: list[dict] = []
    latest_confusions = {}

    for probability in probabilities:
        if stop:
            break
        for repeat in range(args.repeats):
            fault_seed = args.seed + 10_000 * repeat
            for variant, fault_model in (
                ("clean", clean_fault_model),
                ("robust", robust_fault_model),
            ):
                fault_model.set_flip_probability(probability)
                # Paired masks: both variants see the same stochastic faults.
                torch.manual_seed(fault_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(fault_seed)
                metrics = evaluate_complete_test_split(
                    fault_model, test_loader, device
                )
                rows.append(
                    make_result_row(
                        args.model_alias,
                        args.registry_name,
                        variant,
                        args.train_flip_probability,
                        probability,
                        repeat,
                        fault_seed,
                        metrics,
                    )
                )
                latest_confusions[
                    f"{variant}:p={probability}:repeat={repeat}"
                ] = metrics["confusion_matrix"]

            write_rows(output / "bit_flip_results.csv", rows)
            plot_single_model(
                rows,
                args.model_alias,
                args.train_flip_probability,
                output / "bit_flip_robustness",
            )
            print(
                f"[POINT] p={probability:.3g} repeat={repeat} complete",
                flush=True,
            )

    metadata = {
        "model_alias": args.model_alias,
        "registry_name": args.registry_name,
        "clean_checkpoint": str(args.clean_model),
        "robust_checkpoint": str(robust_path),
        "fault_aware_steps_completed": completed_steps,
        "training_flip_probability": args.train_flip_probability,
        "inference_flip_probabilities": probabilities,
        "repeats": args.repeats,
        "uses_complete_configured_test_split": True,
        "balanced_accuracy": "Unweighted mean of all three aggregate class recalls.",
        "fault_model": (
            "Independent x -> 1-x activation flips after FixedBinarization and "
            "after every LogicDense layer; GroupSum is not flipped."
        ),
        "latest_confusion_matrices": latest_confusions,
    }
    (output / "bit_flip_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"[DONE] Outputs written to {output}", flush=True)


if __name__ == "__main__":
    main()
