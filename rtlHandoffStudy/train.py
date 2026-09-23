#!/usr/bin/env python3
"""Train one EFLX-oriented LGN for both minimum steps and wall time."""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from models.registry import REGISTRY
from models.rtl_handoff import RTL_HANDOFF_CONFIGS, get_rtl_handoff_config
from rtlHandoffStudy.portable import compare_frozen_outputs, export_deployment_artifacts
from utils import get_device


def _derive_thresholds(loader, n_bits: int) -> torch.Tensor:
    from torchlogix.layers import Binarization

    samples = []
    for batch_index, (inputs, _) in enumerate(loader):
        if batch_index >= 15:
            break
        samples.append(inputs)
    if not samples:
        raise RuntimeError("Training loader did not yield data for threshold initialization.")
    return Binarization.get_initial_thresholds(
        torch.cat(samples, dim=0),
        num_bits=int(n_bits),
        one_per="feature",
        method="distributive",
    )


def _class_weights(loader) -> torch.Tensor:
    labels = np.asarray(loader.dataset.get_all_class_labels(), dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError(f"Invalid training class counts: {counts.tolist()}.")
    return torch.as_tensor(len(labels) / (3 * counts), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, Any]:
    original_mode = model.training
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    loss_sum = 0.0
    event_count = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, labels in tqdm(loader, desc="Discrete validation", leave=False, unit="batch"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        truth = labels.argmax(dim=1)
        predictions = logits.argmax(dim=1)
        confusion += torch.bincount(
            3 * truth.cpu() + predictions.cpu(), minlength=9
        ).reshape(3, 3)
        loss_sum += float(criterion(logits, labels).item())
        event_count += int(labels.shape[0])
    model.train(original_mode)
    totals = confusion.sum(dim=1)
    if event_count == 0 or torch.any(totals == 0):
        raise RuntimeError(
            f"Invalid validation totals: events={event_count}, classes={totals.tolist()}."
        )
    recalls = confusion.diag().to(torch.float64) / totals
    return {
        "balanced_accuracy": float(recalls.mean().item()),
        "cross_entropy": float(loss_sum / event_count),
        "recalls": [float(value) for value in recalls],
        "events": event_count,
        "confusion_matrix": confusion.tolist(),
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("step", "training_hours", "balanced_accuracy", "cross_entropy", "best")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _requirements_met(step: int, minimum_steps: int, elapsed: float, hours: float) -> bool:
    return step >= minimum_steps and elapsed >= 3600 * hours


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--alias", choices=sorted(c["alias"] for c in RTL_HANDOFF_CONFIGS))
    parser.add_argument("--data-config", default="config/10783560/v3.yml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--minimum-steps", type=int, default=20_000)
    parser.add_argument("--minimum-training-hours", type=float, default=5.0)
    parser.add_argument("--eval-frequency", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.task_index is None) == (args.alias is None):
        parser.error("Specify exactly one of --task-index or --alias.")
    if args.task_index is not None:
        if not 0 <= args.task_index < len(RTL_HANDOFF_CONFIGS):
            parser.error(f"--task-index must be in [0, {len(RTL_HANDOFF_CONFIGS) - 1}].")
        alias = str(RTL_HANDOFF_CONFIGS[args.task_index]["alias"])
    else:
        alias = str(args.alias)
    if args.minimum_steps < 1 or args.minimum_training_hours < 0 or args.eval_frequency < 1:
        parser.error("Training requirements and evaluation frequency are invalid.")

    output = args.output
    complete_path = output / "training_complete.json"
    if complete_path.is_file() and not args.overwrite:
        print(f"[SKIP] Complete model already exists: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    for stale in (complete_path, output / "training_incomplete.json"):
        stale.unlink(missing_ok=True)

    config = get_rtl_handoff_config(alias)
    registry_name = str(config["registry_name"])
    registry = REGISTRY[registry_name]
    input_type = str(registry["input_type"])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)

    train_loader, validation_loader, _test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=0.01,
        balance_classes=True,
        input_type=input_type,
        target_type="classification",
        drop_dominant_features=False,
    )
    model_class = registry["class"]
    thresholds = _derive_thresholds(train_loader, int(model_class.n_bits))
    model = model_class(thresholds=thresholds).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(train_loader).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    training_model = torch.compile(model, dynamic=True) if args.compile_model else model

    campaign_metadata = model_class.deployment_config()
    campaign_metadata.update(
        {
            "registry_name": registry_name,
            "input_type": input_type,
            "seed": args.seed,
            "minimum_steps": args.minimum_steps,
            "minimum_training_hours": args.minimum_training_hours,
            "eval_frequency": args.eval_frequency,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }
    )
    (output / "campaign_config.json").write_text(
        json.dumps(campaign_metadata, indent=2) + "\n"
    )

    validation_sample = next(iter(validation_loader))[0][:2].to(device)
    history: list[dict[str, Any]] = []
    best_balanced_accuracy = -float("inf")
    best_step = -1

    def record(step: int, elapsed_hours: float) -> None:
        nonlocal best_balanced_accuracy, best_step
        metrics = evaluate(model, validation_loader, device)
        is_best = metrics["balanced_accuracy"] > best_balanced_accuracy
        row = {
            "step": step,
            "training_hours": f"{elapsed_hours:.9f}",
            "balanced_accuracy": f"{metrics['balanced_accuracy']:.12g}",
            "cross_entropy": f"{metrics['cross_entropy']:.12g}",
            "best": int(is_best),
        }
        history.append(row)
        _write_history(output / "training_history.tsv", history)
        print(
            f"[VALIDATION] alias={alias} step={step} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f} "
            f"cross_entropy={metrics['cross_entropy']:.6f}",
            flush=True,
        )
        if not is_best:
            return
        difference = compare_frozen_outputs(model, validation_sample)
        _atomic_torch_save(model, output / "best_model.pth")
        export_deployment_artifacts(
            model,
            output,
            validation_metrics=metrics,
            training_step=step,
        )
        best_balanced_accuracy = float(metrics["balanced_accuracy"])
        best_step = int(step)
        print(
            f"[BEST] step={step} balanced_accuracy={best_balanced_accuracy:.6f} "
            f"frozen_max_abs_diff={difference:g}",
            flush=True,
        )

    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        print(f"[SIGNAL] Received {signum}; stopping after the current batch.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    record(0, 0.0)
    step = 0
    epoch = 0
    started = time.monotonic()

    def elapsed_seconds() -> float:
        return time.monotonic() - started

    training_model.train()
    while not _requirements_met(
        step, args.minimum_steps, elapsed_seconds(), args.minimum_training_hours
    ) and not stop_requested:
        train_loader.dataset.shuffle_indices()
        progress = tqdm(train_loader, desc=f"Training epoch {epoch}", unit="batch")
        for inputs, labels in progress:
            if stop_requested or _requirements_met(
                step, args.minimum_steps, elapsed_seconds(), args.minimum_training_hours
            ):
                break
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = training_model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                progress.set_postfix(
                    step=step,
                    loss=f"{loss.item():.4f}",
                    hours=f"{elapsed_seconds() / 3600:.2f}",
                )
            if step % args.eval_frequency == 0:
                record(step, elapsed_seconds() / 3600)
                training_model.train()
        epoch += 1

    complete = _requirements_met(
        step, args.minimum_steps, elapsed_seconds(), args.minimum_training_hours
    )
    if complete and (not history or int(history[-1]["step"]) != step):
        record(step, elapsed_seconds() / 3600)

    status = {
        "complete": bool(complete),
        "alias": alias,
        "registry_name": registry_name,
        "completed_steps": int(step),
        "elapsed_training_hours": float(elapsed_seconds() / 3600),
        "minimum_steps": int(args.minimum_steps),
        "minimum_training_hours": float(args.minimum_training_hours),
        "best_step": int(best_step),
        "best_validation_balanced_accuracy": float(best_balanced_accuracy),
        "stop_requested": bool(stop_requested),
    }
    status_path = complete_path if complete else output / "training_incomplete.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    if not complete:
        raise SystemExit(
            "Training stopped before both requirements were met; wrote "
            f"{status_path} and did not publish training_complete.json."
        )
    print(
        f"[DONE] alias={alias} steps={step} hours={status['elapsed_training_hours']:.3f} "
        f"best_validation_balanced_accuracy={best_balanced_accuracy:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
