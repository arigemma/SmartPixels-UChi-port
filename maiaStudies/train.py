"""Train one MAIA LNN campaign point."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .campaign import get_config
from .data import MaiaDataset, deterministic_subset, load_thresholds
from .models import build_model, resource_proxies


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_metrics(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_sum = 0.0
    count = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, labels, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(criterion(logits, labels).item())
        predictions = logits.argmax(dim=1)
        for truth, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion[int(truth), int(prediction)] += 1
        count += len(labels)
    recalls = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    if was_training:
        model.train()
    return {
        "cross_entropy": loss_sum / max(count, 1),
        "balanced_accuracy": float(recalls.mean()),
        "bib_recall": float(recalls[0]),
        "signal_efficiency": float(recalls[1]),
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    thresholds: torch.Tensor,
    step: int,
    elapsed_hours: float,
    validation: dict,
) -> dict:
    return {
        "format": "maia_lnn_state_dict_v1",
        "campaign_config": config.to_dict(),
        "thresholds": thresholds.detach().cpu(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "elapsed_training_hours": float(elapsed_hours),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--minimum-steps", type=int, default=20_000)
    parser.add_argument("--minimum-training-hours", type=float, default=5.0)
    parser.add_argument("--eval-frequency", type=int, default=5_000)
    parser.add_argument("--validation-examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = get_config(args.task_index)
    config.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training_complete.json").is_file():
        print(f"[SKIP] Complete output: {args.output}")
        return
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    batch_size = args.batch_size or (1024 if config.family == "dense" else 128)
    train_dataset = MaiaDataset(args.cache_dir, "train", config.family)
    validation_dataset = deterministic_subset(
        MaiaDataset(args.cache_dir, "validation", config.family),
        args.validation_examples,
        args.seed + 1,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    thresholds = load_thresholds(args.cache_dir, config.family)
    model = build_model(config, thresholds, device).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(config.to_dict(), indent=2), flush=True)
    print(f"Train class counts: {train_dataset.class_counts()}", flush=True)
    print(f"Batch size: {batch_size}; trainable parameters: {parameter_count:,}", flush=True)

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[SIGNAL] Received {signum}; saving after current batch.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started = time.monotonic()
    step = 0
    epoch = 0
    best_balanced_accuracy = -float("inf")
    last_validation: dict[str, float] = {}
    model.train()
    while not stop_requested:
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            step += 1

            if step == 1 or step % 100 == 0:
                accuracy = float((logits.argmax(1) == labels).float().mean().item())
                print(
                    f"[TRAIN] epoch={epoch} step={step} loss={loss.item():.6f} "
                    f"batch_accuracy={accuracy:.6f}",
                    flush=True,
                )

            if step % args.eval_frequency == 0:
                last_validation = validation_metrics(model, validation_loader, device)
                elapsed = (time.monotonic() - started) / 3600.0
                print(
                    f"[VALIDATION] step={step} elapsed_hours={elapsed:.3f} "
                    f"{json.dumps(last_validation, sort_keys=True)}",
                    flush=True,
                )
                payload = checkpoint_payload(
                    model, optimizer, config, thresholds, step, elapsed, last_validation
                )
                _atomic_torch_save(args.output / "latest_model.pth", payload)
                if last_validation["balanced_accuracy"] > best_balanced_accuracy:
                    best_balanced_accuracy = last_validation["balanced_accuracy"]
                    _atomic_torch_save(args.output / "best_model.pth", payload)
                model.train()

            elapsed_hours = (time.monotonic() - started) / 3600.0
            if (
                step >= args.minimum_steps
                and elapsed_hours >= args.minimum_training_hours
            ):
                stop_requested = True
                break
            if stop_requested:
                break
        epoch += 1

    elapsed_hours = (time.monotonic() - started) / 3600.0
    if not last_validation:
        last_validation = validation_metrics(model, validation_loader, device)
        best_balanced_accuracy = last_validation["balanced_accuracy"]
    final_payload = checkpoint_payload(
        model, optimizer, config, thresholds, step, elapsed_hours, last_validation
    )
    _atomic_torch_save(args.output / "final_model.pth", final_payload)
    if not (args.output / "best_model.pth").is_file():
        _atomic_torch_save(args.output / "best_model.pth", final_payload)

    completed = step >= args.minimum_steps and elapsed_hours >= args.minimum_training_hours
    metadata = {
        "alias": config.alias,
        "task_index": config.task_index,
        "completed": completed,
        "steps": step,
        "elapsed_training_hours": elapsed_hours,
        "minimum_steps": args.minimum_steps,
        "minimum_training_hours": args.minimum_training_hours,
        "batch_size": batch_size,
        "trainable_parameters": parameter_count,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "last_validation": last_validation,
        **resource_proxies(config),
    }
    completion_name = "training_complete.json" if completed else "training_incomplete.json"
    _atomic_json(args.output / completion_name, metadata)
    if not completed:
        raise RuntimeError(
            f"Training stopped before both requirements were met: {step} steps, "
            f"{elapsed_hours:.3f} hours"
        )
    print(f"[DONE] {config.alias}: {step} steps in {elapsed_hours:.3f} hours", flush=True)


if __name__ == "__main__":
    main()
