"""Train one MAIA LNN campaign point."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import roc_auc_score

from .campaign import get_config
from .checkpointing import (
    CHECKPOINT_FORMAT,
    capture_fixed_connections,
    load_model_checkpoint_state,
    snapshot_to_cpu as _snapshot_to_cpu,
)
from .data import INPUT_SCHEMA, MaiaDataset, deterministic_subset, load_thresholds
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
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    discrete: bool = True,
) -> dict[str, float]:
    was_training = model.training
    if discrete:
        model.eval()
    else:
        model.train()
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_sum = 0.0
    count = 0
    all_scores = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, labels, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(criterion(logits, labels).item())
        predictions = logits.argmax(dim=1)
        all_scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        for truth, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion[int(truth), int(prediction)] += 1
        count += len(labels)
    recalls = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    scores = np.concatenate(all_scores)
    truth = np.concatenate(all_labels)
    model.train(was_training)
    return {
        "cross_entropy": loss_sum / max(count, 1),
        "balanced_accuracy": float(recalls.mean()),
        "bib_recall": float(recalls[0]),
        "signal_efficiency": float(recalls[1]),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "prediction_signal_fraction": float(np.mean(scores >= 0.0)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }


def _append_validation_history(
    path: Path,
    step: int,
    elapsed_hours: float,
    discrete: dict[str, float],
    relaxed: dict[str, float] | None,
) -> None:
    metric_names = tuple(discrete)
    fieldnames = ["step", "elapsed_hours"]
    fieldnames.extend(f"discrete_{name}" for name in metric_names)
    fieldnames.extend(f"relaxed_{name}" for name in metric_names)
    row = {"step": int(step), "elapsed_hours": float(elapsed_hours)}
    row.update({f"discrete_{name}": discrete[name] for name in metric_names})
    row.update(
        {
            f"relaxed_{name}": "" if relaxed is None else relaxed[name]
            for name in metric_names
        }
    )
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    thresholds: dict[str, torch.Tensor],
    step: int,
    elapsed_hours: float,
    validation: dict,
    seed: int,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "input_schema": INPUT_SCHEMA,
        "campaign_config": config.to_dict(),
        "model_initialization_seed": int(seed),
        "thresholds": _snapshot_to_cpu(thresholds),
        "model_state_dict": _snapshot_to_cpu(model.state_dict()),
        "fixed_connection_state": capture_fixed_connections(model),
        "optimizer_state_dict": _snapshot_to_cpu(optimizer.state_dict()),
        "step": int(step),
        "elapsed_training_hours": float(elapsed_hours),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--campaign", default="main")
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
    parser.add_argument("--track-relaxed-validation", action="store_true")
    args = parser.parse_args()

    config = get_config(args.task_index, args.campaign)
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
                relaxed_validation = (
                    validation_metrics(
                        model, validation_loader, device, discrete=False
                    )
                    if args.track_relaxed_validation
                    else None
                )
                elapsed = (time.monotonic() - started) / 3600.0
                print(
                    f"[VALIDATION_DISCRETE] step={step} elapsed_hours={elapsed:.3f} "
                    f"{json.dumps(last_validation, sort_keys=True)}",
                    flush=True,
                )
                if relaxed_validation is not None:
                    print(
                        f"[VALIDATION_RELAXED] step={step} "
                        f"elapsed_hours={elapsed:.3f} "
                        f"{json.dumps(relaxed_validation, sort_keys=True)}",
                        flush=True,
                    )
                _append_validation_history(
                    args.output / "training_history.tsv",
                    step,
                    elapsed,
                    last_validation,
                    relaxed_validation,
                )
                payload = checkpoint_payload(
                    model, optimizer, config, thresholds, step, elapsed,
                    last_validation, args.seed
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
        model, optimizer, config, thresholds, step, elapsed_hours,
        last_validation, args.seed
    )
    _atomic_torch_save(args.output / "final_model.pth", final_payload)
    if not (args.output / "best_model.pth").is_file():
        _atomic_torch_save(args.output / "best_model.pth", final_payload)

    # A checkpoint is not complete merely because torch.load succeeds. Rebuild
    # a fresh model, load the selected checkpoint, and require it to reproduce
    # the exact validation subset and metrics used during training.
    best_checkpoint = torch.load(
        args.output / "best_model.pth", map_location="cpu", weights_only=False
    )
    reloaded_thresholds = {
        name: values.float() for name, values in best_checkpoint["thresholds"].items()
    }
    # Re-seeding is an additional safeguard for TorchLogix releases that
    # initialize fixed connections from global random-number generators.
    _seed_everything(int(best_checkpoint["model_initialization_seed"]))
    reloaded_model = build_model(config, reloaded_thresholds, device).to(device)
    load_model_checkpoint_state(reloaded_model, best_checkpoint)
    roundtrip_validation = validation_metrics(
        reloaded_model, validation_loader, device
    )
    stored_validation = best_checkpoint["validation"]
    roundtrip_deltas = {
        metric: abs(float(roundtrip_validation[metric]) - float(stored_validation[metric]))
        for metric in ("balanced_accuracy", "roc_auc", "cross_entropy")
    }
    if (
        roundtrip_deltas["balanced_accuracy"] > 1e-6
        or roundtrip_deltas["roc_auc"] > 1e-6
        or roundtrip_deltas["cross_entropy"] > 1e-5
    ):
        raise RuntimeError(
            "Saved best checkpoint failed exact round-trip validation: "
            f"stored={stored_validation}, reloaded={roundtrip_validation}, "
            f"deltas={roundtrip_deltas}"
        )

    completed = step >= args.minimum_steps and elapsed_hours >= args.minimum_training_hours
    metadata = {
        "alias": config.alias,
        "task_index": config.task_index,
        "completed": completed,
        "steps": step,
        "elapsed_training_hours": elapsed_hours,
        "minimum_steps": args.minimum_steps,
        "minimum_training_hours": args.minimum_training_hours,
        "campaign": args.campaign,
        "tracked_relaxed_validation": args.track_relaxed_validation,
        "batch_size": batch_size,
        "trainable_parameters": parameter_count,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "last_validation": last_validation,
        "checkpoint_roundtrip_validation": {
            "passed": True,
            "checkpoint_step": int(best_checkpoint["step"]),
            "stored": stored_validation,
            "reloaded": roundtrip_validation,
            "absolute_deltas": roundtrip_deltas,
        },
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
