#!/usr/bin/env python3
"""Evaluate one frozen RTL-handoff checkpoint on the complete test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from models.rtl_handoff import RTL_HANDOFF_CONFIGS, get_rtl_handoff_config
from utils import get_device


@torch.no_grad()
def exact_test_metrics(model, loader, device: torch.device) -> dict:
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    cross_entropy = 0.0
    events = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    decoder_checked = False
    for inputs, labels in tqdm(loader, desc="Full discrete test", unit="batch"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        truth = labels.argmax(dim=1)
        prediction = logits.argmax(dim=1)
        confusion += torch.bincount(
            3 * truth.cpu() + prediction.cpu(), minlength=9
        ).reshape(3, 3)
        cross_entropy += float(criterion(logits, labels).item())
        events += int(labels.shape[0])
        if not decoder_checked:
            thresholds = model.bin.get_thresholds().to(device)
            codes = (inputs.unsqueeze(-1) > thresholds).sum(dim=-1)
            decoded = (
                torch.arange(thresholds.shape[-1], device=device)
                .reshape(1, 1, -1)
                .lt(codes.unsqueeze(-1))
                .to(dtype=inputs.dtype)
                .reshape(inputs.shape[0], -1)
            )
            direct = model.bin(inputs)
            if not torch.equal(decoded, direct):
                mismatch = int((decoded != direct).sum().item())
                raise RuntimeError(
                    "Ordinal-code decoder disagrees with FixedBinarization: "
                    f"{mismatch} bits differ."
                )
            decoder_checked = True
    totals = confusion.sum(dim=1)
    if events == 0 or torch.any(totals == 0):
        raise RuntimeError(f"Invalid test totals: events={events}, classes={totals.tolist()}.")
    recalls = confusion.diag().to(torch.float64) / totals
    return {
        "balanced_accuracy": float(recalls.mean().item()),
        "categorical_accuracy": float(confusion.diag().sum().item() / events),
        "cross_entropy": float(cross_entropy / events),
        "recalls": [float(value) for value in recalls],
        "events": events,
        "confusion_matrix": confusion.tolist(),
        "ordinal_decoder_equivalence": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--alias", choices=sorted(c["alias"] for c in RTL_HANDOFF_CONFIGS))
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-config", default="config/10783560/v3.yml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if (args.task_index is None) == (args.alias is None):
        parser.error("Specify exactly one of --task-index or --alias.")
    if args.task_index is not None:
        if not 0 <= args.task_index < len(RTL_HANDOFF_CONFIGS):
            parser.error(f"--task-index must be in [0, {len(RTL_HANDOFF_CONFIGS) - 1}].")
        alias = str(RTL_HANDOFF_CONFIGS[args.task_index]["alias"])
    else:
        alias = str(args.alias)

    output = args.result_root / alias
    completion = output / "training_complete.json"
    checkpoint = output / "best_model.importable.pth"
    if not completion.is_file():
        raise FileNotFoundError(f"Missing training completion marker: {completion}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Missing deployment checkpoint: {checkpoint}")

    config = get_rtl_handoff_config(alias)
    input_type = "y-profile" if config["family"] == "dense" else "y-profile-timing"
    _train, _validation, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=0.01,
        balance_classes=False,
        input_type=input_type,
        target_type="classification",
        drop_dominant_features=False,
    )
    device = get_device(args.device)
    model = torch.load(checkpoint, map_location=device, weights_only=False).to(device)
    if type(model.bin).__name__ != "FixedBinarization":
        raise TypeError(
            f"Evaluation requires a frozen FixedBinarization; found {type(model.bin).__name__}."
        )
    metrics = exact_test_metrics(model, test_loader, device)
    metadata = json.loads((output / "model.json").read_text())
    result = {
        "alias": alias,
        "family": config["family"],
        "checkpoint": str(checkpoint.resolve()),
        "model_config": metadata["model_kwargs"],
        **metrics,
    }
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[DONE] alias={alias} balanced_accuracy={metrics['balanced_accuracy']:.6f} "
        f"checkpoint={checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
