"""Evaluate a trained MAIA model before and after the MAIA timing cut."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .campaign import get_config
from .data import MaiaDataset
from .metrics import evaluate_binary_scores, roc_rows
from .models import build_model, resource_proxies


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


@torch.no_grad()
def collect_scores(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    scores, labels, timing = [], [], []
    model.eval()
    for batch_number, (inputs, truth, times) in enumerate(loader, 1):
        logits = model(inputs.to(device, non_blocking=True))
        scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        labels.append(truth.numpy())
        timing.append(times.numpy())
        if batch_number % 100 == 0:
            print(f"[EVAL] {batch_number}/{len(loader)} batches", flush=True)
    return (
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(timing),
    )


def _finite_for_json(value):
    if isinstance(value, dict):
        return {key: _finite_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_for_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_roc(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timing-min-ns", type=float, default=-0.5)
    parser.add_argument("--timing-max-ns", type=float, default=15.0)
    args = parser.parse_args()

    config = get_config(args.task_index)
    completion = args.model_dir / "training_complete.json"
    checkpoint_path = args.model_dir / "best_model.pth"
    if not completion.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Training is incomplete for {config.alias}: expected {completion} and {checkpoint_path}"
        )
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "maia_lnn_state_dict_v1":
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    if checkpoint["campaign_config"]["alias"] != config.alias:
        raise ValueError("Checkpoint/config alias mismatch")
    thresholds = checkpoint["thresholds"].float()
    model = build_model(config, thresholds, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    batch_size = args.batch_size or (2048 if config.family == "dense" else 256)
    validation = MaiaDataset(args.cache_dir, "validation", config.family)
    test = MaiaDataset(args.cache_dir, "test", config.family)
    val_scores, val_labels, val_timing = collect_scores(
        model, validation, batch_size, args.num_workers, device
    )
    test_scores, test_labels, test_timing = collect_scores(
        model, test, batch_size, args.num_workers, device
    )

    full = evaluate_binary_scores(val_labels, val_scores, test_labels, test_scores)
    val_cut = (val_timing >= args.timing_min_ns) & (val_timing <= args.timing_max_ns)
    test_cut = (test_timing >= args.timing_min_ns) & (test_timing <= args.timing_max_ns)
    timing_cut = evaluate_binary_scores(
        val_labels[val_cut],
        val_scores[val_cut],
        test_labels[test_cut],
        test_scores[test_cut],
    )
    with (args.cache_dir / "preparation_complete.json").open() as handle:
        preparation_metadata = json.load(handle)
    result = {
        "alias": config.alias,
        "task_index": config.task_index,
        "family": config.family,
        "size": config.size,
        "tau": config.tau,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "trainable_parameters": int(trainable_parameters),
        "class_definition": {"0": "BIB (bib_mp + bib_mm)", "1": "signal/PV"},
        "score": "class-1 GroupSum logit minus class-0 GroupSum logit",
        "full_test": full,
        "timing_cut_test": timing_cut,
        "timing_cut": {
            "column": preparation_metadata["timing_column"],
            "minimum_ns_inclusive": args.timing_min_ns,
            "maximum_ns_inclusive": args.timing_max_ns,
            "used_for_training": False,
            "validation_rows_kept": int(val_cut.sum()),
            "test_rows_kept": int(test_cut.sum()),
            "validation_signal_acceptance": float(
                np.count_nonzero(val_cut & (val_labels == 1))
                / np.count_nonzero(val_labels == 1)
            ),
            "validation_bib_acceptance": float(
                np.count_nonzero(val_cut & (val_labels == 0))
                / np.count_nonzero(val_labels == 0)
            ),
            "test_signal_acceptance": float(
                np.count_nonzero(test_cut & (test_labels == 1))
                / np.count_nonzero(test_labels == 1)
            ),
            "test_bib_acceptance": float(
                np.count_nonzero(test_cut & (test_labels == 0))
                / np.count_nonzero(test_labels == 0)
            ),
        },
        **resource_proxies(config),
    }
    result = _finite_for_json(result)
    _atomic_json(args.model_dir / "evaluation_complete.json", result)
    _write_roc(args.model_dir / "roc_full.csv", roc_rows(test_labels, test_scores))
    _write_roc(
        args.model_dir / "roc_timing_cut.csv",
        roc_rows(test_labels[test_cut], test_scores[test_cut]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
