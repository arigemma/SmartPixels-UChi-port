#!/usr/bin/env python3
"""Measure model-2 LGN robustness to an exact number of inference faults."""

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

from lgnStudies.bitFlipStudy.fault_injection import ExactCountBitFlipModel
from lgnStudies.bitFlipStudy.plots import plot_single_model


DEFAULT_FLIP_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
CSV_FIELDS = (
    "model_alias",
    "registry_name",
    "flip_count_per_event",
    "total_fault_sites_per_event",
    "equivalent_fault_fraction",
    "equivalent_fault_percentage",
    "repeat",
    "fault_seed",
    "balanced_accuracy",
    "prediction_change_fraction",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "n_class_0",
    "n_class_1",
    "n_class_2",
    *(f"confusion_{truth}_{prediction}" for truth in range(3) for prediction in range(3)),
)


def validate_model(model, expected_registry_name: str) -> None:
    registry_name, config = class_to_config(type(model))
    if registry_name != expected_registry_name:
        raise ValueError(
            f"Checkpoint contains {registry_name}, expected {expected_registry_name}."
        )
    if config["input_type"] != "y-profile":
        raise ValueError("Bit-flip study is restricted to model-2 y-profile LGNs.")
    if tuple(getattr(model, "hidden_dims", ())) not in (
        (13800, 13800, 13800, 9200, 9200, 9200, 6900),
        (17400, 17400, 17400, 11600, 11600, 11600, 8700),
        (19200, 19200, 19200, 12800, 12800, 12800, 9600),
    ):
        raise ValueError("Checkpoint is not one of the three selected model-2 LGNs.")


def set_fault_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_complete_test_split(
    model: ExactCountBitFlipModel,
    loader,
    device: torch.device,
    reference_predictions: np.ndarray | None = None,
    collect_predictions: bool = False,
) -> tuple[dict, np.ndarray | None]:
    """Compute one aggregate confusion matrix over the configured test split."""
    original_mode = model.training
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    predictions_for_output: list[np.ndarray] = []
    changed = 0
    number_seen = 0

    with torch.no_grad():
        for inputs, labels in tqdm(
            loader, desc="Full test evaluation", unit="batch", leave=False
        ):
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            truth = labels.argmax(dim=1).to(torch.int64)
            confusion += torch.bincount(
                3 * truth + predictions, minlength=9
            ).reshape(3, 3)

            prediction_array = predictions.numpy().astype(np.int8, copy=False)
            if reference_predictions is not None:
                reference_batch = reference_predictions[
                    number_seen : number_seen + len(prediction_array)
                ]
                if len(reference_batch) != len(prediction_array):
                    raise RuntimeError("Reference predictions do not match test split.")
                changed += int(np.count_nonzero(prediction_array != reference_batch))
            if collect_predictions:
                predictions_for_output.append(prediction_array.copy())
            number_seen += len(prediction_array)

    model.train(original_mode)
    if reference_predictions is not None and number_seen != len(reference_predictions):
        raise RuntimeError("Reference predictions do not match test split length.")

    totals = confusion.sum(dim=1)
    if torch.any(totals == 0):
        raise RuntimeError(f"Missing test class; totals={totals.tolist()}.")
    recalls = confusion.diag().to(torch.float64) / totals
    metrics = {
        "balanced_accuracy": float(recalls.mean()),
        "prediction_change_fraction": (
            float(changed / number_seen)
            if reference_predictions is not None
            else 0.0
        ),
        "recalls": [float(value) for value in recalls],
        "class_totals": [int(value) for value in totals],
        "confusion_matrix": confusion.tolist(),
        "number_test_events": number_seen,
    }
    collected = (
        np.concatenate(predictions_for_output)
        if collect_predictions
        else None
    )
    return metrics, collected


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
        **{f"recall_class_{index}": metrics["recalls"][index] for index in range(3)},
        **{f"n_class_{index}": metrics["class_totals"][index] for index in range(3)},
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
            "Model-2 LGN robustness to exactly K random transient activation "
            "bit flips per test event"
        )
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--registry-name", required=True)
    parser.add_argument("--clean-model", type=Path, required=True)
    parser.add_argument("-c", "--data-config", default="config/10783560/v3.yml")
    parser.add_argument("-o", "--output", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024)
    parser.add_argument(
        "--flip-counts",
        type=int,
        nargs="+",
        default=DEFAULT_FLIP_COUNTS,
        help="Exact unique transient activation-bit flips per test event.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent complete-test fault realizations for each nonzero count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    flip_counts = sorted(set(args.flip_counts))
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to estimate uncertainty.")
    if not flip_counts or flip_counts[0] < 0:
        parser.error("--flip-counts must contain non-negative integers.")
    if 0 not in flip_counts:
        flip_counts.insert(0, 0)
    if not args.clean_model.is_file():
        parser.error(f"Clean checkpoint does not exist: {args.clean_model}")

    set_fault_seed(args.seed)
    device = get_device(args.device)
    output = Path(args.output)

    _, _, test_loader = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=False,
        val_size=0.0,
        balance_classes=False,
        input_type="y-profile",
        target_type="classification",
        drop_dominant_features=False,
    )

    clean_model = load_registered_model(
        str(args.clean_model), map_location=device
    ).to(device)
    validate_model(clean_model, args.registry_name)
    fault_model = ExactCountBitFlipModel(clean_model).to(device)
    if flip_counts[-1] > fault_model.total_fault_sites:
        parser.error(
            f"Requested {flip_counts[-1]} flips, but this model has only "
            f"{fault_model.total_fault_sites} eligible sites per event."
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

    fault_model.set_flip_count(0)
    set_fault_seed(args.seed)
    clean_metrics, clean_predictions = evaluate_complete_test_split(
        fault_model,
        test_loader,
        device,
        collect_predictions=True,
    )
    if clean_predictions is None:
        raise RuntimeError("Clean reference predictions were not collected.")
    np.save(output / "clean_predictions.npy", clean_predictions)
    rows.append(
        make_result_row(
            args.model_alias,
            args.registry_name,
            0,
            fault_model.total_fault_sites,
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
        fault_model.set_flip_count(flip_count)
        for repeat in range(args.repeats):
            if stop:
                break
            fault_seed = args.seed + 1_000_003 * flip_count + repeat
            set_fault_seed(fault_seed)
            metrics, _ = evaluate_complete_test_split(
                fault_model,
                test_loader,
                device,
                reference_predictions=clean_predictions,
            )
            rows.append(
                make_result_row(
                    args.model_alias,
                    args.registry_name,
                    flip_count,
                    fault_model.total_fault_sites,
                    repeat,
                    fault_seed,
                    metrics,
                )
            )
            latest_confusions[
                f"K={flip_count}:repeat={repeat}"
            ] = metrics["confusion_matrix"]
            write_rows(output / "bit_flip_results.csv", rows)
            plot_single_model(
                rows,
                args.model_alias,
                output / "bit_flips_per_event",
                x_mode="count",
            )
            plot_single_model(
                rows,
                args.model_alias,
                output / "bit_flip_percentage",
                x_mode="percentage",
            )
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
        "flip_counts_per_event": flip_counts,
        "total_fault_sites_per_event": fault_model.total_fault_sites,
        "stage_widths": fault_model.stage_widths,
        "stage_offsets": fault_model.stage_offsets,
        "repeats_per_nonzero_count": args.repeats,
        "number_test_events": clean_metrics["number_test_events"],
        "uses_complete_configured_test_split": True,
        "balanced_accuracy": "Unweighted mean of all three aggregate class recalls.",
        "prediction_change_fraction": (
            "Fraction of test events whose predicted class differs from the "
            "same event with K=0."
        ),
        "fault_model": (
            "For every test event, uniformly sample exactly K distinct sites "
            "without replacement from the concatenated Boolean outputs of "
            "FixedBinarization and every LogicDense layer, then apply x -> 1-x. "
            "Faults are transient and resampled on each forward pass. GroupSum "
            "and all model parameters are unchanged."
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
