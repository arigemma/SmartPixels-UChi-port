#!/usr/bin/env python3
"""Train one model-2 LGN and plot resource versus balanced accuracy by stage."""

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import create_dataloaders
from models.registry import REGISTRY
from utils import CreateFolder, get_device

from lgnStudies.resource_accounting import estimate_model_resources
from lgnStudies.synthesis_interface import export_dense_lgn_stage


MODEL_ALIASES = {
    "s06_p760M_tau40": "lgn-dense-2-dense-100_128_-model2lgnFull_s06_p760M_40",
    "s08_p1202M_tau40": "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40",
    "s09_p1460M_tau40": "lgn-dense-2-dense-100_128_-model2lgnFull_s09_p1460M_40",
}
CLASS_NAMES = ("positive low-pT", "negative low-pT", "high-pT")
CSV_FIELDS = (
    "model_alias",
    "registry_name",
    "step",
    "balanced_accuracy",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "n_class_0",
    "n_class_1",
    "n_class_2",
    "raw_logic_neurons",
    "reachable_binary_luts",
    "reachable_inverters",
    "simplified_logic_luts",
    "group_sum_luts",
    "total_luts",
    "final_layer_width",
    "neurons_per_class",
    "reachable_luts_by_layer",
)


def derive_thresholds_from_data(data_loader, num_bits: int) -> torch.Tensor:
    from torchlogix.layers import Binarization

    samples = []
    for batch_index, (inputs, _) in enumerate(data_loader):
        if batch_index >= 15:
            break
        samples.append(inputs)
    return Binarization.get_initial_thresholds(
        torch.cat(samples, dim=0),
        num_bits=num_bits,
        one_per="feature",
        method="distributive",
    )


def get_class_weights(data_loader) -> torch.Tensor:
    labels = data_loader.dataset.get_all_class_labels()
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError(f"Training split class counts are invalid: {counts.tolist()}.")
    return torch.as_tensor(
        len(labels) / (len(counts) * counts), dtype=torch.float32
    )


def save_model(model, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model, temporary)
    temporary.replace(path)


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


def exact_balanced_accuracy(model, loader, device: torch.device) -> dict:
    """Compute macro recall from aggregate totals over the complete loader."""
    original_mode = model.training
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in tqdm(
            loader, desc="Full test evaluation", unit="batch", leave=False
        ):
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            truth = labels.argmax(dim=1).to(torch.int64)
            bins = torch.bincount(3 * truth + predictions, minlength=9)
            confusion += bins.reshape(3, 3)

    model.train(original_mode)
    class_totals = confusion.sum(dim=1)
    if torch.any(class_totals == 0):
        raise RuntimeError(
            f"Cannot compute three-class balanced accuracy; class totals are {class_totals.tolist()}."
        )
    recalls = confusion.diag().to(torch.float64) / class_totals
    return {
        "balanced_accuracy": float(recalls.mean()),
        "recalls": [float(value) for value in recalls],
        "class_totals": [int(value) for value in class_totals],
        "confusion_matrix": confusion.tolist(),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_trajectory(rows: list[dict], output: Path, logic_only: bool = False) -> None:
    if not rows:
        return

    resource_key = "simplified_logic_luts" if logic_only else "total_luts"
    x = np.asarray([float(row[resource_key]) / 1_000 for row in rows])
    y = 100 * np.asarray([float(row["balanced_accuracy"]) for row in rows])
    steps = np.asarray([int(row["step"]) for row in rows])

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.plot(x, y, color="0.55", linewidth=1.1, zorder=1)
    points = ax.scatter(x, y, c=steps, cmap="viridis", s=58, zorder=2)
    for x_value, y_value, step in zip(x, y, steps):
        ax.annotate(
            str(step),
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    alias = rows[0]["model_alias"]
    if logic_only:
        ax.set_xlabel("Simplified discovered logic [estimated LUTs, thousands]")
        ax.set_title(f"{alias}: learned logic resource vs balanced accuracy")
    else:
        fixed = float(rows[0]["group_sum_luts"]) / 1_000
        ax.axvspan(0, fixed, color="#d9d9d9", alpha=0.55, label="fixed GroupSum")
        ax.set_xlim(left=0)
        ax.set_xlabel(
            "Total resource: simplified logic + fixed GroupSum "
            "[estimated LUTs, thousands]"
        )
        ax.set_title(f"{alias}: resource vs balanced accuracy during training")
        ax.legend(loc="best")

    ax.set_ylabel("Balanced accuracy (macro recall) [%]")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Training step")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def record_stage(
    model,
    model_alias: str,
    registry_name: str,
    step: int,
    test_loader,
    device: torch.device,
    output: Path,
    rows: list[dict],
) -> None:
    metrics = exact_balanced_accuracy(model, test_loader, device)
    resources = estimate_model_resources(model).to_dict()
    row = {
        "model_alias": model_alias,
        "registry_name": registry_name,
        "step": step,
        "balanced_accuracy": metrics["balanced_accuracy"],
        **{f"recall_class_{i}": metrics["recalls"][i] for i in range(3)},
        **{f"n_class_{i}": metrics["class_totals"][i] for i in range(3)},
        **resources,
        "reachable_luts_by_layer": json.dumps(
            resources["reachable_luts_by_layer"], separators=(",", ":")
        ),
    }
    rows.append(row)
    write_rows(output / "resource_precision.csv", rows)
    export_name = f"{model_alias}_step{step:06d}"
    export_dense_lgn_stage(
        model,
        output / "synthesis_export",
        export_name,
        model_alias=model_alias,
        registry_name=registry_name,
        training_step=step,
        balanced_accuracy=metrics["balanced_accuracy"],
    )
    plot_trajectory(rows, output / "resource_vs_balanced_accuracy.png")
    plot_trajectory(
        rows,
        output / "resource_vs_balanced_accuracy_logic_only.png",
        logic_only=True,
    )

    details = {
        "latest_step": step,
        "class_names": CLASS_NAMES,
        "class_definition": {
            "class_0": "0 <= signed pT <= 0.2 GeV",
            "class_1": "-0.2 <= signed pT < 0 GeV",
            "class_2": "|signed pT| > 0.2 GeV",
        },
        "uses_complete_configured_test_split": True,
        "confusion_matrix": metrics["confusion_matrix"],
        "resource_definition": {
            "logic": (
                "Discrete rank-2 LUTs after constant propagation, passthrough "
                "elimination, inverter folding, and dead-code elimination."
            ),
            "group_sum": (
                "Fixed balanced adder-tree LUT estimate; division by common tau "
                "is omitted because it cannot change argmax."
            ),
            "excluded": "FixedBinarization/input ADC and routing.",
        },
    }
    (output / "study_metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    print(
        f"[STAGE] step={step} balanced_acc={metrics['balanced_accuracy']:.6f} "
        f"logic_luts={resources['simplified_logic_luts']} "
        f"group_sum_luts={resources['group_sum_luts']} "
        f"total_luts={resources['total_luts']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-2 LGN resource-versus-balanced-accuracy trajectory"
    )
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--data-config", "-c", default="config/10783560/v3.yml")
    parser.add_argument("--output", "-o", action=CreateFolder, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", "-bs", type=int, default=1024)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument(
        "--minimum-training-hours",
        type=float,
        default=0.0,
        help=(
            "Continue training until both this accumulated optimizer-loop "
            "duration and --steps have been reached. Full-test evaluations "
            "and synthesis exports are excluded from this duration."
        ),
    )
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()

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
        raise RuntimeError("This study is restricted to the model-2 y-profile input.")

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
    model = model_class(thresholds=thresholds).to(device)
    class_weights = get_class_weights(train_loader).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.compile_model:
        print("Compiling the model...", flush=True)
        model.compile(dynamic=True)

    rows: list[dict] = []
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[SIGNAL] Received {signum}; stopping after the current batch.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    record_stage(
        model, args.model, registry_name, 0, test_loader, device, output, rows
    )

    step = 0
    epoch = 0
    training_started = time.monotonic()
    active_training_seconds = 0.0

    def requirements_met() -> bool:
        return training_requirements_met(
            step,
            args.steps,
            active_training_seconds,
            args.minimum_training_hours,
        )

    model.train()
    while not requirements_met() and not stop_requested:
        train_loader.dataset.shuffle_indices()
        progress = tqdm(train_loader, desc=f"Training epoch {epoch}", unit="batch")
        for inputs, labels in progress:
            if requirements_met() or stop_requested:
                break
            batch_started = time.monotonic()
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            active_training_seconds += time.monotonic() - batch_started
            if step % 20 == 0:
                progress.set_postfix(
                    step=step,
                    loss=f"{loss.item():.4f}",
                    hours=f"{active_training_seconds / 3600:.2f}",
                )

            if step % args.eval_freq == 0:
                record_stage(
                    model,
                    args.model,
                    registry_name,
                    step,
                    test_loader,
                    device,
                    output,
                    rows,
                )
                model.train()
        epoch += 1

    complete = requirements_met() and not stop_requested
    if rows[-1]["step"] != step and complete:
        record_stage(
            model, args.model, registry_name, step, test_loader, device, output, rows
        )

    elapsed_hours = active_training_seconds / 3600
    training_loop_wall_hours = (time.monotonic() - training_started) / 3600
    status = {
        "complete": complete,
        "model_alias": args.model,
        "registry_name": registry_name,
        "minimum_steps": args.steps,
        "completed_steps": step,
        "minimum_training_hours": args.minimum_training_hours,
        "elapsed_training_hours": elapsed_hours,
        "training_loop_wall_hours": training_loop_wall_hours,
        "final_recorded_step": int(rows[-1]["step"]),
        "final_balanced_accuracy": float(rows[-1]["balanced_accuracy"]),
        "stop_requested": stop_requested,
    }
    if complete:
        save_model(model, output / "final_model.pth")
        completion_path.write_text(json.dumps(status, indent=2) + "\n")
        print(
            f"[DONE] step={step} training_hours={elapsed_hours:.3f} "
            f"balanced_acc={rows[-1]['balanced_accuracy']:.6f} output={output}",
            flush=True,
        )
    else:
        save_model(model, output / "interrupted_model.pth")
        incomplete_path.write_text(json.dumps(status, indent=2) + "\n")
        raise SystemExit(
            "Training stopped before all completion requirements were met; "
            "wrote interrupted_model.pth and did not publish final_model.pth."
        )


if __name__ == "__main__":
    main()
