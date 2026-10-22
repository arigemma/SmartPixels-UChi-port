"""Reconstruct and plot MAIA training trajectories from model histories or Slurm logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .campaign import CAMPAIGNS, get_campaign_configs
from .summarize import _legacy_native_configs


SIZE_COLORS = {
    size: color
    for size, color in zip(
        ("s01", "s02", "s03", "s04", "s05"),
        ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"),
    )
}
TAU_LINESTYLES = {
    1.0: (0, (1, 1)),
    3.0: (0, (3, 1, 1, 1)),
    10.0: "-",
    20.0: "--",
    40.0: ":",
}
TAU_COLORS = {
    1.0: "#CC79A7",
    3.0: "#009E73",
    10.0: "#0072B2",
    20.0: "#E69F00",
    40.0: "#D55E00",
}
TRAIN_PATTERN = re.compile(
    r"^\[TRAIN\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"batch_accuracy=(?P<batch_accuracy>[-+0-9.eE]+)"
)
VALIDATION_PATTERN = re.compile(
    r"^\[(?P<tag>VALIDATION(?:_DISCRETE|_RELAXED)?)\]\s+"
    r"step=(?P<step>\d+)\s+elapsed_hours=(?P<elapsed_hours>[-+0-9.eE]+)\s+"
    r"(?P<metrics>\{.*\})\s*$"
)


def _configs(campaign: str):
    if campaign == "native_legacy":
        return _legacy_native_configs()
    return get_campaign_configs(campaign)


def _family_label(family: str, campaign: str) -> str:
    if campaign == "native_legacy":
        return "MAIA-native dense (37 inputs)" if family == "dense" else "MAIA-native conv (276 inputs)"
    return "Canonical dense (14 inputs)" if family == "dense" else "Canonical conv (105 inputs)"


def _parse_training_log(path: Path) -> tuple[list[dict], list[dict]]:
    training: list[dict] = []
    validation: list[dict] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            train_match = TRAIN_PATTERN.match(line)
            if train_match:
                values = train_match.groupdict()
                training.append(
                    {
                        "epoch": int(values["epoch"]),
                        "step": int(values["step"]),
                        "loss": float(values["loss"]),
                        "batch_accuracy": float(values["batch_accuracy"]),
                    }
                )
                continue
            validation_match = VALIDATION_PATTERN.match(line)
            if not validation_match:
                continue
            values = validation_match.groupdict()
            metrics = json.loads(values["metrics"])
            tag = values["tag"]
            mode = "relaxed" if tag.endswith("_RELAXED") else "discrete"
            validation.append(
                {
                    "mode": mode,
                    "step": int(values["step"]),
                    "elapsed_hours": float(values["elapsed_hours"]),
                    **metrics,
                }
            )
    return training, validation


def _read_validation_history(path: Path) -> list[dict]:
    output = []
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle, delimiter="\t"):
            for mode in ("discrete", "relaxed"):
                prefix = f"{mode}_"
                metrics = {
                    key[len(prefix):]: float(value)
                    for key, value in source.items()
                    if key.startswith(prefix) and value not in (None, "")
                }
                if not metrics:
                    continue
                output.append(
                    {
                        "mode": mode,
                        "step": int(float(source["step"])),
                        "elapsed_hours": float(source["elapsed_hours"]),
                        **metrics,
                    }
                )
    return output


def _submitted_training_job_id(result_root: Path) -> str | None:
    path = result_root / "submitted_jobs.txt"
    if not path.is_file():
        return None
    with path.open() as handle:
        for line in handle:
            key, separator, value = line.strip().partition("=")
            if separator and key == "training_job_id":
                return value.strip()
    return None


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _save(figure, output_dir: Path, stem: str, formats: tuple[str, ...], dpi: int):
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(figure)
    return paths


def _style_legends(figure, taus: list[float], *, include_sizes: bool = True) -> None:
    handles = []
    if include_sizes:
        handles.extend(
            Line2D([0], [0], color=color, linewidth=2, label=size)
            for size, color in SIZE_COLORS.items()
        )
    handles.extend(
        Line2D(
            [0], [0], color="#333333", linestyle=TAU_LINESTYLES[tau],
            linewidth=2, label=f"tau={int(tau)}",
        )
        for tau in taus
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        frameon=False,
    )


def _validation_overview(
    configs, validation_by_alias: dict[str, list[dict]], campaign: str
) -> plt.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), sharex="row")
    metrics = (
        ("balanced_accuracy", "Validation balanced accuracy"),
        ("roc_auc", "Validation ROC AUC"),
        ("cross_entropy", "Validation cross entropy"),
    )
    for row_index, family in enumerate(("dense", "conv")):
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for config in configs:
                if config.family != family:
                    continue
                history = [
                    row for row in validation_by_alias.get(config.alias, [])
                    if row["mode"] == "discrete" and metric in row
                ]
                if not history:
                    continue
                axis.plot(
                    [row["elapsed_hours"] for row in history],
                    [row[metric] for row in history],
                    color=SIZE_COLORS[config.size],
                    linestyle=TAU_LINESTYLES[float(config.tau)],
                    linewidth=1.35,
                    alpha=0.88,
                )
            axis.set_title(f"{_family_label(family, campaign)} — {label}")
            axis.set_ylabel(label)
            axis.set_xlabel("Elapsed training time [hours]")
            axis.grid(True, alpha=0.22)
    taus = sorted({float(config.tau) for config in configs})
    _style_legends(figure, taus)
    figure.suptitle("MAIA validation trajectories — all models", y=1.015)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _balanced_accuracy_grid(
    configs, validation_by_alias: dict[str, list[dict]], campaign: str
) -> plt.Figure:
    figure, axes = plt.subplots(2, 5, figsize=(20.0, 8.0), sharex="row", sharey="row")
    for row_index, family in enumerate(("dense", "conv")):
        for column_index, size in enumerate(SIZE_COLORS):
            axis = axes[row_index, column_index]
            for config in configs:
                if config.family != family or config.size != size:
                    continue
                history = [
                    row for row in validation_by_alias.get(config.alias, [])
                    if row["mode"] == "discrete" and "balanced_accuracy" in row
                ]
                if not history:
                    continue
                tau = float(config.tau)
                axis.plot(
                    [row["elapsed_hours"] for row in history],
                    [row["balanced_accuracy"] for row in history],
                    color=TAU_COLORS[tau],
                    linewidth=1.55,
                    label=f"tau={int(tau)}",
                )
            axis.set_title(f"{_family_label(family, campaign)} {size}")
            axis.set_xlabel("Training time [hours]")
            if column_index == 0:
                axis.set_ylabel("Validation balanced accuracy")
            axis.grid(True, alpha=0.22)
    taus = sorted({float(config.tau) for config in configs})
    handles = [
        Line2D([0], [0], color=TAU_COLORS[tau], linewidth=2, label=f"tau={int(tau)}")
        for tau in taus
    ]
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    figure.suptitle("Validation balanced accuracy by architecture", y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _smoothed_xy(rows: list[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    if len(y) > 5:
        window = min(51, max(5, (len(y) // 120) | 1))
        kernel = np.ones(window, dtype=float) / window
        y = np.convolve(y, kernel, mode="same")
        edge = window // 2
        if edge:
            x, y = x[edge:-edge], y[edge:-edge]
    if len(x) > 900:
        indices = np.unique(np.linspace(0, len(x) - 1, 900, dtype=int))
        x, y = x[indices], y[indices]
    return x, y


def _batch_overview(configs, training_by_alias: dict[str, list[dict]], campaign: str):
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex="row")
    for row_index, family in enumerate(("dense", "conv")):
        for column_index, (metric, label) in enumerate(
            (("loss", "Training batch loss"), ("batch_accuracy", "Training batch accuracy"))
        ):
            axis = axes[row_index, column_index]
            for config in configs:
                history = training_by_alias.get(config.alias, [])
                if config.family != family or not history:
                    continue
                x, y = _smoothed_xy(history, "step", metric)
                axis.plot(
                    x, y,
                    color=SIZE_COLORS[config.size],
                    linestyle=TAU_LINESTYLES[float(config.tau)],
                    linewidth=1.1,
                    alpha=0.82,
                )
            axis.set_title(f"{_family_label(family, campaign)} — {label}")
            axis.set_xlabel("Training step")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.22)
    taus = sorted({float(config.tau) for config in configs})
    _style_legends(figure, taus)
    figure.suptitle("MAIA smoothed batch-training trajectories — all models", y=1.015)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _paired_family_figure(
    configs,
    validation_by_alias: dict[str, list[dict]],
    family: str,
    campaign: str,
) -> plt.Figure:
    """Plot like-for-like relaxed/discrete validation metrics for one family."""
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex=True)
    panels = (
        ("relaxed", "balanced_accuracy", "Relaxed validation balanced accuracy"),
        ("discrete", "balanced_accuracy", "Discrete validation balanced accuracy"),
        ("relaxed", "cross_entropy", "Relaxed validation cross entropy"),
        ("discrete", "cross_entropy", "Discrete validation cross entropy"),
    )
    for axis, (mode, metric, label) in zip(axes.flat, panels):
        for config in configs:
            if config.family != family:
                continue
            history = [
                row for row in validation_by_alias.get(config.alias, [])
                if row["mode"] == mode and metric in row
            ]
            if not history:
                continue
            axis.plot(
                [row["step"] for row in history],
                [row[metric] for row in history],
                color=SIZE_COLORS[config.size],
                linestyle=TAU_LINESTYLES[float(config.tau)],
                linewidth=1.35,
                alpha=0.88,
            )
        axis.set_title(label)
        axis.set_xlabel("Training step")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.22)
    taus = sorted({float(config.tau) for config in configs if config.family == family})
    _style_legends(figure, taus)
    figure.suptitle(
        f"{_family_label(family, campaign)} — paired relaxed/discrete validation",
        y=1.015,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _missing_paired_histories(configs, validation_by_alias: dict[str, list[dict]]):
    missing = []
    for config in configs:
        modes = {
            row["mode"]
            for row in validation_by_alias.get(config.alias, [])
            if "balanced_accuracy" in row and "cross_entropy" in row
        }
        if modes != {"relaxed", "discrete"}:
            missing.append(config.alias)
    return missing


def generate_training_curves(
    result_root: Path,
    *,
    logs_dir: Path,
    campaign: str,
    training_job_id: str | None = None,
    output_dir: Path | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 400,
    allow_partial: bool = False,
    require_paired: bool = False,
) -> dict:
    configs = _configs(campaign)
    output_dir = output_dir or result_root / "summary_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_job_id = training_job_id or _submitted_training_job_id(result_root)
    training_by_alias: dict[str, list[dict]] = {}
    validation_by_alias: dict[str, list[dict]] = {}
    missing = []
    sources = {}
    for config in configs:
        log_path = (
            logs_dir / f"maiaLnnTrain_{training_job_id}_{config.task_index}.out"
            if training_job_id
            else None
        )
        history_path = result_root / config.alias / "training_history.tsv"
        training_rows: list[dict] = []
        validation_rows: list[dict] = []
        if log_path is not None and log_path.is_file():
            training_rows, validation_rows = _parse_training_log(log_path)
            sources[config.alias] = str(log_path)
        elif history_path.is_file():
            validation_rows = _read_validation_history(history_path)
            sources[config.alias] = str(history_path)
        if not validation_rows:
            missing.append(config.alias)
            continue
        training_by_alias[config.alias] = training_rows
        validation_by_alias[config.alias] = validation_rows

    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing training trajectories for {len(missing)} of {len(configs)} models: "
            + ", ".join(missing)
        )
    if not validation_by_alias:
        raise FileNotFoundError(
            f"No training histories found in {result_root} or {logs_dir}"
        )
    missing_paired = _missing_paired_histories(configs, validation_by_alias)
    if missing_paired and require_paired:
        raise RuntimeError(
            "Paired relaxed/discrete validation histories are unavailable for "
            f"{len(missing_paired)} of {len(configs)} models: "
            + ", ".join(missing_paired)
            + ". These modes must be evaluated on the same validation sample "
            "during training; relaxed batch loss is not a substitute."
        )

    validation_table = []
    batch_table = []
    config_by_alias = {config.alias: config for config in configs}
    for alias, rows in validation_by_alias.items():
        config = config_by_alias[alias]
        validation_table.extend(
            {
                "alias": alias,
                "family": config.family,
                "size": config.size,
                "tau": config.tau,
                **row,
            }
            for row in rows
        )
    for alias, rows in training_by_alias.items():
        config = config_by_alias[alias]
        batch_table.extend(
            {
                "alias": alias,
                "family": config.family,
                "size": config.size,
                "tau": config.tau,
                **row,
            }
            for row in rows
        )
    validation_table_path = result_root / "validation_training_history.tsv"
    batch_table_path = result_root / "batch_training_history.tsv"
    _write_table(validation_table_path, validation_table)
    _write_table(batch_table_path, batch_table)

    products = []
    products.extend(
        _save(
            _validation_overview(configs, validation_by_alias, campaign),
            output_dir,
            "20_validation_training_curves_all_models",
            formats,
            dpi,
        )
    )
    products.extend(
        _save(
            _balanced_accuracy_grid(configs, validation_by_alias, campaign),
            output_dir,
            "21_validation_balanced_accuracy_by_architecture",
            formats,
            dpi,
        )
    )
    if any(training_by_alias.values()):
        products.extend(
            _save(
                _batch_overview(configs, training_by_alias, campaign),
                output_dir,
                "22_batch_training_curves_all_models",
                formats,
                dpi,
            )
        )
    if not missing_paired:
        for family, stem in (
            ("dense", "23_dense_relaxed_discrete_validation_curves"),
            ("conv", "24_conv_relaxed_discrete_validation_curves"),
        ):
            products.extend(
                _save(
                    _paired_family_figure(
                        configs, validation_by_alias, family, campaign
                    ),
                    output_dir,
                    stem,
                    formats,
                    dpi,
                )
            )
    manifest = {
        "campaign": campaign,
        "models_found": len(validation_by_alias),
        "models_expected": len(configs),
        "training_job_id": training_job_id,
        "missing_models": missing,
        "missing_paired_histories": missing_paired,
        "paired_models_found": len(configs) - len(missing_paired),
        "sources": sources,
        "validation_table": str(validation_table_path),
        "batch_table": str(batch_table_path) if batch_table else None,
        "plots": products,
    }
    with (result_root / "training_curves_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--campaign",
        choices=tuple(CAMPAIGNS) + ("native_legacy",),
        default="native_legacy",
    )
    parser.add_argument("--training-job-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-paired", action="store_true")
    args = parser.parse_args()
    result = generate_training_curves(
        args.result_root,
        logs_dir=args.logs_dir,
        campaign=args.campaign,
        training_job_id=args.training_job_id,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
        allow_partial=args.allow_partial,
        require_paired=args.require_paired,
    )
    print(
        f"[DONE] Training histories: "
        f"{result['models_found']}/{result['models_expected']}"
    )
    print(f"[DONE] Validation table: {result['validation_table']}")
    if result["batch_table"]:
        print(f"[DONE] Batch table:      {result['batch_table']}")
    for path in result["plots"]:
        print(f"[DONE] Plot:             {path}")


if __name__ == "__main__":
    main()
