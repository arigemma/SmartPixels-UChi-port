#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---- What we will plot (TensorBoard scalar tags) ----
TAGS = {
    "balanced_acc_discrete": "Discrete/balanced_acc",
    "balanced_acc_relaxed":  "Relaxed/balanced_acc",
    "roc_auc_discrete":       "Discrete/roc_auc",
    "roc_auc_relaxed":        "Relaxed/roc_auc",
}


@dataclass
class RunSpec:
    label: str
    root: Path  # directory that contains logs/**/events.out.tfevents*


def _parse_run(run_arg: str) -> RunSpec:
    if "=" not in run_arg:
        raise ValueError(f'Invalid --run "{run_arg}". Use "label=PATH".')
    label, path = run_arg.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f'Invalid --run "{run_arg}": empty label.')
    root = Path(path).expanduser()
    return RunSpec(label=label, root=root)


def _find_event_dirs(root: Path) -> List[Path]:

    if not root.exists():
        return []
    event_files = list(root.rglob("events.out.tfevents.*"))
    dirs = sorted({p.parent for p in event_files})
    return dirs


def _load_scalars_from_event_dir(event_dir: Path, size_guidance: int = 0) -> Dict[str, List[Tuple[int, float]]]:

    ea = EventAccumulator(
        str(event_dir),
        size_guidance={
            "scalars": size_guidance,  
        },
    )
    ea.Reload()

    out: Dict[str, List[Tuple[int, float]]] = {}
    scalar_tags = ea.Tags().get("scalars", [])
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def _merge_series(series_list: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:

    last_by_step: Dict[int, float] = {}
    for series in series_list:
        for step, val in series:
            last_by_step[step] = val
    merged = sorted(last_by_step.items(), key=lambda x: x[0])
    return merged


def _cap_steps(series: List[Tuple[int, float]], max_step: Optional[int]) -> List[Tuple[int, float]]:
    if max_step is None:
        return series
    return [(s, v) for (s, v) in series if s <= max_step]


def _smooth(series: List[Tuple[int, float]], window: int) -> List[Tuple[int, float]]:

    if window <= 1 or len(series) < 3:
        return series
    steps = [s for s, _ in series]
    vals = [v for _, v in series]
    smoothed = []
    half = window // 2
    n = len(vals)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(vals[lo:hi]) / (hi - lo))
    return list(zip(steps, smoothed))


def _plot_metric(
    runs_data: Dict[str, List[Tuple[int, float]]],
    title: str,
    ylabel: str,
    out_path: Path,
    max_step: Optional[int],
    smooth_window: int,
):
    plt.figure()
    for label, series in runs_data.items():
        series = _cap_steps(series, max_step)
        series = _smooth(series, smooth_window)
        if not series:
            continue
        xs = [s for s, _ in series]
        ys = [v for _, v in series]
        plt.plot(xs, ys, label=label)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot TensorBoard scalars across multiple runs.")
    ap.add_argument(
        "-r", "--run",
        action="append",
        required=True,
        help='Repeatable. Format: "label=PATH". PATH is your training output folder (the one you passed as -o to train.py).',
    )
    ap.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only plot points with step <= max-step. (default: no cap)",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average smoothing window (odd integers work best). 1 disables smoothing.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="tb_plots",
        help="Where to write PNGs (default: tb_plots).",
    )
    ap.add_argument(
        "--prefer-latest",
        action="store_true",
        help="If a run directory contains multiple event subdirs, only use the most recent one (lexicographically).",
    )
    args = ap.parse_args()

    runs = [_parse_run(r) for r in args.run]
    outdir = Path(args.outdir).expanduser()

    per_run_series: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}  
    missing_runs = []
    for run in runs:
        event_dirs = _find_event_dirs(run.root)
        if not event_dirs:
            missing_runs.append(f"{run.label} ({run.root})")
            continue

        if args.prefer_latest:
            event_dirs = [event_dirs[-1]]

        tag_to_series_list: Dict[str, List[List[Tuple[int, float]]]] = {}
        for ed in event_dirs:
            scalars = _load_scalars_from_event_dir(ed, size_guidance=0)
            for tag, series in scalars.items():
                tag_to_series_list.setdefault(tag, []).append(series)

        merged: Dict[str, List[Tuple[int, float]]] = {}
        for tag, series_list in tag_to_series_list.items():
            merged[tag] = _merge_series(series_list)

        per_run_series[run.label] = merged

    if missing_runs:
        print("WARNING: no TensorBoard event files found for:")
        for s in missing_runs:
            print(f"  - {s}")

    if not per_run_series:
        raise SystemExit("No valid runs found (no event files).")

    def collect(tag: str) -> Dict[str, List[Tuple[int, float]]]:
        out: Dict[str, List[Tuple[int, float]]] = {}
        for label, tagmap in per_run_series.items():
            if tag in tagmap:
                out[label] = tagmap[tag]
        return out

    plots = [
        ("Balanced Accuracy (Discrete)", TAGS["balanced_acc_discrete"], "balanced_acc", outdir / "balanced_acc_discrete.png"),
        ("Balanced Accuracy (Relaxed)",  TAGS["balanced_acc_relaxed"],  "balanced_acc", outdir / "balanced_acc_relaxed.png"),
        ("ROC AUC (Discrete)",           TAGS["roc_auc_discrete"],      "roc_auc",       outdir / "roc_auc_discrete.png"),
        ("ROC AUC (Relaxed)",            TAGS["roc_auc_relaxed"],       "roc_auc",       outdir / "roc_auc_relaxed.png"),
    ]

    for title, tag, ylabel, path in plots:
        data = collect(tag)
        if not data:
            print(f"WARNING: tag not found in any run: {tag}")
            continue
        _plot_metric(
            runs_data=data,
            title=title,
            ylabel=ylabel,
            out_path=path,
            max_step=args.max_step,
            smooth_window=args.smooth,
        )
        print(f"Wrote: {path}")

    # print("\nPer-run tag availability:")
    # for label, tagmap in per_run_series.items():
    #     available = set(tagmap.keys())
    #     wanted = [TAGS[k] for k in TAGS]
    #     have = [t for t in wanted if t in available]
    #     miss = [t for t in wanted if t not in available]
    #     print(f"- {label}:")
    #     print(f"    have: {have}")
    #     if miss:
    #         print(f"    missing: {miss}")


if __name__ == "__main__":
    main()
