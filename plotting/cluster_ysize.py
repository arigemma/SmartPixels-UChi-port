#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def load_yaml(p: Path) -> dict:
    with p.open("r") as f:
        return yaml.safe_load(f)


def iter_recon_files(cfg: dict) -> Iterable[Path]:
    root = Path(cfg["root"])
    for pair in cfg.get("train", []):
        if isinstance(pair, list) and len(pair) >= 1:
            yield root / pair[0]


def read_recon_tyx(recon_path: Path, max_events: Optional[int]) -> np.ndarray:
    cols = [str(i) for i in range(20 * 13 * 21)]
    df = pd.read_parquet(recon_path, engine="pyarrow", columns=cols)
    if max_events is not None:
        df = df.iloc[:max_events]
    flat = df.to_numpy(dtype=np.float32, copy=False)
    if flat.shape[1] != 20 * 13 * 21:
        raise ValueError(f"Unexpected recon width {flat.shape[1]} in {recon_path}")
    return flat.reshape(flat.shape[0], 20, 13, 21)


def yprofile_from_tyx(tyx: np.ndarray, slice_index: int, use_abs: bool) -> np.ndarray:
    xy = tyx[:, slice_index, :, :]
    if use_abs:
        xy = np.abs(xy)
    return xy.sum(axis=2)



def ysize_from_yprofile(yprof: np.ndarray, thr_abs: Optional[float], thr_frac: float) -> np.ndarray:
    if thr_abs is not None:
        thr = float(thr_abs)
        mask = yprof > thr
    else:
        vmax = yprof.max(axis=1)
        thr_evt = (vmax * float(thr_frac)).reshape(-1, 1)
        mask = yprof > thr_evt
    return mask.sum(axis=1).astype(np.int32)


def load_geom(
    cfg_path: Path,
    slice_index: int,
    thr_abs: Optional[float],
    thr_frac: float,
    use_abs: bool,
    max_events: Optional[int],
    per_file: bool,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    cfg = load_yaml(cfg_path)
    recon_paths = list(iter_recon_files(cfg))
    combined: List[np.ndarray] = []
    per: List[Tuple[str, np.ndarray]] = []

    for rp in recon_paths:
        if not rp.exists():
            raise FileNotFoundError(f"Recon parquet not found: {rp}")
        tyx = read_recon_tyx(rp, max_events=max_events)
        yprof = yprofile_from_tyx(tyx, slice_index=slice_index, use_abs=use_abs)
        ys = ysize_from_yprofile(yprof, thr_abs=thr_abs, thr_frac=thr_frac)
        combined.append(ys)
        if per_file:
            per.append((rp.name, ys))

    return (np.concatenate(combined) if combined else np.array([], dtype=np.int32)), per


def step_hist(ax, data: np.ndarray, bins: np.ndarray, label: str, lw: float = 2.0, alpha: float = 1.0):
    c, e = np.histogram(data, bins=bins)
    ax.step(e[:-1], c, where="post", linewidth=lw, label=label, alpha=alpha)


def small_legend(ax, loc: str, title: Optional[str] = None, ncol: int = 1):
    ax.legend(
        loc=loc,
        title=title,
        fontsize=7,
        title_fontsize=8,
        framealpha=0.2,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        ncol=ncol,
    )


def fig3(
    cfg_root: Path,
    dataset: str,
    outpath: Path,
    slice_index: int,
    thr_abs: Optional[float],
    thr_frac: float,
    use_abs: bool,
    max_events: Optional[int],
    save_pdf: bool,
):
    subdir = "barrel-phys-pt" if dataset == "phys" else "barrel-flat-pt"
    geoms_left = ["50x10", "50x12P5", "50x15", "50x20", "50x25"]
    geoms_right = ["100x25", "100x25x150"]
    bins = np.arange(-0.5, 13.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)

    ax = axes[0]
    for g in geoms_left:
        cfg_path = cfg_root / subdir / f"{g}.yml"
        if not cfg_path.exists():
            continue
        ys, _ = load_geom(cfg_path, slice_index, thr_abs, thr_frac, use_abs, max_events, per_file=False)
        lbl = f"{g.replace('P','.') }x100"
        step_hist(ax, ys, bins, lbl)
    ax.set_xlabel("Cluster Y-size [pixels]")
    ax.set_ylabel("Number of events")
    ax.set_xlim(0, 13.0)
    ax.grid(True, alpha=0.35)
    small_legend(ax, "upper right", title=r"Sensor geometry [$\mu m^3$]")

    ax = axes[1]
    for g in geoms_right:
        cfg_path = cfg_root / subdir / f"{g}.yml"
        if not cfg_path.exists():
            continue
        ys, _ = load_geom(cfg_path, slice_index, thr_abs, thr_frac, use_abs, max_events, per_file=False)
        lbl = "100x25x100" if g == "100x25" else "100x25x150"
        step_hist(ax, ys, bins, lbl)
    ax.set_xlabel("Cluster Y-size [pixels]")
    ax.set_ylabel("Number of events")
    ax.set_xlim(0, 13.0)
    ax.grid(True, alpha=0.35)
    small_legend(ax, "upper right", title=r"Sensor geometry [$\mu m^3$]")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    if save_pdf:
        fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def perfile_plot(
    cfg_root: Path,
    dataset: str,
    geom: str,
    outpath: Path,
    slice_index: int,
    thr_abs: Optional[float],
    thr_frac: float,
    use_abs: bool,
    max_events: Optional[int],
    save_pdf: bool,
):
    subdir = "barrel-phys-pt" if dataset == "phys" else "barrel-flat-pt"
    cfg_path = cfg_root / subdir / f"{geom}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    combined, per = load_geom(cfg_path, slice_index, thr_abs, thr_frac, use_abs, max_events, per_file=True)
    bins = np.arange(-0.5, 13.5, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.0), constrained_layout=True)
    step_hist(ax, combined, bins, f"{geom} combined", lw=2.5, alpha=1.0)
    for name, ys in per:
        step_hist(ax, ys, bins, name, lw=1.0, alpha=0.45)

    ax.set_xlabel("Cluster Y-size [pixels]")
    ax.set_ylabel("Number of events")
    ax.set_xlim(0, 13.0)
    ax.grid(True, alpha=0.35)
    small_legend(ax, "upper right", title=f"{geom} per-file", ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    if save_pdf:
        fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-root", default="config/17180303")
    ap.add_argument("--outdir", default="plotting/figures")
    ap.add_argument("--dataset", choices=["phys", "flat"], default="phys")
    ap.add_argument("--slice-index", type=int, default=-1)
    ap.add_argument("--thr-abs", type=float, default=None)
    ap.add_argument("--thr-frac", type=float, default=0.05)
    ap.add_argument("--no-abs", action="store_true")
    ap.add_argument("--max-events-per-file", type=int, default=None)
    ap.add_argument("--perfile-geom", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    cfg_root = Path(args.config_root)
    outdir = Path(args.outdir)
    use_abs = not args.no_abs
    save_pdf = not args.no_pdf

    fig3(
        cfg_root,
        args.dataset,
        outdir / f"fig3_cluster_ysize_{args.dataset}.png",
        args.slice_index,
        args.thr_abs,
        args.thr_frac,
        use_abs,
        args.max_events_per_file,
        save_pdf,
    )

    if args.perfile_geom is not None:
        perfile_plot(
            cfg_root,
            args.dataset,
            args.perfile_geom,
            outdir / f"fig3_cluster_ysize_perfile_{args.perfile_geom}_{args.dataset}.png",
            args.slice_index,
            args.thr_abs,
            args.thr_frac,
            use_abs,
            args.max_events_per_file,
            save_pdf,
        )


if __name__ == "__main__":
    main()
