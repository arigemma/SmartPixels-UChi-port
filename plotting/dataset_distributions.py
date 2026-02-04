
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


PT_CANDIDATES = ["pt", "pT", "pT_GeV", "pt_GeV", "pt_true", "pT_true", "true_pt", "PT"]

ALPHA_CANDIDATES = ["alpha", "Alpha", "inc_alpha", "track_alpha", "alpha_rad", "ALPHA", "cotAlpha", "cotalpha"]
BETA_CANDIDATES  = ["beta", "Beta", "inc_beta", "track_beta", "beta_rad", "BETA", "cotBeta", "cotbeta"]
NX_CANDIDATES = ["n_x", "nx", "dir_x"]
NY_CANDIDATES = ["n_y", "ny", "dir_y"]
NZ_CANDIDATES = ["n_z", "nz", "dir_z"]




def _resolve_column(df_cols: Iterable[str], candidates: List[str], label: str) -> str:
    cols = list(df_cols)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]



def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def iter_label_files(cfg: dict) -> Iterable[Path]:
    root = Path(cfg["root"])
    if "train" not in cfg:
        raise KeyError(f"Config missing 'train' key: {cfg.keys()}")
    for pair in cfg["train"]:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        labels = pair[1]
        yield root / labels


def load_pt_alpha_beta_from_config(cfg_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_yaml(cfg_path)
    label_paths = list(iter_label_files(cfg))


    pt_list: List[np.ndarray] = []
    a_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []

    for lp in label_paths:
        if not lp.exists():
            raise FileNotFoundError(f"Labels parquet not found: {lp}")

        cols = list(pd.read_parquet(lp, engine="pyarrow", columns=None).columns)

        pt_col = _resolve_column(cols, PT_CANDIDATES, "pt")

        nx_col = _resolve_column(cols, NX_CANDIDATES, "n_x")
        ny_col = _resolve_column(cols, NY_CANDIDATES, "n_y")
        nz_col = _resolve_column(cols, NZ_CANDIDATES, "n_z")

        df = pd.read_parquet(lp, engine="pyarrow", columns=[pt_col, nx_col, ny_col, nz_col])

        pt = np.abs(df[pt_col].to_numpy(dtype=np.float32, copy=False))

        nx = df[nx_col].to_numpy(dtype=np.float32, copy=False)
        ny = df[ny_col].to_numpy(dtype=np.float32, copy=False)
        nz = df[nz_col].to_numpy(dtype=np.float32, copy=False)

      
        alpha = np.arctan(nx / nz).astype(np.float32, copy=False)
        beta  = np.arctan(ny / nz).astype(np.float32, copy=False)

        pt_list.append(pt)
        a_list.append(alpha)
        b_list.append(beta)

    pt = np.concatenate(pt_list) if pt_list else np.array([], dtype=np.float32)
    alpha = np.concatenate(a_list) if a_list else np.array([], dtype=np.float32)
    beta = np.concatenate(b_list) if b_list else np.array([], dtype=np.float32)
    return pt, alpha, beta




def step_hist(ax, data, bins, rng, label, linestyle, linewidth=2.0):
    counts, edges = np.histogram(data, bins=bins, range=rng)
    ax.step(edges[:-1], counts, where="post", label=label, linestyle=linestyle, linewidth=linewidth)


def make_figure(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    outpath: Path,
    title: Optional[str] = None,
) -> None:
    # Match paper-like layout: 3 panels horizontally
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2), constrained_layout=True)

    # Styles (mirror paper legend semantics)
    # (We rely on matplotlib's default color cycle; linestyle encodes dataset type.)
    style = {
        "barrel_phys": dict(label="Physical p$_T$, barrel region", linestyle="-"),
        "barrel_flat": dict(label="Flat p$_T$, barrel region", linestyle="--"),
        "endcap_phys": dict(label="Physical p$_T$, endcap region", linestyle="-."),
        "endcap_flat": dict(label="Flat p$_T$, endcap region", linestyle="-"),
    }
    legend_kw = dict(
        fontsize=6,          
        framealpha=0.2,     
        borderpad=0.25,      
        labelspacing=0.2,    
        handlelength=1.2,   
        handletextpad=0.4,  
        borderaxespad=0.2,   
    )
    # Panel 1: pT (linear y)
    ax = axes[0]
    for key in ["barrel_phys", "barrel_flat", "endcap_phys", "endcap_flat"]:
        if key not in datasets:
            continue
        pt, _, _ = datasets[key]
        step_hist(ax, pt, bins=120, rng=(0.0, 5.0), **style[key])
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Number of events")
    ax.set_xlim(0.0, 5.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", **legend_kw)

    # Panel 2: alpha (log y)
    ax = axes[1]
    for key in ["barrel_phys", "barrel_flat", "endcap_phys", "endcap_flat"]:
        if key not in datasets:
            continue
        _, a, _ = datasets[key]
        step_hist(ax, a, bins=120, rng=(-1.6, 1.6), **style[key])
    ax.set_xlabel(r"$\alpha$ [rad]")
    ax.set_ylabel("Number of events (log)")
    ax.set_xlim(-1.6, 1.6)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper left", **legend_kw)

    # Panel 3: beta (log y)
    ax = axes[2]
    for key in ["barrel_phys", "barrel_flat", "endcap_phys", "endcap_flat"]:
        if key not in datasets:
            continue
        _, _, b = datasets[key]
        step_hist(ax, b, bins=120, rng=(-1.0, 1.0), **style[key])
    ax.set_xlabel(r"$\beta$ [rad]")
    ax.set_ylabel("Number of events (log)")
    ax.set_xlim(-1.0, 1.0)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right", **legend_kw)

    if title:
        fig.suptitle(title, fontsize=11)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    # also save PDF for paper-ready inclusion
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-root",
        type=str,
        default="config/17180303",
        help="Path to the config root containing barrel-flat-pt/, barrel-phys-pt/, etc.",
    )
    ap.add_argument(
        "--geometry",
        type=str,
        default="50x10",
        help="Geometry YAML basename (without .yml), e.g. 50x10, 50x12P5, 100x25x150",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="plotting/figures",
        help="Output directory for generated images",
    )
    ap.add_argument(
        "--no-endcap",
        action="store_true",
        help="Only plot barrel datasets (ignore endcap even if configs exist).",
    )
    args = ap.parse_args()

    cfg_root = Path(args.config_root)
    geom = args.geometry
    outdir = Path(args.outdir)

    barrel_flat = cfg_root / "barrel-flat-pt" / f"{geom}.yml"
    barrel_phys = cfg_root / "barrel-phys-pt" / f"{geom}.yml"

    if not barrel_flat.exists():
        raise FileNotFoundError(f"Missing barrel flat config: {barrel_flat}")
    if not barrel_phys.exists():
        raise FileNotFoundError(f"Missing barrel phys config: {barrel_phys}")

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print(f"[INFO] Loading: {barrel_phys}")
    datasets["barrel_phys"] = load_pt_alpha_beta_from_config(barrel_phys)
    print(f"[INFO] Loading: {barrel_flat}")
    datasets["barrel_flat"] = load_pt_alpha_beta_from_config(barrel_flat)

    if not args.no_endcap:
        endcap_flat = cfg_root / "endcap-flat-pt" / f"{geom}.yml"
        endcap_phys = cfg_root / "endcap-phys-pt" / f"{geom}.yml"
        if endcap_phys.exists() and endcap_flat.exists():
            print(f"[INFO] Loading: {endcap_phys}")
            datasets["endcap_phys"] = load_pt_alpha_beta_from_config(endcap_phys)
            print(f"[INFO] Loading: {endcap_flat}")
            datasets["endcap_flat"] = load_pt_alpha_beta_from_config(endcap_flat)
        else:
            print("[WARN] Endcap configs not found (or incomplete). Plotting barrel only.")

    outpath = outdir / f"fig2_distributions_{geom}.png"
    title = f"Distributions for geometry {geom}"
    make_figure(datasets, outpath=outpath, title=None)  
    print(f"[DONE] Wrote: {outpath} and {outpath.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
