from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


PT_CANDIDATES = ["pt", "pT", "pT_GeV", "pt_GeV", "pt_true", "pT_true", "true_pt", "PT"]


def _resolve_column(cols: List[str], candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find {label}. Tried {candidates}. Available={cols[:50]}")


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def iter_label_files(cfg: dict) -> List[Path]:
    root = Path(cfg["root"])
    if "train" not in cfg:
        raise KeyError(f"Config missing 'train'. Keys={list(cfg.keys())}")
    out = []
    for pair in cfg["train"]:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        out.append(root / pair[1])
    return out


def read_pt(lp: Path) -> np.ndarray:
    cols = list(pd.read_parquet(lp, engine="pyarrow", columns=None).columns)
    pt_col = _resolve_column(cols, PT_CANDIDATES, "pt")
    df = pd.read_parquet(lp, engine="pyarrow", columns=[pt_col])
    return df[pt_col].to_numpy(dtype=np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (barrel-flat-pt/50x10.yml etc.)")
    ap.add_argument("--outdir", default="plotting/figures", help="Where to write plots")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--pt-max", type=float, default=5.0)
    ap.add_argument("--logy", action="store_true", help="Use log-scale y-axis")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    label_paths = iter_label_files(cfg)

    if not label_paths:
        raise RuntimeError(f"No labels found in {cfg_path}")

    per_file: List[Tuple[str, np.ndarray]] = []
    for lp in label_paths:
        if not lp.exists():
            raise FileNotFoundError(f"Missing labels parquet: {lp}")
        pt_signed = read_pt(lp)
        per_file.append((lp.name, pt_signed))

    print(f"\n[INFO] Config: {cfg_path}")
    print(f"[INFO] root: {cfg.get('root')}")
    print(f"[INFO] #files: {len(per_file)}\n")

    header = f"{'file':<22} {'N':>8} {'min(pt)':>10} {'max(pt)':>10} {'mean(|pt|)':>12} {'neg_frac':>10}"
    print(header)
    print("-" * len(header))
    for name, pt_signed in per_file:
        neg_frac = float(np.mean(pt_signed < 0))
        mean_abs = float(np.mean(np.abs(pt_signed)))
        print(f"{name:<22} {pt_signed.size:>8d} {pt_signed.min():>10.4f} {pt_signed.max():>10.4f} {mean_abs:>12.4f} {neg_frac:>10.3f}")

    try:
        from scipy.stats import ks_2samp  # type: ignore
        ref = np.abs(per_file[0][1])
        print("\nKS test vs first file (on |pt|):")
        for name, pt_signed in per_file[1:]:
            stat, pval = ks_2samp(ref, np.abs(pt_signed))
            print(f"  {name:<22} KS={stat:.4f}  p={pval:.3e}")
    except Exception:
        print("\n[INFO] scipy not available; skipping KS test.")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2), constrained_layout=True)
    bins = args.bins
    rng = (0.0, float(args.pt_max))

    for name, pt_signed in per_file:
        pt = np.abs(pt_signed)
        counts, edges = np.histogram(pt, bins=bins, range=rng)
        ax.step(edges[:-1], counts, where="post", linewidth=1.6, label=name)

    ax.set_xlabel(r"$|p_T|$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_xlim(*rng)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35)

    ax.legend(fontsize=7, ncols=3, framealpha=0.9)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = cfg_path.stem
    out_png = outdir / f"pt_by_file_{stem}.png"
    out_pdf = outdir / f"pt_by_file_{stem}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"\n[DONE] Wrote:\n  {out_png}\n  {out_pdf}\n")


if __name__ == "__main__":
    main()
