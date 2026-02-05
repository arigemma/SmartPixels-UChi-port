from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt



def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def iter_recon_files(cfg: dict) -> Iterable[Path]:
    root = Path(cfg["root"])

    for pair in cfg["train"]:
        if not isinstance(pair, list) or len(pair) < 1:
            continue
        yield root / pair[0]



def _is_numeric_str_col(c: str) -> bool:
    try:
        int(c)
        return True
    except Exception:
        return False


def _get_flat_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if _is_numeric_str_col(c)]

    cols = sorted(cols, key=lambda x: int(x))
    return df[cols].to_numpy(dtype=np.float32, copy=False)


def _infer_txy(F: int) -> Tuple[int, int, int]:
    if F == 20 * 21 * 13:
        return 20, 21, 13
    preferred = [(20, 21, 13), (20, 13, 21), (1, 21, 13), (1, 13, 21)]
    for T, X, Y in preferred:
        if T * X * Y == F:
            return T, X, Y
    best = None
    best_score = 1e18
    for T in range(1, 65):
        if F % T != 0:
            continue
        rem = F // T
        for X in range(5, 65):
            if rem % X != 0:
                continue
            Y = rem // X
            if Y < 5 or Y > 65:
                continue
            score = min((X - 21) ** 2 + (Y - 13) ** 2, (X - 13) ** 2 + (Y - 21) ** 2)
            if score < best_score:
                best_score = score
                best = (T, X, Y)

    return best


def _reshape_flat(flat: np.ndarray, T: int, X: int, Y: int, order: str) -> np.ndarray:

    N, F = flat.shape
    if T * X * Y != F:
        raise ValueError(f"T*X*Y={T*X*Y} != F={F}")
    if order == "TXY":
        return flat.reshape(N, T, X, Y)
    if order == "XYT":
        return flat.reshape(N, X, Y, T).transpose(0, 3, 1, 2)



def _choose_order(flat: np.ndarray, T: int, X: int, Y: int) -> str:
   
    txy = _reshape_flat(flat, T, X, Y, "TXY")
    s0 = float(np.mean(np.sum(txy[:, 0, :, :], axis=(1, 2))))
    sL = float(np.mean(np.sum(txy[:, -1, :, :], axis=(1, 2))))
    return "TXY" if sL >= s0 else "XYT"


def cluster_ysize_from_recon(
    recon_path: Path,
    threshold: float,
    slice_index: Optional[int],
    max_events: Optional[int],
    verbose: bool = False,
) -> np.ndarray:
    df = pd.read_parquet(recon_path, engine="pyarrow")
    if max_events is not None:
        df = df.iloc[:max_events].copy()

    flat = _get_flat_numeric_matrix(df)
    N, F = flat.shape
    T, X, Y = _infer_txy(F)
    order = _choose_order(flat, T, X, Y)

    if verbose:
        print(f"[INFO] {recon_path.name}: N={N}, F={F} -> (T,X,Y)=({T},{X},{Y}), order={order}")

    txy = _reshape_flat(flat, T, X, Y, order)

    if slice_index is None:
        slice_index = -1
    xy = txy[:, slice_index, :, :]  
    if xy.shape[1] < xy.shape[2]:
        xy = xy.transpose(0, 2, 1)

    mask = xy > threshold
    y_active = mask.any(axis=1)    
    return y_active.sum(axis=1).astype(np.int32)


def load_ysizes_from_config(
    cfg_path: Path,
    threshold: float,
    slice_index: Optional[int],
    max_events_per_file: Optional[int],
    verbose: bool,
) -> np.ndarray:
    cfg = load_yaml(cfg_path)
    recon_paths = list(iter_recon_files(cfg))


    chunks: List[np.ndarray] = []
    for i, rp in enumerate(recon_paths):

        chunks.append(
            cluster_ysize_from_recon(
                rp,
                threshold=threshold,
                slice_index=slice_index,
                max_events=max_events_per_file,
                verbose=verbose and (i == 0),
            )
        )
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)



def step_hist(ax, data: np.ndarray, bins: np.ndarray, label: str, linestyle: str):
    counts, edges = np.histogram(data, bins=bins)
    ax.step(edges[:-1], counts, where="post", linewidth=2.0, label=label, linestyle=linestyle)


def make_plot(datasets: Dict[str, np.ndarray], outpath: Path, geom: str):
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0), constrained_layout=True)
    bins = np.arange(-0.5, 15.5, 1.0)

    legend_kw = dict(
        fontsize=8,
        framealpha=0.2,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    for label, (data, linestyle) in datasets.items():
        step_hist(ax, data, bins=bins, label=label, linestyle=linestyle)

    ax.set_xlabel("Cluster Y-size [pixels]")
    ax.set_ylabel("Number of events")
    ax.set_xlim(0, 14.5)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", title=f"{geom}", **legend_kw)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-root", default="config/17180303", help="Config root")
    ap.add_argument("--geometry", default="50x10", help="Which geometry YAML to use (basename w/o .yml)")
    ap.add_argument("--outdir", default="plotting/figures", help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.0, help="Pixel charge threshold")
    ap.add_argument("--slice-index", type=int, default=-1, help="Time slice index (-1 = last)")
    ap.add_argument("--max-events-per-file", type=int, default=None, help="Cap per file for quick testing")
    ap.add_argument("--verbose", action="store_true", help="Print inferred shapes/orders")
    ap.add_argument("--dataset", choices=["phys", "flat", "both"], default="both",
                    help="Plot physical, flat, or both overlaid")
    args = ap.parse_args()

    cfg_root = Path(args.config_root)
    geom = args.geometry
    outdir = Path(args.outdir)
    slice_index = None if args.slice_index == -1 else args.slice_index

    to_load: List[Tuple[str, str, str]] = []
    if args.dataset in ("phys", "both"):
        to_load.append(("barrel-phys-pt", "Physical p$_T$", "-"))
    if args.dataset in ("flat", "both"):
        to_load.append(("barrel-flat-pt", "Flat p$_T$", "--"))

    datasets: Dict[str, Tuple[np.ndarray, str]] = {}
    for subdir, label, linestyle in to_load:
        cfg_path = cfg_root / subdir / f"{geom}.yml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        print(f"[INFO] Loading {geom} ({label}): {cfg_path}")
        ys = load_ysizes_from_config(
            cfg_path,
            threshold=args.threshold,
            slice_index=slice_index,
            max_events_per_file=args.max_events_per_file,
            verbose=args.verbose,
        )
        datasets[label] = (ys, linestyle)

    outpath = outdir / f"cluster_ysize_{geom}_{args.dataset}.png"
    make_plot(datasets, outpath, geom=geom)
    print(f"[DONE] Wrote: {outpath} and {outpath.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
