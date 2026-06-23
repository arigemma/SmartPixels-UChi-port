#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import create_dataloaders
from utils import get_device, CreateFolder
from models import registry


def true_class_from_signed_pt(pt_signed: np.ndarray, pt_boundary: float) -> np.ndarray:
    c = np.empty_like(pt_signed, dtype=np.int64)
    c[(pt_signed >= 0.0) & (pt_signed <= pt_boundary)] = 0
    c[(pt_signed < 0.0) & (pt_signed >= -pt_boundary)] = 1
    c[(pt_signed > pt_boundary) | (pt_signed < -pt_boundary)] = 2
    return c


def savefig(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def sanitize_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def acceptance_curve(pt_true_signed: np.ndarray, preds: np.ndarray, edges: np.ndarray) -> np.ndarray:
    acc = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    for i in range(len(edges) - 1):
        idx = (pt_true_signed >= edges[i]) & (pt_true_signed < edges[i + 1])
        n = int(idx.sum())
        if n == 0:
            continue
        acc[i] = float((preds[idx] == 2).sum()) / n
    return acc


def sig_eff_and_bkg_rej(pt_true_signed: np.ndarray, preds: np.ndarray, pt_boundary: float) -> Tuple[float, float]:
    high_idx = np.abs(pt_true_signed) >= pt_boundary
    low_idx = ~high_idx
    high_n = int(high_idx.sum())
    low_n = int(low_idx.sum())
    sig_eff = float((preds[high_idx] == 2).sum()) / max(high_n, 1)
    bkg_rej = float((preds[low_idx] != 2).sum()) / max(low_n, 1)
    return sig_eff, bkg_rej


def score_model(sig_eff: float, bkg_rej: float) -> float:
    return sig_eff * bkg_rej


def balanced_accuracy(
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    pt_boundary: float = 0.2,
) -> float:
    true_cls = true_class_from_signed_pt(pt_true_signed, pt_boundary=pt_boundary)

    recalls = [
        float((preds[true_cls == c] == c).mean())
        for c in range(3)
        if (true_cls == c).any()
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def label_with_balanced_accuracy(
    name: str,
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
) -> str:
    bal = balanced_accuracy(pt_true_signed, preds)
    return f"{name} (bal={bal:.3f})"


def print_balanced_accuracy(name: str, bal: float) -> None:
    print(f"[METRIC] {name} balanced_acc={bal:.6f}")


def print_balanced_accuracy_summary(rows: List[Tuple[str, str, str, float]]) -> None:
    if not rows:
        print("\nBalanced accuracy summary: no evaluated models.")
        return

    print("\nBalanced accuracy summary:")
    print("label\tinput_type\tbalanced_acc\tpath")
    for label, input_type, path, bal in rows:
        print(f"{label}\t{input_type}\t{bal:.6f}\t{path}")


def write_balanced_accuracy_summary(outdir: Path, rows: List[Tuple[str, str, str, float]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "balanced_accuracy_summary.tsv"
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", "input_type", "balanced_acc", "path"])
        for label, input_type, path, bal in rows:
            writer.writerow([label, input_type, f"{bal:.6f}", path])
    print("[DONE] Wrote balanced accuracy table to:", str(outpath))


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    return registry.load_registered_model(path, map_location=device)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_gate_count(model: torch.nn.Module) -> Optional[int]:
    gate = 0
    for m in model.modules():
        cn = type(m).__name__
        if ("Logix" in cn) or ("Gate" in cn):
            gate += 1
    return gate


@torch.no_grad()
def estimate_macs_from_sample(model: torch.nn.Module, x_sample: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def conv1d_macs(m: torch.nn.Conv1d, out: torch.Tensor) -> int:
        k = int(m.kernel_size[0])
        cin = int(m.in_channels)
        groups = int(m.groups)
        return int(out.numel() * (cin // groups) * k)

    def conv2d_macs(m: torch.nn.Conv2d, out: torch.Tensor) -> int:
        kH, kW = int(m.kernel_size[0]), int(m.kernel_size[1])
        cin = int(m.in_channels)
        groups = int(m.groups)
        return int(out.numel() * (cin // groups) * kH * kW)

    def conv3d_macs(m: torch.nn.Conv3d, out: torch.Tensor) -> int:
        kD, kH, kW = int(m.kernel_size[0]), int(m.kernel_size[1]), int(m.kernel_size[2])
        cin = int(m.in_channels)
        groups = int(m.groups)
        return int(out.numel() * (cin // groups) * kD * kH * kW)

    def linear_macs(m: torch.nn.Linear, out: torch.Tensor) -> int:
        return int(out.numel() * int(m.in_features))

    def hook_fn(module, inp, out):
        nonlocal macs
        if isinstance(module, torch.nn.Conv1d):
            macs += conv1d_macs(module, out)
        elif isinstance(module, torch.nn.Conv2d):
            macs += conv2d_macs(module, out)
        elif isinstance(module, torch.nn.Conv3d):
            macs += conv3d_macs(module, out)
        elif isinstance(module, torch.nn.Linear):
            macs += linear_macs(module, out)

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    _ = model(x_sample)

    for h in hooks:
        h.remove()

    return int(macs)


def hw_ballpark(model: torch.nn.Module, x_sample: torch.Tensor, kind: str) -> Dict[str, Optional[float]]:
    MAC_NAND2 = 2500.0
    CMP_NAND2 = 200.0
    BOOL_NAND2 = 1.0

    p = float(count_params(model))
    g = float(estimate_gate_count(model) or 0)

    if kind == "tf":
        macs = float(estimate_macs_from_sample(model, x_sample))
        nand2 = macs * MAC_NAND2
        return {"macs": macs, "comparators": None, "bool_ops": None, "nand2_eq": nand2}

    comps = p
    bool_ops = g
    nand2 = comps * CMP_NAND2 + bool_ops * BOOL_NAND2
    return {"macs": None, "comparators": comps, "bool_ops": bool_ops, "nand2_eq": nand2}


def per_bin_rate(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full_like(denom, np.nan, dtype=np.float64)
    m = denom > 0
    out[m] = numer[m] / denom[m]
    return out


def y0_region_metrics(
    y0: np.ndarray,
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    pt_boundary: float,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_pt_boundary = 0.2
    sig_pt_min = 2.0
    true_cls = true_class_from_signed_pt(pt_true_signed, pt_boundary=class_pt_boundary)

    nb = len(edges) - 1
    counts = np.zeros(nb, dtype=np.int64)
    sig_denom = np.zeros(nb, dtype=np.int64)
    sig_numer = np.zeros(nb, dtype=np.int64)
    acc_denom = np.zeros(nb, dtype=np.int64)
    acc_numer = np.zeros(nb, dtype=np.int64)

    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        idx = (y0 >= lo) & (y0 < hi)
        n = int(idx.sum())
        counts[i] = n
        if n == 0:
            continue

        acc_denom[i] = n
        acc_numer[i] = int((preds[idx] == true_cls[idx]).sum())

        sig_mask = idx & (np.abs(pt_true_signed) > sig_pt_min)
        s = int(sig_mask.sum())
        sig_denom[i] = s
        if s > 0:
            sig_numer[i] = int((preds[sig_mask] == 2).sum())

    sig_eff = 100.0 * per_bin_rate(sig_numer.astype(np.float64), sig_denom.astype(np.float64))
    acc = 100.0 * per_bin_rate(acc_numer.astype(np.float64), acc_denom.astype(np.float64))
    return sig_eff, acc, counts


def plot_y0_eff_acc_paper_style(
    y0: np.ndarray,
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    outpath: Path,
    pt_boundary: float,
    edges: Optional[np.ndarray] = None,
):
    if edges is None:
        edges = np.arange(-8.0, 9.0, 1.0, dtype=np.float64)

    sig_eff, acc, _ = y0_region_metrics(
        y0=y0,
        pt_true_signed=pt_true_signed,
        preds=preds,
        pt_boundary=pt_boundary,
        edges=edges,
    )

    sig_y = np.r_[sig_eff, sig_eff[-1]]
    acc_y = np.r_[acc, acc[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.6), constrained_layout=False)
    ax.step(edges, sig_y, where="post", linewidth=1.6, label="signal efficiency")
    ax.step(edges, acc_y, where="post", linewidth=1.6, label="accuracy")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel(r"$y_0$ [mm]")

    yt = np.arange(0, 101, 20)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:d}%" for t in yt])

    ax.set_xticks(np.arange(int(edges[0]), int(edges[-1]) + 1, 2))
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.legend(loc="lower right")

    savefig(fig, outpath)


def plot_y0_histogram(
    y0: np.ndarray,
    outpath: Path,
    edges: Optional[np.ndarray] = None,
):
    if edges is None:
        edges = np.arange(-8.0, 9.0, 1.0, dtype=np.float64)

    counts, _ = np.histogram(y0, bins=edges)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.2), constrained_layout=False)
    ax.step(edges, np.r_[counts, counts[-1]], where="post", linewidth=1.6)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"$y_0$ [mm]")
    ax.set_ylabel("events")
    ax.set_xticks(np.arange(int(edges[0]), int(edges[-1]) + 1, 2))
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.7)

    savefig(fig, outpath)


def wilson_interval(k: int, n: int, z: float = 1.0) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return center - half, center + half


def acceptance_with_wilson_errors(
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    edges: np.ndarray,
    z: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    err_lo = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    err_hi = np.full(len(edges) - 1, np.nan, dtype=np.float64)

    for i in range(len(edges) - 1):
        idx = (pt_true_signed >= edges[i]) & (pt_true_signed < edges[i + 1])
        n = int(idx.sum())
        if n == 0:
            continue
        k = int((preds[idx] == 2).sum())
        p = k / n
        lo, hi = wilson_interval(k, n, z=z)
        acc[i] = p
        err_lo[i] = p - lo
        err_hi[i] = hi - p

    return acc, err_lo, err_hi


def bin_centers_and_xerr(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * (edges[1:] - edges[:-1])
    return centers, xerr


def plot_curve_with_wilson(
    ax,
    pt_true_signed: np.ndarray,
    preds: np.ndarray,
    edges: np.ndarray,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
    color=None,
    z: float = 1.0,
):
    centers, xerr = bin_centers_and_xerr(edges)
    acc, err_lo, err_hi = acceptance_with_wilson_errors(
        pt_true_signed=pt_true_signed,
        preds=preds,
        edges=edges,
        z=z,
    )
    return ax.errorbar(
        centers,
        acc,
        xerr=xerr,
        yerr=[err_lo, err_hi],
        marker=marker,
        linestyle=linestyle,
        color=color,
        capsize=2.5,
        elinewidth=1.0,
        label=label,
    )


@torch.no_grad()
def eval_model_pred_classes(
    model_path: str,
    data_config: str,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    forward_mode: str,
) -> Tuple[torch.nn.Module, str, str, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    model = load_model(model_path, device).to(device)

    model_name, model_cfg = registry.class_to_config(type(model))
    input_type = model_cfg["input_type"]

    if forward_mode == "discrete":
        model.eval()
    elif forward_mode == "baseline":
        model.eval()
    else:
        model.train()

    if forward_mode == "baseline":
        _, _, test_loader = create_dataloaders(
            config_path=data_config,
            batch_size=batch_size,
            shuffle=False,
            input_type=input_type,
            target_type="raw",
            balance_classes=False,
            val_size=0.0,
        )
    else:
        _, _, test_loader = create_dataloaders(
            config_path=data_config,
            batch_size=batch_size,
            shuffle=False,
            input_type=input_type,
            target_type="raw",
            balance_classes=False,
            val_size=0.0,
            drop_dominant_features=False,
            dominant_thr=0.95,
            dominant_num_batches=10,
        )

    label_names = test_loader.dataset.label_names
    if "pt" not in label_names:
        raise ValueError(f"'pt' not found in label_names={label_names}")
    if "y-local" not in label_names:
        raise ValueError(f"'y-local' not found in label_names={label_names} (needed for y0 region plots)")

    pt_idx = label_names.index("pt")
    y0_idx = label_names.index("y-local")

    true_pts: List[np.ndarray] = []
    true_y0s: List[np.ndarray] = []
    logits_chunks: List[torch.Tensor] = []
    x_sample: Optional[torch.Tensor] = None

    for i, (x, y) in enumerate(tqdm(test_loader, desc=f"eval {Path(model_path).name} [{forward_mode}]")):
        x = x.to(device)

        if x_sample is None:
            x_sample = x[:1].detach()

        logits_chunks.append(model(x))
        true_pts.append(y[:, pt_idx].cpu().numpy())
        true_y0s.append(y[:, y0_idx].cpu().numpy())

        if max_batches > 0 and (i + 1) >= max_batches:
            break

    assert x_sample is not None, "Empty dataloader? Could not capture x_sample."

    true_pts_arr = np.concatenate(true_pts, axis=0)
    true_y0_arr = np.concatenate(true_y0s, axis=0)
    logits = torch.cat(logits_chunks, dim=0)
    pred_classes = logits.argmax(dim=1).cpu().numpy().astype(np.int8)

    return model, model_name, input_type, true_pts_arr, true_y0_arr, pred_classes, x_sample


def plot_acceptance(
    input_type: str,
    pt_true_signed: np.ndarray,
    baseline_relaxed: np.ndarray,
    baseline_discrete: np.ndarray,
    best_relaxed: np.ndarray,
    best_discrete: np.ndarray,
    outpath: Path,
    baseline_label: str,
    best_relaxed_label: str,
    best_discrete_label: str,
    baseline_kind: str,
    z: float = 1.0,
):
    edges = np.array(
        [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4,
         -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        dtype=np.float64
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.2), constrained_layout=False)

    def plot_pair(label_prefix: str, preds_rel: np.ndarray, preds_dis: np.ndarray):
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=preds_rel,
            edges=edges,
            label=label_with_balanced_accuracy(f"{label_prefix} (relaxed)", pt_true_signed, preds_rel),
            marker="o",
            linestyle="-",
            z=z,
        )
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=preds_dis,
            edges=edges,
            label=label_with_balanced_accuracy(f"{label_prefix} (discrete)", pt_true_signed, preds_dis),
            marker="*",
            linestyle="--",
            z=z,
        )
    if baseline_kind in ("tf", "quantized"):
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=baseline_relaxed,
            edges=edges,
            label=label_with_balanced_accuracy(baseline_label, pt_true_signed, baseline_relaxed),
            marker="o",
            linestyle="-",
            z=z,
        )
    else:
        plot_pair(baseline_label, baseline_relaxed, baseline_discrete)

    plot_curve_with_wilson(
        ax=ax,
        pt_true_signed=pt_true_signed,
        preds=best_relaxed,
        edges=edges,
        label=label_with_balanced_accuracy(best_relaxed_label, pt_true_signed, best_relaxed),
        marker="o",
        linestyle="-",
        z=z,
    )

    plot_curve_with_wilson(
        ax=ax,
        pt_true_signed=pt_true_signed,
        preds=best_discrete,
        edges=edges,
        label=label_with_balanced_accuracy(best_discrete_label, pt_true_signed, best_discrete),
        marker="*",
        linestyle="--",
        z=z,
    )

    ax.set_title(f"Acceptance vs true $p_T$ — {input_type}")
    ax.set_xlabel(r"true $p_T$ [GeV]")
    ax.set_ylabel(r"acceptance as high-$p_T$ (class=2)")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.grid(True, alpha=0.35)

    fig.subplots_adjust(right=0.72)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.85,
        borderaxespad=0.0,
    )

    savefig(fig, outpath)



def plot_acceptance_multiple_baselines(
    baseline_items: List[Tuple[str, np.ndarray, np.ndarray]],
    outpath: Path,
    title: str = "Acceptance vs true $p_T$",
    z: float = 1.0,
):
    """
    baseline_items: list of (label, true_pts_signed, preds)
    """
    edges = np.array(
        [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4,
         -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        dtype=np.float64
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4), constrained_layout=True)

    for i, (label, pt_true_signed, preds) in enumerate(baseline_items):

        if "Baseline" in label:
            color = "tab:blue"
            linestyle = "-"
            marker = "o"

        elif "All" in label:
            color = "black"
            linestyle = "--"
            marker = "s"

        else:
            color = None
            linestyle = "-"
            marker = "o"

        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=preds,
            edges=edges,
            label=label_with_balanced_accuracy(label, pt_true_signed, preds),
            marker=marker,
            linestyle=linestyle,
            color=color,
            z=z,
        )

    ax.set_title(title)
    ax.set_xlabel(r"true $p_T$ [GeV]")
    ax.set_ylabel(r"acceptance as high-$p_T$ (class=2)")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    savefig(fig, outpath)

def plot_acceptance_baseline_only(
    input_type: str,
    pt_true_signed: np.ndarray,
    baseline_relaxed: np.ndarray,
    baseline_discrete: np.ndarray,
    outpath: Path,
    baseline_label: str,
    baseline_kind: str,
    z: float = 1.0,
):
    edges = np.array(
        [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4,
         -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        dtype=np.float64
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)

    if baseline_kind == "lgn":
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=baseline_relaxed,
            edges=edges,
            label=label_with_balanced_accuracy(f"{baseline_label} (relaxed)", pt_true_signed, baseline_relaxed),
            marker="o",
            linestyle="-",
            z=z,
        )
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=baseline_discrete,
            edges=edges,
            label=label_with_balanced_accuracy(f"{baseline_label} (discrete)", pt_true_signed, baseline_discrete),
            marker="*",
            linestyle="--",
            z=z,
        )
    else:
        plot_curve_with_wilson(
            ax=ax,
            pt_true_signed=pt_true_signed,
            preds=baseline_relaxed,
            edges=edges,
            label=label_with_balanced_accuracy(baseline_label, pt_true_signed, baseline_relaxed),
            marker="o",
            linestyle="-",
            z=z,
        )

    ax.set_title(f"Acceptance vs true $p_T$ — {input_type}")
    ax.set_xlabel(r"true $p_T$ [GeV]")
    ax.set_ylabel(r"acceptance as high-$p_T$ (class=2)")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    savefig(fig, outpath)
def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected key=value, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_kv_pairs(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected key=value, got: {it}")
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def parse_name_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"Expected /path:Label, got: {it}")
        p, label = it.split(":", 1)
        out[p.strip()] = label.strip().strip('"').strip("'")
    return out


def label_for(path: str, fallback: str, name_overrides: Dict[str, str]) -> str:
    return name_overrides.get(path, fallback)


def checkpoint_label(path: str) -> str:
    p = Path(path)
    if p.parent.parent.name:
        return f"{p.parent.parent.name}/{p.parent.name}"
    return p.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--outdir", action=CreateFolder, default="results/eval_acceptance_best")
    ap.add_argument("--pt-boundary", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = no limit")

    ap.add_argument(
        "--baseline-kind",
        choices=["lgn", "tf", "quantized"],
        default="lgn",
        help="Baseline type: 'lgn' has relaxed+discrete; 'tf' and 'quantized' are single-forward.",
    )
    ap.add_argument(
        "--baseline",
        nargs="+",
        default=[],
        help="List input_type=path entries. Optional when only --candidates are being evaluated.",
    )
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=[],
        help="List of LGN model .pth paths. Leave empty for baseline-only plotting.",
    )
    ap.add_argument(
        "--candidate-forward",
        choices=["both", "relaxed", "discrete"],
        default="both",
        help="Which LGN forward mode(s) to evaluate for --candidates.",
    )
    ap.add_argument(
        "--name",
        nargs="*",
        default=[],
        help='Optional name overrides: use syntax /path/to/model.pth:"Human readable name"',
    )
    ap.add_argument(
        "--baseline-error-z",
        type=float,
        default=1.0,
        help="Wilson interval z-value for error bars. 1.0 ~ 68%%, 1.96 ~ 95%%.",
    )
    ap.add_argument(
    "--combine-baselines",
    action="store_true",
    help="If set and no candidates are given, plot all baselines on one acceptance plot.",
    )
    ap.add_argument(
        "--stream-metrics",
        action="store_true",
        help="Print each balanced accuracy as soon as it is computed. The final summary is always printed.",
    )

    args = ap.parse_args()
    device = get_device(args.device)
    outdir = Path(args.outdir)

    name_overrides = parse_name_overrides(args.name)
    baseline_map = parse_kv_list(args.baseline)
    metric_rows: List[Tuple[str, str, str, float]] = []

    if not baseline_map and not args.candidates:
        raise SystemExit("No models specified. Provide --baseline and/or --candidates.")

    def record_balanced_accuracy(
        label: str,
        input_type: str,
        path: str,
        pt_true_signed: np.ndarray,
        preds: np.ndarray,
    ) -> float:
        bal = balanced_accuracy(pt_true_signed, preds)
        metric_rows.append((label, input_type, path, bal))
        if args.stream_metrics:
            print_balanced_accuracy(label, bal)
        return bal

    if args.baseline_kind == "quantized" and args.combine_baselines:
        baseline_items_by_input: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
        for input_type, path in parse_kv_pairs(args.baseline):
            model, mname, it, true_pts, true_y0, preds, x_sample = eval_model_pred_classes(
                model_path=path,
                data_config=args.data_config,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                forward_mode="baseline",
            )
            if it != input_type:
                print(f"[WARN] Baseline key '{input_type}' but model reports input_type '{it}'. Using '{it}'.")
                input_type = it

            fallback = Path(path).parent.name
            label = label_for(path, fallback, name_overrides)
            record_balanced_accuracy(label, input_type, path, true_pts, preds)
            baseline_items_by_input.setdefault(input_type, []).append((label, true_pts, preds))

        for input_type, baseline_items in baseline_items_by_input.items():
            outpath = outdir / f"acceptance_quantized_{sanitize_filename(input_type)}.png"
            plot_acceptance_multiple_baselines(
                baseline_items=baseline_items,
                outpath=outpath,
                title=f"Acceptance vs true $p_T$ — {input_type}",
                z=args.baseline_error_z,
            )

        print_balanced_accuracy_summary(metric_rows)
        write_balanced_accuracy_summary(outdir, metric_rows)
        print("[DONE] Wrote outputs to:", str(outdir))
        return

    baseline_by_input: Dict[str, Dict] = {}
    for input_type, path in baseline_map.items():
        if args.baseline_kind in ("tf", "quantized"):
            model, mname, it, true_pts, true_y0, preds, x_sample = eval_model_pred_classes(
                model_path=path,
                data_config=args.data_config,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                forward_mode="baseline",
            )
            if it != input_type:
                print(f"[WARN] Baseline key '{input_type}' but model reports input_type '{it}'. Using '{it}'.")
                input_type = it

            fallback = Path(path).parent.name
            label = label_for(path, fallback, name_overrides)
            record_balanced_accuracy(label, input_type, path, true_pts, preds)

            baseline_by_input[input_type] = dict(
                path=path,
                model=model,
                x_sample=x_sample,
                true_pts=true_pts,
                true_y0=true_y0,
                preds_relaxed=preds,
                preds_discrete=preds,
                kind=args.baseline_kind,
                hw=hw_ballpark(model, x_sample, kind="tf"),
            )
        else:
            model_r, mname_r, it_r, true_pts_r, true_y0_r, preds_r, x_sample_r = eval_model_pred_classes(
                model_path=path,
                data_config=args.data_config,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                forward_mode="relaxed",
            )
            model_d, mname_d, it_d, true_pts_d, true_y0_d, preds_d, x_sample_d = eval_model_pred_classes(
                model_path=path,
                data_config=args.data_config,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                forward_mode="discrete",
            )

            if it_r != input_type:
                print(f"[WARN] Baseline key '{input_type}' but model reports input_type '{it_r}'. Using '{it_r}'.")
                input_type = it_r

            fallback = Path(path).parent.name
            label = label_for(path, fallback, name_overrides)
            record_balanced_accuracy(f"{label} (relaxed)", input_type, path, true_pts_r, preds_r)
            record_balanced_accuracy(f"{label} (discrete)", input_type, path, true_pts_d, preds_d)

            baseline_by_input[input_type] = dict(
                path=path,
                model=model_r,
                x_sample=x_sample_r,
                true_pts=true_pts_r,
                true_y0=true_y0_r,
                preds_relaxed=preds_r,
                preds_discrete=preds_d,
                kind="lgn",
                hw=hw_ballpark(model_r, x_sample_r, kind="lgn"),
            )

    if not args.candidates and args.combine_baselines:
        baseline_items = []
        for input_type, b in baseline_by_input.items():
            baseline_fallback = f"Baseline ({Path(b['path']).parent.name})"
            baseline_label = label_for(b["path"], baseline_fallback, name_overrides)
            baseline_items.append((baseline_label, b["true_pts"], b["preds_relaxed"]))

        outpath = outdir / "acceptance_baselines_combined.png"
        plot_acceptance_multiple_baselines(
            baseline_items=baseline_items,
            outpath=outpath,
            title="Acceptance vs true $p_T$",
            z=args.baseline_error_z,
        )

        print_balanced_accuracy_summary(metric_rows)
        write_balanced_accuracy_summary(outdir, metric_rows)
        print("[DONE] Wrote outputs to:", str(outdir))
        return

    best_relaxed: Dict[str, Dict] = {}
    best_discrete: Dict[str, Dict] = {}
    candidate_items_by_input: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}

    if args.candidates:
        for cand_path in args.candidates:
            if args.candidate_forward in ("both", "relaxed"):
                model_r, mname, it, true_pts_r, true_y0_r, preds_r, x_sample_r = eval_model_pred_classes(
                    model_path=cand_path,
                    data_config=args.data_config,
                    device=device,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    forward_mode="relaxed",
                )
                se_r, br_r = sig_eff_and_bkg_rej(true_pts_r, preds_r, pt_boundary=args.pt_boundary)
                sc_r = score_model(se_r, br_r)
                label_r = f"{checkpoint_label(cand_path)} (relaxed)"
                bal_r = record_balanced_accuracy(
                    label_r,
                    it,
                    cand_path,
                    true_pts_r,
                    preds_r,
                )
                candidate_items_by_input.setdefault(it, []).append((label_r, true_pts_r, preds_r))
                cur = best_relaxed.get(it)
                if (cur is None) or (sc_r > cur["score"]):
                    best_relaxed[it] = dict(
                        path=cand_path,
                        model=model_r,
                        x_sample=x_sample_r,
                        true_pts=true_pts_r,
                        true_y0=true_y0_r,
                        preds=preds_r,
                        score=sc_r,
                        balanced_acc=bal_r,
                        hw=hw_ballpark(model_r, x_sample_r, kind="lgn"),
                    )

            if args.candidate_forward in ("both", "discrete"):
                model_d, mname, it, true_pts_d, true_y0_d, preds_d, x_sample_d = eval_model_pred_classes(
                    model_path=cand_path,
                    data_config=args.data_config,
                    device=device,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    forward_mode="discrete",
                )
                se_d, br_d = sig_eff_and_bkg_rej(true_pts_d, preds_d, pt_boundary=args.pt_boundary)
                sc_d = score_model(se_d, br_d)
                label_d = f"{checkpoint_label(cand_path)} (discrete)"
                bal_d = record_balanced_accuracy(
                    label_d,
                    it,
                    cand_path,
                    true_pts_d,
                    preds_d,
                )
                candidate_items_by_input.setdefault(it, []).append((label_d, true_pts_d, preds_d))
                cur = best_discrete.get(it)
                if (cur is None) or (sc_d > cur["score"]):
                    best_discrete[it] = dict(
                        path=cand_path,
                        model=model_d,
                        x_sample=x_sample_d,
                        true_pts=true_pts_d,
                        true_y0=true_y0_d,
                        preds=preds_d,
                        score=sc_d,
                        balanced_acc=bal_d,
                        hw=hw_ballpark(model_d, x_sample_d, kind="lgn"),
                    )

    if args.candidates and not baseline_by_input:
        for input_type, candidate_items in candidate_items_by_input.items():
            outpath = outdir / f"acceptance_candidates_{sanitize_filename(input_type)}.png"
            plot_acceptance_multiple_baselines(
                baseline_items=candidate_items,
                outpath=outpath,
                title=f"Acceptance vs true $p_T$ — {input_type}",
                z=args.baseline_error_z,
            )

    for input_type, b in baseline_by_input.items():
        baseline_fallback = f"Baseline ({Path(b['path']).parent.name})"
        baseline_label = label_for(b["path"], baseline_fallback, name_overrides)

        if not args.candidates:
            outpath = outdir / f"acceptance_{sanitize_filename(input_type)}.png"
            plot_acceptance_baseline_only(
            input_type=input_type,
            pt_true_signed=b["true_pts"],
            baseline_relaxed=b["preds_relaxed"],
            baseline_discrete=b["preds_discrete"],
            outpath=outpath,
            baseline_label=baseline_label,
            baseline_kind=b["kind"],
            z=args.baseline_error_z,
            )

            out_y0_effacc = outdir / f"y0_eff_acc_{sanitize_filename(input_type)}.png"
            out_y0_hist = outdir / f"y0_hist_{sanitize_filename(input_type)}.png"

            plot_y0_eff_acc_paper_style(
                y0=b["true_y0"],
                pt_true_signed=b["true_pts"],
                preds=b["preds_relaxed"],
                outpath=out_y0_effacc,
                pt_boundary=args.pt_boundary,
            )
            plot_y0_histogram(
                y0=b["true_y0"],
                outpath=out_y0_hist,
            )
            continue

        if input_type not in best_relaxed or input_type not in best_discrete:
            print(f"[WARN] Missing best relaxed/discrete for input_type={input_type}; skipping.")
            continue

        br = best_relaxed[input_type]
        bd = best_discrete[input_type]

        best_rel_fallback = f"LGN ({Path(br['path']).parent.name})"
        best_dis_fallback = f"LGN ({Path(bd['path']).parent.name})"
        best_rel_label = label_for(br["path"], best_rel_fallback, name_overrides) + " (relaxed)"
        best_dis_label = label_for(bd["path"], best_dis_fallback, name_overrides) + " (discrete)"

        outpath = outdir / f"acceptance_{sanitize_filename(input_type)}.png"
        plot_acceptance(
            input_type=input_type,
            pt_true_signed=b["true_pts"],
            baseline_relaxed=b["preds_relaxed"],
            baseline_discrete=b["preds_discrete"],
            best_relaxed=br["preds"],
            best_discrete=bd["preds"],
            outpath=outpath,
            baseline_label=baseline_label,
            best_relaxed_label=best_rel_label,
            best_discrete_label=best_dis_label,
            baseline_kind=b["kind"],
            z=args.baseline_error_z,
        )

        out_y0_effacc = outdir / f"y0_eff_acc_{sanitize_filename(input_type)}.png"
        out_y0_hist = outdir / f"y0_hist_{sanitize_filename(input_type)}.png"

        plot_y0_eff_acc_paper_style(
            y0=bd["true_y0"],
            pt_true_signed=bd["true_pts"],
            preds=bd["preds"],
            outpath=out_y0_effacc,
            pt_boundary=args.pt_boundary,
        )
        plot_y0_histogram(
            y0=bd["true_y0"],
            outpath=out_y0_hist,
        )

    print_balanced_accuracy_summary(metric_rows)
    write_balanced_accuracy_summary(outdir, metric_rows)
    print("[DONE] Wrote outputs to:", str(outdir))


if __name__ == "__main__":
    main()
