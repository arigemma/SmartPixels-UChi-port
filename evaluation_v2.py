#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import joblib
import yaml
import matplotlib.pyplot as plt

from tqdm import tqdm

from dataloader import create_dataloaders
from utils import get_device, CreateFolder
from models import registry
# import hist


LABEL_ORDER = [
    "x-entry", "y-entry", "z-entry",
    "n_x", "n_y", "n_z",
    "number_eh_pairs", "y-local",
    "pt",
    "cotAlpha", "cotBeta",
    "y-midplane", "x-midplane",
]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_label_index(name: str, label_names: List[str]) -> int:
    if name in label_names:
        return label_names.index(name)
    if name in LABEL_ORDER:
        for n in LABEL_ORDER:
            if n in label_names and n == name:
                return label_names.index(n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def true_class_from_signed_pt(pt_signed: np.ndarray, pt_boundary: float) -> np.ndarray:
    c = np.empty_like(pt_signed, dtype=np.int64)
    c[(pt_signed >= 0.0) & (pt_signed <= pt_boundary)] = 0
    c[(pt_signed < 0.0) & (pt_signed >= -pt_boundary)] = 1
    c[(pt_signed > pt_boundary) | (pt_signed < -pt_boundary)] = 2
    return c


def confusion_matrix_3(true_c: np.ndarray, pred_c: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true_c, pred_c):
        cm[int(t), int(p)] += 1
    return cm


def roc_curve_binary(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y = y_true[order].astype(np.int32)
    s = score[order]
    P = max(int(y.sum()), 1)
    N = max(int((1 - y).sum()), 1)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / N
    keep = np.r_[True, np.diff(s) != 0]
    return fpr[keep], tpr[keep]


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def savefig(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_confusion(cm_by_model: Dict[str, np.ndarray], outpath: Path):
    n = len(cm_by_model)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_by_model.items()):
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["pos low", "neg low", "high"])
        ax.set_yticklabels(["pos low", "neg low", "high"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, outpath)


def plot_roc(rocs: Dict[str, Tuple[np.ndarray, np.ndarray, float]], outpath: Path):
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.4), constrained_layout=True)
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("High-$p_T$ ROC")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.2)
    savefig(fig, outpath)


def plot_prob_vs_pt(pt_true_signed: np.ndarray, p_high_by_model: Dict[str, np.ndarray], outpath: Path,
                    xmin: float = -5.0, xmax: float = 5.0, nbins: int = 40):
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)

    for name, p_high in p_high_by_model.items():
        mean = np.full(nbins, np.nan, dtype=np.float64)
        err = np.full(nbins, np.nan, dtype=np.float64)
        for i in range(nbins):
            m = (pt_true_signed >= edges[i]) & (pt_true_signed < edges[i + 1])
            n = int(m.sum())
            if n == 0:
                continue
            ph = float(np.mean(p_high[m]))
            mean[i] = ph
            err[i] = np.sqrt(ph * (1.0 - ph) / n)
        ax.errorbar(centers, mean, yerr=err, fmt="o", markersize=3.0, capsize=2.0, label=name)

    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(r"$P(\mathrm{high}\ p_T)$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, framealpha=0.2)
    savefig(fig, outpath)



def plot_acceptance_vs_pt(pt_true_signed: np.ndarray, pred_dict: dict[str, np.ndarray], outpath: Path):
    
    # use variable bin edges
    edges = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    edges = np.array(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    for model_name, preds in pred_dict.items():
        assert pt_true_signed.shape == preds.shape, "pt_true_signed and preds must have the same shape"
        high_pt_idx = np.abs(pt_true_signed) >= 2.0
        high_pt_n = int(high_pt_idx.sum())
        high_pt_preds = preds[high_pt_idx]
        sig_eff = float((high_pt_preds == 2).sum()) / max(high_pt_n, 1)
        low_pt_idx = np.abs(pt_true_signed) < 2.0
        low_pt_n = int(low_pt_idx.sum())
        low_pt_preds = preds[low_pt_idx]
        bkg_rejection = float((low_pt_preds != 2).sum()) / max(low_pt_n, 1)
        acceptances = []
        for i_bin in range(len(edges) - 1):
            idx = (pt_true_signed >= edges[i_bin]) & (pt_true_signed < edges[i_bin + 1])
            n = int(idx.sum())
            if n == 0:
                acceptances = np.nan
            else:
                acceptance = float((preds[idx] == 2).sum()) / n
            print(f"Bin {i_bin}: true pt in [{edges[i_bin]:.2f}, {edges[i_bin + 1]:.2f}), acceptance: {acceptance:.4f} ({(preds[idx] == 2).sum()}/{n})")
            acceptances.append(acceptance)
        plt.plot(centers, acceptances, label=f"{model_name} (sig eff: {sig_eff:.4f}, bkg rej: {bkg_rejection:.4f})", marker="o")
    # add horizontal lines and y-ticks at 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.35)

    plt.xlabel(r"true $p_T$ [GeV]")
    # plt.ylabel(r"acceptance as \textit{high-$p_T$}")
    plt.ylabel(r"acceptance as $\it{high}$-$p_T$")
    plt.legend(loc="best", fontsize=8, framealpha=0.2)
    plt.savefig(outpath, dpi=160)

    # fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model-paths", nargs='+', help="paths to the trained models")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--outdir", action=CreateFolder, default="results/eval_towards")
    ap.add_argument("--pt-boundary", type=float, default=2.0)
    ap.add_argument(
    "--forward",
    choices=["relaxed", "discrete"],
    default="relaxed",
    help="Which forward to use for TorchLogix models. "
         "'relaxed' uses the soft/relaxation path; 'discrete' uses hard/thresholded path.",
    )

    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = get_device(args.device)
    outdir = Path(args.outdir)

    def load_model(path: str) -> torch.nn.Module:
        return torch.load(path, map_location=device, weights_only=False)

    data = {}
    # for model in [m1, m2, m3]:
    for model_path in args.model_paths:

        print(f"\n[INFO] Loading model from {model_path}...")

        model = load_model(model_path).to(device)
        if args.forward == "discrete":
            model.eval()
        else:
            model.train()

        model_name, model_cfg = registry.class_to_config(type(model))

        input_type = model_cfg["input_type"]

        print(f"\n[INFO] Evaluating {model_name} with input_type={input_type}...")

        _, _, test_loader = create_dataloaders(
            config_path=args.data_config,
            batch_size=128,
            shuffle=False,
            input_type=input_type,
            target_type="raw",  # return raw labels for most flexibility in evaluation
            balance_classes=False,  # no balancing for evaluation
            val_size=0.0,
        )

        label_names = test_loader.dataset.label_names

        print(f"label_names: {label_names}")

        pt_idx = label_names.index("pt")

        true_pts, logits = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(test_loader)):
            # for x, y in tqdm(test_loader):
                x = x.to(device)
                logits.append(model(x))  # logits
                true_pts.append(y[:, pt_idx])
                # preds.append(pred.argmax(dim=1))  # Get the class with highest probability
                if i >= 1000:  # Limit to first 100 batches for quick evaluation
                    break

        true_pts = np.concatenate(true_pts, axis=0)
        # logits = np.concatenate(logits, axis=0)
        logits = torch.cat(logits, dim=0)

        if args.forward == "discrete":
            pred_classes = logits.argmax(dim=1).cpu().numpy().astype(np.int8)
            probs = None
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = np.argmax(probs, axis=1).astype(np.int8)
        true_classes = true_class_from_signed_pt(true_pts, pt_boundary=args.pt_boundary)
        high_true = (true_classes == 2).astype(np.int32)

        data[model_name] = {
            "true_pts": true_pts,
            "logits": logits,
            "probs": probs,
            "pred_classes": pred_classes,
            "true_classes": true_classes,
            "high_true": high_true,
        }

    # p_high = {
    #     "towards-model-1": data["towards-model-1"]["probs"][:, 2].astype(np.float32),
    #     "towards-model-2": data["towards-model-2"]["probs"][:, 2].astype(np.float32),
    #     "towards-model-3": data["towards-model-3"]["probs"][:, 2].astype(np.float32),
    # }

    # cm_by_model = {
    #     "towards-model-1": confusion_matrix_3(data["towards-model-1"]["true_classes"], data["towards-model-1"]["pred_classes"]),
    #     "towards-model-2": confusion_matrix_3(data["towards-model-2"]["true_classes"], data["towards-model-2"]["pred_classes"]),
    #     "towards-model-3": confusion_matrix_3(data["towards-model-3"]["true_classes"], data["towards-model-3"]["pred_classes"]),
    # }

    # rocs = {}
    # for name, ph in p_high.items():
    #     fpr, tpr = roc_curve_binary(high_true, ph)
    #     auc = auc_trapz(fpr, tpr)
    #     rocs[name] = (fpr, tpr, auc)

    # for name, cm in cm_by_model.items():
    #     acc = float(np.trace(cm) / max(cm.sum(), 1))
    #     print(f"[METRIC] {name} accuracy: {acc:.6f}")
    # for name, (_, _, auc) in rocs.items():
    #     print(f"[METRIC] {name} high-vs-low AUC: {auc:.6f}")

    # plot_confusion(cm_by_model, outdir / "confusion_matrices.png")
    # plot_roc(rocs, outdir / "roc_highpt.png")
    # plot_prob_vs_pt(true_pts, p_high, outdir / "pt_confidence_vs_true_pt.png", nbins=40)

    plot_acceptance_vs_pt(true_pts, {name: data[name]["pred_classes"] for name in data.keys()}, outdir / "acceptance_vs_true_pt.png")

    print("[DONE] Wrote outputs to:", str(outdir))        


if __name__ == "__main__":
    main()
