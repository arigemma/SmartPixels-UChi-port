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
import hist


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


def load_recon_flat(path: str, nrows: Optional[int]) -> np.ndarray:
    cols = [str(i) for i in range(20 * 13 * 21)]
    df = pd.read_parquet(path, columns=cols)
    if nrows is not None:
        df = df.iloc[:nrows]
    x = df.to_numpy(dtype=np.float32, copy=True)
    return x


def load_labels(path: str, nrows: Optional[int]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_parquet(path)
    if nrows is not None:
        df = df.iloc[:nrows]
    return df.to_numpy(dtype=np.float32, copy=True), list(df.columns)


def make_features(
    recon_flat: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    pt_idx: int,
    y0_idx: int,
    thr_frac: float,
    slice_index_4000ps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = recon_flat.shape[0]
    recon = recon_flat.reshape(n, 20, 13, 21)

    y0 = labels[:, y0_idx].astype(np.float32, copy=False)

    xy_4000 = recon[:, slice_index_4000ps, :, :]
    yprof_4000 = xy_4000.sum(axis=2)
    a = np.abs(yprof_4000)
    vmax = a.max(axis=1)
    thr = (vmax * float(thr_frac)).reshape(-1, 1)
    ysize = (a > thr).sum(axis=1).astype(np.float32)

    f1 = np.stack([y0, ysize], axis=1).astype(np.float32)
    f2 = np.concatenate([y0.reshape(-1, 1), yprof_4000.astype(np.float32)], axis=1).astype(np.float32)

    t8 = recon[:, :8, :, :].sum(axis=3)
    f3 = np.concatenate([y0.reshape(-1, 1), t8.reshape(n, -1).astype(np.float32)], axis=1).astype(np.float32)

    pt_signed = labels[:, pt_idx].astype(np.float32, copy=False)
    return f1, f2, f3


@torch.no_grad()
def infer(model: torch.nn.Module, x_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    n = x_np.shape[0]
    for i in tqdm(range(0, n, batch_size)):
        xb = torch.from_numpy(x_np[i:i + batch_size]).to(device)
        out = model(xb).detach().cpu().numpy()
        outs.append(out)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model1", required=True)
    ap.add_argument("--model2", required=True)
    ap.add_argument("--model3", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--outdir", action=CreateFolder, default="results/eval_towards")
    ap.add_argument("--pt-boundary", type=float, default=0.2)
    ap.add_argument("--thr-frac", type=float, default=0.01)
    ap.add_argument("--slice-index", type=int, default=19)
    ap.add_argument("--max-events-per-file", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--trigger-only-highpt", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    outdir = Path(args.outdir)

    def load_model(path: str) -> torch.nn.Module:
        return torch.load(path, map_location=device, weights_only=False)

    m1 = load_model(args.model1).to(device)
    m2 = load_model(args.model2).to(device)
    m3 = load_model(args.model3).to(device)

    data = {}
    for model in [m1, m2, m3]:

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

        pt_idx = label_names.index("pt")
        y0_idx = label_names.index("y-entry")

        true_pts, logits = [], []
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(test_loader)):
                x = x.to(device)
                logits.append(model(x))  # logits
                true_pts.append(y[:, pt_idx])
                # preds.append(pred.argmax(dim=1))  # Get the class with highest probability
                if i >= 100:  # Limit to first 100 batches for quick evaluation
                    break

        true_pts = np.concatenate(true_pts, axis=0)
        logits = np.concatenate(logits, axis=0)
        probs = softmax_np(logits)
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

    p_high = {
        "towards-model-1": data["towards-model-1"]["probs"][:, 2].astype(np.float32),
        "towards-model-2": data["towards-model-2"]["probs"][:, 2].astype(np.float32),
        "towards-model-3": data["towards-model-3"]["probs"][:, 2].astype(np.float32),
    }

    cm_by_model = {
        "towards-model-1": confusion_matrix_3(data["towards-model-1"]["true_classes"], data["towards-model-1"]["pred_classes"]),
        "towards-model-2": confusion_matrix_3(data["towards-model-2"]["true_classes"], data["towards-model-2"]["pred_classes"]),
        "towards-model-3": confusion_matrix_3(data["towards-model-3"]["true_classes"], data["towards-model-3"]["pred_classes"]),
    }



    rocs = {}
    for name, ph in p_high.items():
        fpr, tpr = roc_curve_binary(high_true, ph)
        auc = auc_trapz(fpr, tpr)
        rocs[name] = (fpr, tpr, auc)

    for name, cm in cm_by_model.items():
        acc = float(np.trace(cm) / max(cm.sum(), 1))
        print(f"[METRIC] {name} accuracy: {acc:.6f}")
    for name, (_, _, auc) in rocs.items():
        print(f"[METRIC] {name} high-vs-low AUC: {auc:.6f}")

    plot_confusion(cm_by_model, outdir / "confusion_matrices.png")
    plot_roc(rocs, outdir / "roc_highpt.png")
    plot_prob_vs_pt(true_pts, p_high, outdir / "pt_probability_vs_true_pt.png", nbins=40)

    print("[DONE] Wrote outputs to:", str(outdir))        

    # for ip, lp in zip(test_image_paths, test_label_paths):
    #     recon_flat = load_recon_flat(ip, nrows=args.max_events_per_file)
    #     y_raw, _ = load_labels(lp, nrows=args.max_events_per_file)

    #     y = y_raw
    #     if scaler_y is not None:
    #         try:
    #             y = scaler_y.inverse_transform(y_raw)
    #         except Exception:
    #             y = y_raw

    #     f1, f2, f3 = make_features(
    #         recon_flat=recon_flat,
    #         labels=y,
    #         label_names=label_names,
    #         pt_idx=pt_idx,
    #         y0_idx=y0_idx,
    #         thr_frac=args.thr_frac,
    #         slice_index_4000ps=args.slice_index,
    #     )

    #     pt_signed = y[:, pt_idx].astype(np.float32, copy=False)

    #     X1_all.append(f1)
    #     X2_all.append(f2)
    #     X3_all.append(f3)
    #     pt_all.append(pt_signed)

    # X1 = np.concatenate(X1_all, axis=0)
    # X2 = np.concatenate(X2_all, axis=0)
    # X3 = np.concatenate(X3_all, axis=0)
    # pt_true_signed = np.concatenate(pt_all, axis=0)

    # true_class = true_class_from_signed_pt(pt_true_signed, pt_boundary=args.pt_boundary)
    # high_true = (true_class == 2).astype(np.int32)

    # if args.trigger_only_highpt:
    #     keep = np.abs(pt_true_signed) > 2.0
    #     X1 = X1[keep]
    #     X2 = X2[keep]
    #     X3 = X3[keep]
    #     pt_true_signed = pt_true_signed[keep]
    #     true_class = true_class[keep]
    #     high_true = high_true[keep]

    # def load_model(path: str) -> torch.nn.Module:
    #     return torch.load(path, map_location=device, weights_only=False)

    # m1 = load_model(args.model1).to(device)
    # m2 = load_model(args.model2).to(device)
    # m3 = load_model(args.model3).to(device)

    # logits1 = infer(m1, X1, device=device, batch_size=args.batch_size)
    # logits2 = infer(m2, X2, device=device, batch_size=args.batch_size)
    # logits3 = infer(m3, X3, device=device, batch_size=args.batch_size)

    # probs1 = logits1 if (logits1.ndim == 2 and logits1.shape[1] == 3) else softmax_np(logits1)
    # probs2 = logits2 if (logits2.ndim == 2 and logits2.shape[1] == 3) else softmax_np(logits2)
    # probs3 = logits3 if (logits3.ndim == 2 and logits3.shape[1] == 3) else softmax_np(logits3)

    # pred1 = np.argmax(probs1, axis=1).astype(np.int64)
    # pred2 = np.argmax(probs2, axis=1).astype(np.int64)
    # pred3 = np.argmax(probs3, axis=1).astype(np.int64)

    # p_high = {
    #     "towards-model-1": probs1[:, 2].astype(np.float32),
    #     "towards-model-2": probs2[:, 2].astype(np.float32),
    #     "towards-model-3": probs3[:, 2].astype(np.float32),
    # }

    # cm_by_model = {
    #     "towards-model-1": confusion_matrix_3(true_class, pred1),
    #     "towards-model-2": confusion_matrix_3(true_class, pred2),
    #     "towards-model-3": confusion_matrix_3(true_class, pred3),
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
    # plot_prob_vs_pt(pt_true_signed, p_high, outdir / "pt_probability_vs_true_pt.png", nbins=40)

    # print("[DONE] Wrote outputs to:", str(outdir))


if __name__ == "__main__":
    main()
