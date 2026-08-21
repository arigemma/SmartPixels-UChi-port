"""Backend-neutral helpers for HGQ2 training and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


CLASS_NAMES = ("positive low-pT", "negative low-pT", "high-pT")


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def confusion_from_predictions(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return np.bincount(3 * truth + predictions, minlength=9).reshape(3, 3)


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    totals = confusion.sum(axis=1)
    if np.any(totals == 0):
        raise RuntimeError(f"Cannot compute balanced accuracy with empty classes: {totals.tolist()}.")
    recalls = np.diag(confusion).astype(np.float64) / totals
    return {
        "balanced_accuracy": float(recalls.mean()),
        "recalls": [float(value) for value in recalls],
        "class_totals": [int(value) for value in totals],
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
