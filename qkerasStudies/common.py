"""Shared helpers for QKeras training and robustness evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np


CLASS_NAMES = ("positive low-pT", "negative low-pT", "high-pT")


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def configure_device(device: str) -> None:
    import tensorflow as tf

    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
    elif device == "cuda" and not tf.config.list_physical_devices("GPU"):
        print(
            "[WARN] --device cuda requested, but TensorFlow sees no GPU. "
            "TensorFlow will use CPU.",
            flush=True,
        )


def get_class_weights(loader) -> np.ndarray:
    labels = loader.dataset.get_all_class_labels()
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError(f"Training split class counts are invalid: {counts.tolist()}.")
    return len(labels) / (len(counts) * counts)


def compute_input_quantization_range(loader, max_batches: int = 64) -> dict[str, float]:
    minimum = np.inf
    maximum = -np.inf
    for batch_index, (inputs, _labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        values = to_numpy(inputs).astype(np.float32, copy=False)
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise RuntimeError("Could not calibrate input quantization range.")
    if np.isclose(minimum, maximum):
        maximum = minimum + 1.0
    margin = 0.01 * (maximum - minimum)
    return {
        "minimum": float(minimum - margin),
        "maximum": float(maximum + margin),
    }


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    totals = confusion.sum(axis=1)
    if np.any(totals == 0):
        raise RuntimeError(f"Missing test class; totals={totals.tolist()}.")
    recalls = np.diag(confusion).astype(np.float64) / totals
    return {
        "balanced_accuracy": float(recalls.mean()),
        "recalls": [float(value) for value in recalls],
        "class_totals": [int(value) for value in totals],
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def confusion_from_predictions(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return np.bincount(3 * truth + predictions, minlength=9).reshape(3, 3)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        paths: Iterable[Path] = (path,)
        root = path.parent
    elif path.is_dir():
        paths = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
    else:
        raise FileNotFoundError(path)

    for item in paths:
        digest.update(str(item.relative_to(root)).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
