"""Structural and lightweight physics audit for the prepared MAIA cache."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .campaign import N_CONV_INPUT_FEATURES, N_DENSE_INPUT_FEATURES, N_THRESHOLDS
from .data import CACHE_VERSION, INPUT_SCHEMA, SOURCES, SOURCE_LABEL


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sample(cache_dir: Path, split: str, maximum_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    pieces = []
    labels = []
    quotas = {
        "signal": maximum_per_class,
        "bib_mp": maximum_per_class // 2,
        "bib_mm": maximum_per_class - maximum_per_class // 2,
    }
    for source in SOURCES:
        indices = np.load(cache_dir / source / f"{split}_indices.npy", mmap_mode="r")
        take = min(quotas[source], len(indices))
        positions = np.sort(rng.choice(len(indices), take, replace=False))
        rows = np.asarray(indices[positions], dtype=np.int64)
        values = np.load(cache_dir / source / "dense_inputs.npy", mmap_mode="r")
        pieces.append(np.asarray(values[rows], dtype=np.float32))
        labels.append(np.full(take, SOURCE_LABEL[source], dtype=np.int64))
    return np.concatenate(pieces), np.concatenate(labels)


def audit_cache(
    cache_dir: Path,
    *,
    maximum_per_class: int = 20_000,
    minimum_baseline_auc: float = 0.60,
) -> dict:
    complete = cache_dir / "preparation_complete.json"
    with complete.open() as handle:
        metadata = json.load(handle)
    if metadata.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("Prepared cache version mismatch")
    if metadata.get("input_schema") != INPUT_SCHEMA:
        raise RuntimeError("Prepared cache input-schema mismatch")

    source_rows = {}
    for source in SOURCES:
        dense = np.load(cache_dir / source / "dense_inputs.npy", mmap_mode="r")
        conv = np.load(cache_dir / source / "conv_inputs.npy", mmap_mode="r")
        if dense.ndim != 2 or dense.shape[1] != N_DENSE_INPUT_FEATURES:
            raise RuntimeError(f"{source}: invalid dense shape {dense.shape}")
        if conv.ndim != 2 or conv.shape[1] != N_CONV_INPUT_FEATURES:
            raise RuntimeError(f"{source}: invalid conv shape {conv.shape}")
        if len(dense) != len(conv):
            raise RuntimeError(f"{source}: dense/conv row-count mismatch")
        probe = np.linspace(0, len(dense) - 1, min(2048, len(dense)), dtype=np.int64)
        if not np.isfinite(np.asarray(dense[probe])).all():
            raise RuntimeError(f"{source}: non-finite dense inputs")
        if not np.isfinite(np.asarray(conv[probe])).all():
            raise RuntimeError(f"{source}: non-finite conv inputs")
        source_rows[source] = len(dense)

    dense_thresholds = np.load(cache_dir / "dense_thresholds.npy")
    conv_thresholds = np.load(cache_dir / "conv_thresholds.npy")
    expected_shapes = {
        "dense": (N_DENSE_INPUT_FEATURES, N_THRESHOLDS),
        "conv": (N_CONV_INPUT_FEATURES, N_THRESHOLDS),
    }
    observed_shapes = {
        "dense": tuple(dense_thresholds.shape),
        "conv": tuple(conv_thresholds.shape),
    }
    if observed_shapes != expected_shapes:
        raise RuntimeError(
            f"Threshold shapes are wrong: {observed_shapes}; expected {expected_shapes}"
        )
    for name, values in (
        ("dense", dense_thresholds),
        ("conv", conv_thresholds),
    ):
        if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) < 0):
            raise RuntimeError(f"{name}: thresholds are non-finite or unordered")

    train_x, train_y = _sample(cache_dir, "train", maximum_per_class, 2718)
    validation_x, validation_y = _sample(
        cache_dir, "validation", maximum_per_class, 3141
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=60,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier.fit(train_x, train_y)
    scores = classifier.predict_proba(validation_x)[:, 1]
    baseline_auc = float(roc_auc_score(validation_y, scores))
    baseline_balanced_accuracy = float(
        balanced_accuracy_score(validation_y, scores >= 0.5)
    )
    if baseline_auc < minimum_baseline_auc:
        raise RuntimeError(
            "Corrected MAIA features fail the lightweight baseline audit: "
            f"validation ROC AUC={baseline_auc:.4f} < {minimum_baseline_auc:.4f}"
        )

    result = {
        "passed": True,
        "cache_version": CACHE_VERSION,
        "input_schema": INPUT_SCHEMA,
        "source_rows": source_rows,
        "threshold_shapes": {
            key: list(value) for key, value in observed_shapes.items()
        },
        "baseline": {
            "model": "HistGradientBoostingClassifier (diagnostic only)",
            "training_examples": int(len(train_y)),
            "validation_examples": int(len(validation_y)),
            "validation_roc_auc": baseline_auc,
            "validation_balanced_accuracy": baseline_balanced_accuracy,
            "minimum_required_roc_auc": minimum_baseline_auc,
        },
    }
    _atomic_json(cache_dir / "cache_audit_complete.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--maximum-per-class", type=int, default=20_000)
    parser.add_argument("--minimum-baseline-auc", type=float, default=0.60)
    args = parser.parse_args()
    result = audit_cache(
        args.cache_dir,
        maximum_per_class=args.maximum_per_class,
        minimum_baseline_auc=args.minimum_baseline_auc,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
