"""Binary physics metrics for PV-signal versus BIB classification."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


TARGET_SIGNAL_EFFICIENCIES = (0.90, 0.95, 0.99)


def confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= threshold
    signal = labels == 1
    bib = labels == 0
    tp = int(np.count_nonzero(predictions & signal))
    fn = int(np.count_nonzero(~predictions & signal))
    fp = int(np.count_nonzero(predictions & bib))
    tn = int(np.count_nonzero(~predictions & bib))
    signal_efficiency = tp / (tp + fn) if tp + fn else float("nan")
    bib_acceptance = fp / (fp + tn) if fp + tn else float("nan")
    bib_rejection = None if bib_acceptance == 0 else 1.0 / bib_acceptance
    bib_rejection_fraction = 1.0 - bib_acceptance
    balanced_accuracy = 0.5 * (
        signal_efficiency + (tn / (tn + fp) if tn + fp else float("nan"))
    )
    return {
        "threshold": float(threshold),
        "true_signal": tp,
        "missed_signal": fn,
        "accepted_bib": fp,
        "rejected_bib": tn,
        "signal_efficiency": signal_efficiency,
        "bib_acceptance": bib_acceptance,
        "bib_rejection": bib_rejection,
        "bib_rejection_fraction": bib_rejection_fraction,
        "balanced_accuracy": balanced_accuracy,
    }


def threshold_for_signal_efficiency(
    labels: np.ndarray, scores: np.ndarray, target: float
) -> float:
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    if not len(positive):
        raise ValueError("Cannot set an operating point without signal examples")
    # Selecting the kth-lowest positive score makes at least target of the
    # validation signal pass score >= threshold (ties may increase efficiency).
    rank = max(0, min(len(positive) - 1, len(positive) - math.ceil(target * len(positive))))
    return float(np.partition(positive, rank)[rank])


def evaluate_binary_scores(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    targets: Iterable[float] = TARGET_SIGNAL_EFFICIENCIES,
) -> dict:
    if set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Validation sample must contain both classes")
    if set(np.unique(test_labels)) != {0, 1}:
        raise ValueError("Test sample must contain both classes")
    operating_points = {}
    for target in targets:
        threshold = threshold_for_signal_efficiency(
            validation_labels, validation_scores, target
        )
        key = f"signal_efficiency_{int(round(100 * target))}pct"
        values = confusion_metrics(test_labels, test_scores, threshold)
        values["target_validation_signal_efficiency"] = float(target)
        values["validation"] = confusion_metrics(
            validation_labels, validation_scores, threshold
        )
        operating_points[key] = values
    return {
        "n_examples": int(len(test_labels)),
        "n_signal": int(np.count_nonzero(test_labels == 1)),
        "n_bib": int(np.count_nonzero(test_labels == 0)),
        "argmax": confusion_metrics(test_labels, test_scores, 0.0),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "average_precision": float(average_precision_score(test_labels, test_scores)),
        "validation_derived_operating_points": operating_points,
    }


def roc_rows(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    rows = []
    for fpr, tpr, threshold in zip(false_positive_rate, true_positive_rate, thresholds):
        rows.append(
            {
                "score_threshold": float(threshold),
                "signal_efficiency": float(tpr),
                "bib_acceptance": float(fpr),
                "bib_rejection": float("inf") if fpr == 0 else float(1.0 / fpr),
                "bib_rejection_fraction": float(1.0 - fpr),
            }
        )
    return rows
