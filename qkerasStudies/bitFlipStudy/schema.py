"""CSV schema shared by QKeras bit-flip evaluation and combine jobs."""

from __future__ import annotations

import csv
from pathlib import Path


CSV_FIELDS = (
    "model_alias",
    "registry_name",
    "flip_count_per_event",
    "total_fault_sites_per_event",
    "equivalent_fault_fraction",
    "equivalent_fault_percentage",
    "repeat",
    "fault_seed",
    "balanced_accuracy",
    "prediction_change_fraction",
    "balanced_repair_gain",
    "balanced_damage_loss",
    "repair_fraction",
    "damage_fraction",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "n_class_0",
    "n_class_1",
    "n_class_2",
    "repairs_class_0",
    "repairs_class_1",
    "repairs_class_2",
    "damages_class_0",
    "damages_class_1",
    "damages_class_2",
    *(f"confusion_{truth}_{prediction}" for truth in range(3) for prediction in range(3)),
)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
