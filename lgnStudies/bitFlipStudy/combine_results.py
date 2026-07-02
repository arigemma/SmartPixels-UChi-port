#!/usr/bin/env python3
"""Combine the three exact-count bit-flip studies into paper plots."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.plots import plot_combined_models


MODEL_ORDER = (
    "s06_p760M_tau40",
    "s08_p1202M_tau40",
    "s09_p1460M_tau40",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows_by_model = {}
    for model_alias in MODEL_ORDER:
        path = args.root / model_alias / "bit_flip_results.csv"
        if not path.is_file():
            raise SystemExit(f"Missing completed result: {path}")
        with path.open(newline="") as handle:
            rows_by_model[model_alias] = list(csv.DictReader(handle))
        metadata_path = args.root / model_alias / "bit_flip_metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"Missing study metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete"):
            raise SystemExit(f"Study is marked incomplete: {metadata_path}")

        expected_counts = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
        observed_counts = {
            int(row["flip_count_per_event"]) for row in rows_by_model[model_alias]
        }
        if observed_counts != expected_counts:
            raise SystemExit(
                f"Incomplete result for {model_alias}: counts={sorted(observed_counts)}"
            )
        rows_per_count = Counter(
            int(row["flip_count_per_event"])
            for row in rows_by_model[model_alias]
        )
        expected_rows = {0: 1, **{count: 5 for count in expected_counts if count}}
        if dict(rows_per_count) != expected_rows:
            raise SystemExit(
                f"Wrong repeat counts for {model_alias}: {dict(rows_per_count)}"
            )

    plot_combined_models(
        rows_by_model,
        args.root / "bit_flips_per_event_all_models",
        x_mode="count",
    )
    plot_combined_models(
        rows_by_model,
        args.root / "bit_flip_percentage_all_models",
        x_mode="percentage",
    )
    print(f"[DONE] Combined plots written under {args.root}")


if __name__ == "__main__":
    main()
