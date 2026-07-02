#!/usr/bin/env python3
"""Combine the three per-model bit-flip studies into one paper plot."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--train-flip-probability", type=float, default=1e-3)
    args = parser.parse_args()

    rows_by_model = {}
    for model_alias in MODEL_ORDER:
        path = args.root / model_alias / "bit_flip_results.csv"
        if not path.is_file():
            raise SystemExit(f"Missing completed result: {path}")
        with path.open(newline="") as handle:
            rows_by_model[model_alias] = list(csv.DictReader(handle))

    plot_combined_models(
        rows_by_model,
        args.train_flip_probability,
        args.root / "bit_flip_robustness_all_models",
    )
    print(f"[DONE] Combined plots written under {args.root}")


if __name__ == "__main__":
    main()
