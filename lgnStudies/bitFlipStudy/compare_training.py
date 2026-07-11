#!/usr/bin/env python3
"""Compare inference-only and fault-aware bit-flip robustness CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lgnStudies.bitFlipStudy.campaign import MODELS
from lgnStudies.bitFlipStudy.combine_results import read_csv
from lgnStudies.bitFlipStudy.plots import plot_fault_aware_training_comparison


DEFAULT_INFERENCE_ROOT = Path(
    "results/SLURM/2026_07_04_model2_bit_flip_5h_20k_dense_sweep"
)
DEFAULT_FAULT_AWARE_ROOT = Path(
    "results/SLURM/2026_07_05_model2_fault_aware_training_1pct_5h_20k"
)
MODEL_ORDER = tuple(model_alias for model_alias, _ in MODELS)


def load_rows_by_model(root: Path) -> dict[str, list[dict]]:
    rows_by_model = {}
    for model_alias in MODEL_ORDER:
        csv_path = root / model_alias / "bit_flip_results.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing merged bit-flip CSV: {csv_path}")
        rows_by_model[model_alias] = read_csv(csv_path)
    return rows_by_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one balanced-accuracy comparison per model from existing "
            "inference-only and fault-aware bit-flip CSVs."
        )
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=DEFAULT_INFERENCE_ROOT,
        help="Root containing inference-only MODEL/bit_flip_results.csv files.",
    )
    parser.add_argument(
        "--fault-aware-root",
        type=Path,
        default=DEFAULT_FAULT_AWARE_ROOT,
        help="Root containing fault-aware MODEL/bit_flip_results.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory for comparison plots; defaults to --fault-aware-root.",
    )
    parser.add_argument(
        "--title-context",
        help="Optional provenance line appended to each plot title.",
    )
    args = parser.parse_args()

    output_root = args.output_root or args.fault_aware_root
    inference_rows = load_rows_by_model(args.inference_root)
    fault_aware_rows = load_rows_by_model(args.fault_aware_root)
    plot_fault_aware_training_comparison(
        inference_rows,
        fault_aware_rows,
        output_root,
        x_mode="percentage",
        title_context=args.title_context,
    )
    print(f"[DONE] Wrote fault-aware comparison plots to {output_root}")


if __name__ == "__main__":
    main()
