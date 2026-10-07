"""CLI for one-time streaming projection of the MAIA Parquet dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .data import DEFAULT_TIMING_COLUMN, prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MAIA_DATA_ROOT",
                "/scratch/gpfs/IOJALVO/mb7126/SmartPixels/22016562",
            )
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--timing-column", default=DEFAULT_TIMING_COLUMN)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--thresholds", type=int, default=100)
    parser.add_argument("--threshold-sample-rows", type=int, default=50_000)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    metadata = prepare_cache(
        args.data_root,
        args.cache_dir,
        timing_column=args.timing_column,
        batch_size=args.batch_size,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        n_thresholds=args.thresholds,
        threshold_sample_rows=args.threshold_sample_rows,
        overwrite=args.overwrite_cache,
    )
    print(f"[DONE] Prepared {args.cache_dir}")
    print(metadata["task"])


if __name__ == "__main__":
    main()
