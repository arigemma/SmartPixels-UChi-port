"""Streaming preparation and memory-mapped access for Zenodo record 22016562.

The class label is intentionally derived from the source collection rather
than any truth column:

* ``signal`` -> 1 (PV/collision muon)
* ``bib_mp`` and ``bib_mm`` -> 0 (beam-induced background)

Truth timing is retained only as evaluation metadata. The only label-table
column used as a model input is ``y-local`` (the established ``y0`` feature).
It is not a class label.
"""

from __future__ import annotations

import bisect
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Subset

from .campaign import (
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    LAST_FRAME,
    N_SLICES,
    N_TIME,
    N_X,
    N_Y,
)


CACHE_VERSION = 3
INPUT_SCHEMA = "maia_model2_yprofile_model3_yprofile_timing8_v3"
SOURCES = ("signal", "bib_mp", "bib_mm")
SOURCE_LABEL = {"signal": 1, "bib_mp": 0, "bib_mm": 0}
N_RECON_COLUMNS = N_TIME * N_Y * N_X
DEFAULT_TIMING_COLUMN = "adjusted_hit_time"
Y0_COLUMN = "y-local"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def locate_source_files(data_root: Path, source: str) -> tuple[Path, Path]:
    candidates = (
        (
            data_root / "raw" / f"{source}_recon3D_combined.parquet",
            data_root / "raw" / f"{source}_labels_combined.parquet",
        ),
        (
            data_root / source / "recon3D" / "recon3D_combined.parquet",
            data_root / source / "labels" / "labels_combined.parquet",
        ),
    )
    for recon, labels in candidates:
        if recon.is_file() and labels.is_file():
            return recon.resolve(), labels.resolve()
    checked = "\n".join(f"  {a}\n  {b}" for a, b in candidates)
    raise FileNotFoundError(f"Could not locate {source} Parquet pair. Checked:\n{checked}")


def _read_label_columns(
    labels_path: Path, columns: Iterable[str], expected_rows: int
) -> dict[str, np.ndarray]:
    parquet = pq.ParquetFile(labels_path)
    columns = tuple(columns)
    missing = [column for column in columns if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(
            f"Required label columns {missing!r} are absent from {labels_path}. Available: "
            f"{parquet.schema_arrow.names}"
        )
    table = pq.read_table(labels_path, columns=list(columns))
    if table.num_rows != expected_rows:
        raise ValueError(
            f"Label/reconstruction row mismatch for {labels_path}: "
            f"{table.num_rows} != {expected_rows}"
        )
    return {
        column: np.asarray(
            table.column(index).combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        for index, column in enumerate(columns)
    }


def _symmetric_log(values: np.ndarray) -> np.ndarray:
    """Monotone dynamic-range compression for non-normal charge and coordinates."""
    return np.sign(values) * np.log1p(np.abs(values))


def _build_model_inputs(
    pixels: np.ndarray, y0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the canonical Model-2 and Model-3 vectors for one batch.

    This matches ``dataloader.get_transform`` on the original SmartPixels
    branch: symmetric-log each pixel, average over x, use the final frame for
    Model 2, and use time slices 0--7 transposed to (y,time) for Model 3.
    """
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (N_TIME, N_Y, N_X):
        raise ValueError(f"Expected pixels shaped (batch,{N_TIME},{N_Y},{N_X})")
    y0 = np.asarray(y0, dtype=np.float32)
    if y0.shape != (len(pixels),):
        raise ValueError(f"Expected y0 shaped (batch,), got {y0.shape}")

    transformed = _symmetric_log(pixels)
    y_profile = transformed.mean(axis=-1, dtype=np.float32)  # (B,20,13)
    dense = np.concatenate(
        (y0[:, None], y_profile[:, LAST_FRAME, :]),
        axis=1,
    ).astype(np.float32)
    timing_profile = y_profile[:, :N_SLICES, :].transpose(0, 2, 1)
    conv = np.concatenate(
        (y0[:, None], timing_profile.reshape(len(pixels), -1)),
        axis=1,
    ).astype(np.float32)
    return dense, conv


def _prepare_source(
    source: str,
    recon_path: Path,
    labels_path: Path,
    output_dir: Path,
    timing_column: str,
    batch_size: int,
) -> dict:
    recon = pq.ParquetFile(recon_path)
    labels = pq.ParquetFile(labels_path)
    n_rows = recon.metadata.num_rows
    if labels.metadata.num_rows != n_rows:
        raise ValueError(
            f"{source}: reconstruction has {n_rows} rows but labels have "
            f"{labels.metadata.num_rows}"
        )
    if len(recon.schema_arrow.names) != N_RECON_COLUMNS:
        raise ValueError(
            f"{source}: expected {N_RECON_COLUMNS} reconstruction columns for "
            f"(20,13,21), found {len(recon.schema_arrow.names)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / "dense_inputs.npy"
    conv_path = output_dir / "conv_inputs.npy"
    timing_path = output_dir / "timing.npy"
    dense = np.lib.format.open_memmap(
        dense_path, mode="w+", dtype=np.float32, shape=(n_rows, N_DENSE_INPUT_FEATURES)
    )
    conv = np.lib.format.open_memmap(
        conv_path, mode="w+", dtype=np.float32, shape=(n_rows, N_CONV_INPUT_FEATURES)
    )
    timing = np.lib.format.open_memmap(
        timing_path, mode="w+", dtype=np.float32, shape=(n_rows,)
    )
    label_values = _read_label_columns(
        labels_path,
        (timing_column, Y0_COLUMN),
        n_rows,
    )
    timing[:] = label_values[timing_column]
    y0 = label_values[Y0_COLUMN]

    offset = 0
    for batch_number, batch in enumerate(recon.iter_batches(batch_size=batch_size), 1):
        # Converting one small record batch at a time avoids decompressing an
        # entire 600 MB Parquet file into tens of GB of memory.
        array = batch.to_pandas().to_numpy(dtype=np.float32, copy=False)
        if array.ndim != 2 or array.shape[1] != N_RECON_COLUMNS:
            raise ValueError(f"{source}: unexpected batch shape {array.shape}")
        count = len(array)
        pixels = array.reshape(count, N_TIME, N_Y, N_X)

        batch_dense, batch_conv = _build_model_inputs(
            pixels, y0[offset : offset + count]
        )
        dense[offset : offset + count] = batch_dense
        conv[offset : offset + count] = batch_conv
        offset += count
        print(
            f"[{source}] batch {batch_number}: {offset:,}/{n_rows:,} rows",
            flush=True,
        )

    if offset != n_rows:
        raise RuntimeError(f"{source}: wrote {offset} rows, expected {n_rows}")
    dense.flush()
    conv.flush()
    timing.flush()
    return {
        "source": source,
        "class_index": SOURCE_LABEL[source],
        "rows": n_rows,
        "reconstruction": str(recon_path),
        "labels": str(labels_path),
        "dense_inputs": str(dense_path),
        "conv_inputs": str(conv_path),
        "timing": str(timing_path),
    }


def _split_selected_indices(
    selected: np.ndarray, seed: int, train_fraction: float, validation_fraction: float
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(selected)
    n_train = int(len(order) * train_fraction)
    n_validation = int(len(order) * validation_fraction)
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation :]),
    }


def _write_splits(
    cache_dir: Path,
    row_counts: dict[str, int],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict:
    total_bib = row_counts["bib_mp"] + row_counts["bib_mm"]
    if row_counts["signal"] < total_bib:
        raise ValueError(
            "The requested signal-to-BIB matching requires at least as many "
            f"signal rows as combined BIB rows ({row_counts['signal']} < {total_bib})."
        )
    signal_rng = np.random.default_rng(seed)
    signal_selected = signal_rng.choice(
        row_counts["signal"], size=total_bib, replace=False
    ).astype(np.int64)
    selected_by_source = {
        "signal": signal_selected,
        "bib_mp": np.arange(row_counts["bib_mp"], dtype=np.int64),
        "bib_mm": np.arange(row_counts["bib_mm"], dtype=np.int64),
    }
    summary: dict[str, dict[str, int]] = {}
    for source_index, source in enumerate(SOURCES):
        splits = _split_selected_indices(
            selected_by_source[source],
            seed=seed + 1009 * (source_index + 1),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        summary[source] = {}
        source_dir = cache_dir / source
        for split, indices in splits.items():
            path = source_dir / f"{split}_indices.npy"
            np.save(path, indices, allow_pickle=False)
            summary[source][split] = len(indices)
    return {
        "selection": "all BIB; deterministic signal undersampling to match combined BIB",
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "counts": summary,
    }


def _threshold_sample(
    cache_dir: Path, family: str, maximum_rows: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_train = sum(
        len(np.load(cache_dir / source / "train_indices.npy", mmap_mode="r"))
        for source in SOURCES
    )
    pieces = []
    remaining = maximum_rows
    for source_index, source in enumerate(SOURCES):
        indices = np.load(cache_dir / source / "train_indices.npy", mmap_mode="r")
        if source_index == len(SOURCES) - 1:
            take = remaining
        else:
            take = round(maximum_rows * len(indices) / total_train)
            take = min(take, remaining)
        take = min(take, len(indices))
        selected_positions = rng.choice(len(indices), size=take, replace=False)
        selected_rows = np.sort(
            np.asarray(indices[selected_positions], dtype=np.int64)
        )
        filename = "dense_inputs.npy" if family == "dense" else "conv_inputs.npy"
        values = np.load(cache_dir / source / filename, mmap_mode="r")
        pieces.append(np.asarray(values[selected_rows], dtype=np.float32))
        remaining -= take
    return np.concatenate(pieces, axis=0)


def _derive_thresholds(
    cache_dir: Path, n_thresholds: int, maximum_rows: int, seed: int
) -> dict:
    probabilities = np.arange(1, n_thresholds + 1, dtype=np.float64) / (
        n_thresholds + 1
    )
    dense = _threshold_sample(cache_dir, "dense", maximum_rows, seed)
    dense_thresholds = np.quantile(dense, probabilities, axis=0).T.astype(np.float32)
    conv = _threshold_sample(cache_dir, "conv", maximum_rows, seed + 1)
    conv_thresholds = np.quantile(conv, probabilities, axis=0).T.astype(np.float32)
    np.save(cache_dir / "dense_thresholds.npy", dense_thresholds, allow_pickle=False)
    np.save(cache_dir / "conv_thresholds.npy", conv_thresholds, allow_pickle=False)
    return {
        "method": "training-only empirical quantiles",
        "sample_rows": min(maximum_rows, len(dense)),
        "thresholds_per_feature": n_thresholds,
        "dense_shape": list(dense_thresholds.shape),
        "conv_shape": list(conv_thresholds.shape),
        "threshold_layout": "one ordered quantile set per deployed input feature",
    }


def prepare_cache(
    data_root: Path,
    cache_dir: Path,
    *,
    timing_column: str = DEFAULT_TIMING_COLUMN,
    batch_size: int = 4096,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    n_thresholds: int = 100,
    threshold_sample_rows: int = 50_000,
    overwrite: bool = False,
) -> dict:
    complete_path = cache_dir / "preparation_complete.json"
    if complete_path.is_file() and not overwrite:
        with complete_path.open() as handle:
            metadata = json.load(handle)
        if (
            metadata.get("cache_version") == CACHE_VERSION
            and metadata.get("input_schema") == INPUT_SCHEMA
            and metadata.get("timing_column") == timing_column
        ):
            print(f"[SKIP] Existing complete cache: {cache_dir}")
            return metadata
        raise RuntimeError(
            f"Cache metadata mismatch at {cache_dir}: expected version/schema "
            f"{CACHE_VERSION}/{INPUT_SCHEMA} and timing column {timing_column!r}; "
            f"found {metadata.get('cache_version')}/{metadata.get('input_schema')} "
            f"and {metadata.get('timing_column')!r}. Use a "
            "different MAIA_CACHE_ROOT or --overwrite-cache intentionally."
        )
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be below one")

    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source in SOURCES:
        recon, labels = locate_source_files(data_root, source)
        sources[source] = _prepare_source(
            source,
            recon,
            labels,
            cache_dir / source,
            timing_column,
            batch_size,
        )
    row_counts = {source: int(values["rows"]) for source, values in sources.items()}
    splits = _write_splits(
        cache_dir, row_counts, seed, train_fraction, validation_fraction
    )
    thresholds = _derive_thresholds(
        cache_dir, n_thresholds, threshold_sample_rows, seed
    )
    metadata = {
        "cache_version": CACHE_VERSION,
        "input_schema": INPUT_SCHEMA,
        "data_root": str(data_root.resolve()),
        "task": {
            "negative_class": {"index": 0, "name": "BIB", "sources": ["bib_mp", "bib_mm"]},
            "positive_class": {"index": 1, "name": "signal/PV", "sources": ["signal"]},
        },
        "inputs": {
            "dense": (
                "y-local followed by the 13-bin x-averaged y profile from "
                "time slice 19 (14 features)"
            ),
            "conv": (
                "y-local followed by the x-averaged 13x8 y/time profile from "
                "time slices 0--7, flattened in (y,time) order (105 features)"
            ),
            "position_features": ["y-local (y0)"],
            "class_truth_features_used": [],
            "truth_timing_column_used_as_input": False,
            "charge_time_slices_used": {"dense": [LAST_FRAME], "conv": list(range(N_SLICES))},
            "charge_transform": "sign(x)*log1p(abs(x)) per pixel, then mean over x",
        },
        "timing_column": timing_column,
        "sources": sources,
        "splits": splits,
        "thresholds": thresholds,
    }
    _atomic_json(complete_path, metadata)
    return metadata


class MaiaDataset(Dataset):
    """Map-style view over source-specific memory maps and split indices."""

    def __init__(self, cache_dir: str | Path, split: str, family: str):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.family = family
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        if family not in {"dense", "conv"}:
            raise ValueError(f"Unknown family {family!r}")
        complete_path = self.cache_dir / "preparation_complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"Incomplete or missing cache: {self.cache_dir}")
        with complete_path.open() as handle:
            metadata = json.load(handle)
        if (
            metadata.get("cache_version") != CACHE_VERSION
            or metadata.get("input_schema") != INPUT_SCHEMA
        ):
            raise RuntimeError(
                f"Refusing stale MAIA cache {self.cache_dir}: expected "
                f"version/schema {CACHE_VERSION}/{INPUT_SCHEMA}, found "
                f"{metadata.get('cache_version')}/{metadata.get('input_schema')}"
            )

        filename = "dense_inputs.npy" if family == "dense" else "conv_inputs.npy"
        self._segments = []
        self._ends = []
        end = 0
        for source in SOURCES:
            indices = np.load(
                self.cache_dir / source / f"{split}_indices.npy", mmap_mode="r"
            )
            values = np.load(self.cache_dir / source / filename, mmap_mode="r")
            timing = np.load(self.cache_dir / source / "timing.npy", mmap_mode="r")
            self._segments.append((source, indices, values, timing))
            end += len(indices)
            self._ends.append(end)

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, item: int):
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        segment_index = bisect.bisect_right(self._ends, item)
        start = 0 if segment_index == 0 else self._ends[segment_index - 1]
        source, indices, values, timing = self._segments[segment_index]
        row = int(indices[item - start])
        x = torch.from_numpy(np.array(values[row], dtype=np.float32, copy=True))
        y = torch.tensor(SOURCE_LABEL[source], dtype=torch.long)
        t = torch.tensor(float(timing[row]), dtype=torch.float32)
        return x, y, t

    def class_counts(self) -> dict[int, int]:
        result = {0: 0, 1: 0}
        previous = 0
        for (source, _, _, _), end in zip(self._segments, self._ends):
            result[SOURCE_LABEL[source]] += end - previous
            previous = end
        return result


def deterministic_subset(dataset: Dataset, maximum_examples: int, seed: int) -> Dataset:
    if maximum_examples <= 0 or len(dataset) <= maximum_examples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), maximum_examples, replace=False))
    return Subset(dataset, indices.tolist())


def load_thresholds(cache_dir: str | Path, family: str) -> dict[str, torch.Tensor]:
    cache_dir = Path(cache_dir)
    if family == "dense":
        names = {"dense": "dense_thresholds.npy"}
    elif family == "conv":
        names = {"conv": "conv_thresholds.npy"}
    else:
        raise ValueError(f"Unknown family {family!r}")
    return {
        key: torch.from_numpy(
            np.array(np.load(cache_dir / filename), copy=True)
        ).float()
        for key, filename in names.items()
    }
