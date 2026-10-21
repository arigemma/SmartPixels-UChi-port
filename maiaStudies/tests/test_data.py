import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maiaStudies.campaign import N_CONV_INPUT_FEATURES, N_DENSE_INPUT_FEATURES
from maiaStudies.data import (
    CACHE_VERSION,
    INPUT_SCHEMA,
    MaiaDataset,
    SOURCES,
    _build_model_inputs,
    _write_splits,
)


class DataTest(unittest.TestCase):
    def test_balanced_source_split_and_feature_only_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row_counts = {"signal": 22, "bib_mp": 10, "bib_mm": 10}
            for source in SOURCES:
                directory = root / source
                directory.mkdir()
                count = row_counts[source]
                np.save(
                    directory / "dense_inputs.npy",
                    np.zeros((count, N_DENSE_INPUT_FEATURES), np.float32),
                )
                np.save(
                    directory / "conv_inputs.npy",
                    np.zeros((count, N_CONV_INPUT_FEATURES), np.float32),
                )
                np.save(directory / "timing.npy", np.arange(count, dtype=np.float32))
            summary = _write_splits(root, row_counts, 42, 0.8, 0.1)
            (root / "preparation_complete.json").write_text(
                json.dumps(
                    {"cache_version": CACHE_VERSION, "input_schema": INPUT_SCHEMA}
                )
            )
            dataset = MaiaDataset(root, "train", "dense")
            self.assertEqual(dataset.class_counts(), {0: 16, 1: 16})
            x, label, timing = dataset[0]
            self.assertEqual(tuple(x.shape), (N_DENSE_INPUT_FEATURES,))
            self.assertIn(int(label), (0, 1))
            self.assertEqual(timing.ndim, 0)
            self.assertIn("deterministic signal undersampling", summary["selection"])

    def test_model2_and_model3_match_canonical_y_profile_schemas(self):
        pixels = np.zeros((1, 20, 13, 21), dtype=np.float32)
        pixels[0, 0, 2, 7] = 3.0
        pixels[0, 7, 12, 20] = 5.0
        pixels[0, 8, 4, 1] = 7.0  # excluded from both deployed schemas
        pixels[0, 19, 5, 11] = 11.0
        y0 = np.array([2.5], dtype=np.float32)
        dense, conv = _build_model_inputs(pixels, y0)
        self.assertEqual(dense.shape, (1, N_DENSE_INPUT_FEATURES))
        self.assertEqual(conv.shape, (1, N_CONV_INPUT_FEATURES))
        self.assertEqual(float(dense[0, 0]), 2.5)
        self.assertEqual(float(conv[0, 0]), 2.5)

        expected_final = np.log1p(11.0) / 21.0
        self.assertAlmostEqual(float(dense[0, 1 + 5]), expected_final, places=6)
        self.assertEqual(np.count_nonzero(dense[0, 1:]), 1)

        # After y0, Model 3 is flattened in (y,time) order over slices 0--7.
        self.assertAlmostEqual(
            float(conv[0, 1 + 2 * 8 + 0]), np.log1p(3.0) / 21.0, places=6
        )
        self.assertAlmostEqual(
            float(conv[0, 1 + 12 * 8 + 7]), np.log1p(5.0) / 21.0, places=6
        )
        self.assertEqual(np.count_nonzero(conv[0, 1:]), 2)


if __name__ == "__main__":
    unittest.main()
