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
    _position_features,
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

    def test_position_feature_split_matches_ten_13mm_modules(self):
        values = _position_features(
            np.array([-65.0, -58.5, -52.0, 0.0, 64.999], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(values[:, 0], [-5, -5, -4, 0, 4])
        np.testing.assert_allclose(
            values[:, 1], [-6.5, 0.0, -6.5, -6.5, 6.499], atol=2e-3
        )
        np.testing.assert_array_equal(values[:, 2], [1, 2, 3, 4, 5])

    def test_model2_and_model3_inputs_preserve_both_spatial_axes_and_position(self):
        pixels = np.zeros((1, 20, 13, 21), dtype=np.float32)
        pixels[0, 0, 2, 7] = 3.0
        pixels[0, 1, 2, 7] = 5.0
        position = np.array([[4.0, -1.5, 2.5]], dtype=np.float32)
        dense, conv = _build_model_inputs(pixels, position)
        self.assertEqual(dense.shape, (1, N_DENSE_INPUT_FEATURES))
        self.assertEqual(conv.shape, (1, N_CONV_INPUT_FEATURES))
        expected = np.log1p(8.0)
        self.assertAlmostEqual(float(dense[0, 7]), expected)
        self.assertAlmostEqual(float(dense[0, 21 + 2]), expected)
        np.testing.assert_array_equal(dense[0, -3:], position[0])
        self.assertAlmostEqual(float(conv[0, 2 * 21 + 7]), expected)
        np.testing.assert_array_equal(conv[0, -3:], position[0])


if __name__ == "__main__":
    unittest.main()
