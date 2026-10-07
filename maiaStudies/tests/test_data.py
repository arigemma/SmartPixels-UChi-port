import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from maiaStudies.data import MaiaDataset, SOURCES, SOURCE_LABEL, _write_splits


class DataTest(unittest.TestCase):
    def test_balanced_source_split_and_feature_only_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row_counts = {"signal": 22, "bib_mp": 10, "bib_mm": 10}
            for source in SOURCES:
                directory = root / source
                directory.mkdir()
                count = row_counts[source]
                np.save(directory / "dense_profiles.npy", np.zeros((count, 13), np.float32))
                np.save(
                    directory / "temporal_profiles.npy",
                    np.zeros((count, 13, 20), np.float32),
                )
                np.save(directory / "timing.npy", np.arange(count, dtype=np.float32))
            summary = _write_splits(root, row_counts, 42, 0.8, 0.1)
            (root / "preparation_complete.json").write_text(
                json.dumps({"cache_version": 1})
            )
            dataset = MaiaDataset(root, "train", "dense")
            self.assertEqual(dataset.class_counts(), {0: 16, 1: 16})
            x, label, timing = dataset[0]
            self.assertEqual(tuple(x.shape), (13,))
            self.assertIn(int(label), (0, 1))
            self.assertEqual(timing.ndim, 0)
            self.assertIn("deterministic signal undersampling", summary["selection"])


if __name__ == "__main__":
    unittest.main()
