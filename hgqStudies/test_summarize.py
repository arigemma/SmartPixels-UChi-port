import json
import tempfile
import unittest
from pathlib import Path

from hgqStudies.campaign import MODEL_CONFIGS
from hgqStudies.common import sha256_file
from hgqStudies.summarize import collect, write_summary


class SummarizeTest(unittest.TestCase):
    def populate(self, root: Path) -> None:
        for config in MODEL_CONFIGS:
            model_dir = root / config.alias
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "best_model.keras"
            checkpoint.write_bytes(f"checkpoint-{config.index}".encode())
            metrics = {
                "alias": config.alias,
                "balanced_accuracy": 0.7 + config.index / 1000,
                "categorical_accuracy": 0.8,
                "recalls": [0.6, 0.7, 0.8],
                "hgq_ebops": 1000 + config.index,
                "keras_parameter_count": 123,
            }
            (model_dir / "test_metrics.json").write_text(json.dumps(metrics))
            completion = {
                "complete": True,
                "alias": config.alias,
                "best_model_sha256": sha256_file(checkpoint),
            }
            (model_dir / "training_complete.json").write_text(json.dumps(completion))

    def test_strict_complete_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            rows = write_summary(root, root / "summary")
            self.assertEqual(len(rows), 21)
            self.assertTrue((root / "summary" / "balanced_accuracy_summary.tsv").is_file())
            self.assertTrue((root / "summary" / "balanced_accuracy_summary.json").is_file())

    def test_missing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            (root / MODEL_CONFIGS[4].alias / "training_complete.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "1 problem"):
                collect(root)


if __name__ == "__main__":
    unittest.main()
