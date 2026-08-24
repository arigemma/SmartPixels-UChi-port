import json
import tempfile
import unittest
from pathlib import Path

from hgqStudies.campaign import MODEL_CONFIGS
from hgqStudies.common import sha256_file
from hgqStudies.summarize_calibrated import collect


class CalibratedSummaryTest(unittest.TestCase):
    def populate(self, root: Path, failing_index=None) -> None:
        for config in MODEL_CONFIGS:
            output = root / config.alias / "calibration"
            output.mkdir(parents=True)
            checkpoint = output / "best_model.calibrated.keras"
            checkpoint.write_bytes(f"calibrated-{config.index}".encode())
            digest = sha256_file(checkpoint)
            passes = config.index != failing_index
            result = {
                "alias": config.alias,
                "task_index": config.index,
                "passes_acceptance": passes,
                "checkpoint_sha256": digest,
                "selected_source_checkpoint": "final_model.keras",
                "calibrated_ebops": 1000 + config.index,
                "selected_validation_metrics": {
                    "balanced_accuracy": 0.75,
                    "cross_entropy": 0.5,
                },
                "test_metrics": {
                    "balanced_accuracy": 0.74,
                    "categorical_accuracy": 0.8,
                    "cross_entropy": 0.6,
                },
            }
            (output / "calibrated_test_metrics.json").write_text(json.dumps(result))
            completion = {
                "complete": True,
                "alias": config.alias,
                "passes_acceptance": passes,
                "checkpoint_sha256": digest,
            }
            (output / "calibration_complete.json").write_text(json.dumps(completion))

    def test_collects_only_complete_accepted_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            rows = collect(root)
            self.assertEqual(len(rows), 21)
            self.assertEqual(rows[0]["balanced_accuracy"], 0.74)

    def test_rejects_failed_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root, failing_index=3)
            with self.assertRaisesRegex(RuntimeError, "did not pass acceptance"):
                collect(root)

    def test_rejects_alias_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            path = (
                root
                / MODEL_CONFIGS[0].alias
                / "calibration"
                / "calibrated_test_metrics.json"
            )
            result = json.loads(path.read_text())
            result["alias"] = MODEL_CONFIGS[1].alias
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, "metadata alias mismatch"):
                collect(root)


if __name__ == "__main__":
    unittest.main()
