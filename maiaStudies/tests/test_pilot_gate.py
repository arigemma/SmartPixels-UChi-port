import json
import tempfile
import unittest
from pathlib import Path

from maiaStudies.campaign import get_config
from maiaStudies.data import INPUT_SCHEMA
from maiaStudies.pilot_gate import PILOT_TASKS, check_pilot


class PilotGateTest(unittest.TestCase):
    def _write_result(self, root: Path, task: int, balanced: float, auc: float) -> None:
        directory = root / get_config(task).alias
        directory.mkdir(parents=True)
        (directory / "evaluation_complete.json").write_text(
            json.dumps(
                {
                    "input_schema": INPUT_SCHEMA,
                    "full_test": {
                        "argmax": {"balanced_accuracy": balanced},
                        "roc_auc": auc,
                    },
                }
            )
        )

    def test_gate_accepts_two_learning_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in PILOT_TASKS:
                self._write_result(root, task, 0.70, 0.80)
            result = check_pilot(root)
            self.assertTrue(result["passed"])
            self.assertTrue((root / "pilot_gate_complete.json").is_file())

    def test_gate_rejects_chance_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_result(root, PILOT_TASKS[0], 0.70, 0.80)
            self._write_result(root, PILOT_TASKS[1], 0.50, 0.51)
            with self.assertRaisesRegex(RuntimeError, "pilot gate failed"):
                check_pilot(root)
            self.assertTrue((root / "pilot_gate_failed.json").is_file())


if __name__ == "__main__":
    unittest.main()
