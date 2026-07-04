import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from lgnStudies.recover_resource_checkpoint import rebuild_registered_dense_lgn
from lgnStudies.resource_precision_study import training_requirements_met
from lgnStudies.training_completion import validate_training_completion


class TrainingCompletionTest(unittest.TestCase):
    def test_rebuilds_registered_forward_from_portable_sequential(self):
        class FakeRegistered(torch.nn.Module):
            def forward(self, values):
                values = self.bin(values)
                for layer in self.layers:
                    values = layer(values)
                return self.group_sum(values)

        portable = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 3),
        ).eval()
        recovered = rebuild_registered_dense_lgn(portable, FakeRegistered)
        values = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        self.assertTrue(torch.equal(portable(values), recovered(values)))

    def test_requires_both_steps_and_time(self):
        self.assertFalse(
            training_requirements_met(19_999, 20_000, 5 * 3600, 5.0)
        )
        self.assertFalse(
            training_requirements_met(20_000, 20_000, 4.99 * 3600, 5.0)
        )
        self.assertTrue(
            training_requirements_met(20_000, 20_000, 5 * 3600, 5.0)
        )
        self.assertTrue(
            training_requirements_met(25_000, 20_000, 5.2 * 3600, 5.0)
        )

    def test_validates_completed_checkpoint_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final_model.pth"
            status_path = root / "training_complete.json"
            checkpoint.write_bytes(b"checkpoint")
            status = {
                "complete": True,
                "minimum_training_hours": 5.0,
                "elapsed_training_hours": 5.1,
                "minimum_steps": 2_000,
                "completed_steps": 7_500,
                "final_recorded_step": 7_500,
                "final_balanced_accuracy": 0.81,
                "stop_requested": False,
            }
            status_path.write_text(json.dumps(status))
            self.assertEqual(
                validate_training_completion(checkpoint, status_path, 5.0, 2_000),
                status,
            )

            status["elapsed_training_hours"] = 2.0
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                validate_training_completion(checkpoint, status_path, 5.0)

            status["elapsed_training_hours"] = 5.1
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                validate_training_completion(
                    checkpoint, status_path, 5.0, 20_000
                )

    def test_rejects_interrupted_or_step_mismatched_runs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final_model.pth"
            status_path = root / "training_complete.json"
            checkpoint.write_bytes(b"checkpoint")
            status = {
                "complete": True,
                "minimum_training_hours": 5.0,
                "elapsed_training_hours": 5.1,
                "minimum_steps": 2_000,
                "completed_steps": 7_500,
                "final_recorded_step": 7_000,
                "final_balanced_accuracy": 0.81,
                "stop_requested": False,
            }
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                validate_training_completion(checkpoint, status_path, 5.0)

    def test_accepts_recovered_recorded_stage_by_step_evidence(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "final_model.pth"
            status_path = root / "training_complete.json"
            checkpoint.write_bytes(b"checkpoint")
            status = {
                "complete": True,
                "completion_basis": "recovered_latest_recorded_stage",
                "minimum_training_hours": None,
                "elapsed_training_hours": None,
                "minimum_steps": 20_000,
                "completed_steps": 100_000,
                "final_recorded_step": 100_000,
                "final_balanced_accuracy": 0.8,
                "stop_requested": False,
            }
            status_path.write_text(json.dumps(status))
            self.assertEqual(
                validate_training_completion(
                    checkpoint, status_path, 5.0, 20_000
                ),
                status,
            )

            status["final_recorded_step"] = 7_500
            status["stop_requested"] = True
            status_path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                validate_training_completion(checkpoint, status_path, 5.0)


if __name__ == "__main__":
    unittest.main()
