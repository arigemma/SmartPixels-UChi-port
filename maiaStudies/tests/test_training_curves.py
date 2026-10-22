import tempfile
import unittest
from pathlib import Path

from maiaStudies.training_curves import (
    generate_training_curves,
    _parse_training_log,
    _read_validation_history,
    _submitted_training_job_id,
)


class TrainingCurvesTest(unittest.TestCase):
    def test_parses_legacy_and_paired_validation_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.out"
            path.write_text(
                "[TRAIN] epoch=2 step=100 loss=0.250000 batch_accuracy=0.900000\n"
                "[VALIDATION] step=100 elapsed_hours=0.125 "
                '{"balanced_accuracy": 0.91, "roc_auc": 0.96, "cross_entropy": 0.2}\n'
                "[VALIDATION_RELAXED] step=200 elapsed_hours=0.250 "
                '{"balanced_accuracy": 0.92, "roc_auc": 0.97, "cross_entropy": 0.18}\n'
            )
            training, validation = _parse_training_log(path)
        self.assertEqual(training[0]["step"], 100)
        self.assertAlmostEqual(training[0]["loss"], 0.25)
        self.assertEqual(validation[0]["mode"], "discrete")
        self.assertEqual(validation[1]["mode"], "relaxed")
        self.assertAlmostEqual(validation[1]["roc_auc"], 0.97)

    def test_reads_new_validation_history_tsv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_history.tsv"
            path.write_text(
                "step\telapsed_hours\tdiscrete_balanced_accuracy\t"
                "relaxed_balanced_accuracy\n"
                "5000\t0.5\t0.91\t0.93\n"
            )
            rows = _read_validation_history(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["mode"] for row in rows}, {"discrete", "relaxed"})

    def test_finds_training_job_id_without_sourcing_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "submitted_jobs.txt").write_text(
                "prepare_job_id=10\ntraining_job_id=15689861\n"
            )
            self.assertEqual(_submitted_training_job_id(root), "15689861")

    def test_generates_complete_legacy_plot_package_from_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "results"
            logs = base / "logs"
            root.mkdir()
            logs.mkdir()
            (root / "submitted_jobs.txt").write_text("training_job_id=12345\n")
            for task in range(30):
                (logs / f"maiaLnnTrain_12345_{task}.out").write_text(
                    "[TRAIN] epoch=0 step=100 loss=0.4 batch_accuracy=0.8\n"
                    "[VALIDATION_DISCRETE] step=100 elapsed_hours=0.1 "
                    '{"balanced_accuracy": 0.8, "roc_auc": 0.9, '
                    '"cross_entropy": 0.3}\n'
                    "[VALIDATION_RELAXED] step=100 elapsed_hours=0.1 "
                    '{"balanced_accuracy": 0.81, "roc_auc": 0.91, '
                    '"cross_entropy": 0.29}\n'
                    "[TRAIN] epoch=1 step=200 loss=0.3 batch_accuracy=0.85\n"
                    "[VALIDATION_DISCRETE] step=200 elapsed_hours=0.2 "
                    '{"balanced_accuracy": 0.82, "roc_auc": 0.92, '
                    '"cross_entropy": 0.25}\n'
                    "[VALIDATION_RELAXED] step=200 elapsed_hours=0.2 "
                    '{"balanced_accuracy": 0.83, "roc_auc": 0.93, '
                    '"cross_entropy": 0.24}\n'
                )
            result = generate_training_curves(
                root,
                logs_dir=logs,
                campaign="native_legacy",
                formats=("png",),
                dpi=30,
                require_paired=True,
            )
            self.assertEqual(result["models_found"], 30)
            self.assertEqual(result["paired_models_found"], 30)
            self.assertEqual(len(result["plots"]), 5)
            self.assertTrue(all(Path(path).is_file() for path in result["plots"]))


if __name__ == "__main__":
    unittest.main()
