import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lgnStudies.bitFlipStudy.campaign import (
    EXPECTED_FLIP_COUNTS,
    FLIP_COUNT_CHUNKS,
    MODELS,
    NUMBER_TASKS,
    REPEATS,
    task_configuration,
)
from lgnStudies.bitFlipStudy.combine_results import load_and_combine_model_chunks
from lgnStudies.bitFlipStudy.plots import _aggregate
from lgnStudies.bitFlipStudy.study import (
    DEFAULT_FLIP_COUNTS,
    evaluate_complete_test_split,
    make_result_row,
    validate_clean_checkpoint_metrics,
    validate_transition_accounting,
    write_rows,
)


class PredictionModel(torch.nn.Module):
    def __init__(self, predictions: list[int]):
        super().__init__()
        self.register_buffer(
            "predictions", torch.as_tensor(predictions, dtype=torch.int64)
        )

    def forward(self, inputs):
        indices = inputs[:, 0].to(torch.int64)
        return torch.nn.functional.one_hot(
            self.predictions[indices], num_classes=3
        ).to(torch.float32)


def fake_metrics() -> dict:
    return {
        "balanced_accuracy": 0.5,
        "prediction_change_fraction": 0.1,
        "balanced_repair_gain": 0.08,
        "balanced_damage_loss": 0.03,
        "repair_fraction": 0.07,
        "damage_fraction": 0.02,
        "repairs": [1, 2, 3],
        "damages": [0, 1, 2],
        "recalls": [0.4, 0.5, 0.6],
        "class_totals": [10, 20, 30],
        "confusion_matrix": [[4, 3, 3], [4, 10, 6], [5, 7, 18]],
        "number_test_events": 60,
    }


class BitFlipAnalysisTest(unittest.TestCase):
    def test_clean_checkpoint_must_match_training_final_accuracy(self):
        status = {
            "complete": True,
            "model_alias": MODELS[0][0],
            "registry_name": MODELS[0][1],
            "final_balanced_accuracy": 0.8,
        }
        validate_clean_checkpoint_metrics(
            status,
            MODELS[0][0],
            MODELS[0][1],
            {"balanced_accuracy": 0.8},
        )
        with self.assertRaises(RuntimeError):
            validate_clean_checkpoint_metrics(
                status,
                MODELS[0][0],
                MODELS[0][1],
                {"balanced_accuracy": 0.617},
            )

    def test_repair_damage_identity_explains_balanced_accuracy_delta(self):
        truth = torch.tensor([0, 0, 1, 1, 2, 2])
        labels = torch.nn.functional.one_hot(truth, num_classes=3).float()
        inputs = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        clean_predictions = np.asarray([0, 1, 0, 1, 2, 0], dtype=np.int8)
        model = PredictionModel([0, 0, 1, 2, 1, 2])

        metrics, _ = evaluate_complete_test_split(
            model,
            loader,
            torch.device("cpu"),
            reference_predictions=clean_predictions,
        )

        self.assertEqual(metrics["repairs"], [1, 1, 1])
        self.assertEqual(metrics["damages"], [0, 1, 1])
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["balanced_repair_gain"], 1 / 2)
        self.assertAlmostEqual(metrics["balanced_damage_loss"], 1 / 3)
        self.assertAlmostEqual(
            metrics["balanced_accuracy"] - 1 / 2,
            metrics["balanced_repair_gain"]
            - metrics["balanced_damage_loss"],
        )
        validate_transition_accounting(
            {
                "balanced_accuracy": 1 / 2,
                "class_totals": metrics["class_totals"],
            },
            metrics,
        )

    def test_campaign_task_mapping_covers_every_model_and_chunk(self):
        self.assertEqual(DEFAULT_FLIP_COUNTS, EXPECTED_FLIP_COUNTS)
        observed = [task_configuration(task_id) for task_id in range(NUMBER_TASKS)]
        self.assertEqual(len(observed), len(MODELS) * len(FLIP_COUNT_CHUNKS))
        self.assertEqual(
            {(model, chunk) for model, _, chunk, _ in observed},
            {
                (model, chunk)
                for model, _ in MODELS
                for chunk in range(len(FLIP_COUNT_CHUNKS))
            },
        )
        with self.assertRaises(ValueError):
            task_configuration(NUMBER_TASKS)
        slurm_text = Path(__file__).with_name("run.slurm").read_text()
        self.assertIn(f"#SBATCH --array=0-{NUMBER_TASKS - 1}", slurm_text)

    def test_chunk_combiner_produces_one_clean_row_and_complete_sweep(self):
        model_alias, registry_name = MODELS[0]
        with TemporaryDirectory() as temporary:
            model_root = Path(temporary) / model_alias
            for chunk_index, counts in enumerate(FLIP_COUNT_CHUNKS):
                chunk_root = model_root / "chunks" / f"chunk_{chunk_index:02d}"
                chunk_root.mkdir(parents=True)
                rows = [
                    make_result_row(
                        model_alias,
                        registry_name,
                        0,
                        77_300,
                        0,
                        42,
                        fake_metrics(),
                    )
                ]
                for count in counts:
                    for repeat in range(REPEATS):
                        rows.append(
                            make_result_row(
                                model_alias,
                                registry_name,
                                count,
                                77_300,
                                repeat,
                                42 + count + repeat,
                                fake_metrics(),
                            )
                        )
                write_rows(chunk_root / "bit_flip_results.csv", rows)
                np.save(
                    chunk_root / "clean_predictions.npy",
                    np.asarray([0, 1, 2], dtype=np.int8),
                )
                (chunk_root / "bit_flip_metadata.json").write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "flip_counts_per_event": [0, *counts],
                            "confusion_matrices": {
                                f"K={count}:repeat=0": [[1, 0, 0]] * 3
                                for count in (0, *counts)
                            },
                        }
                    )
                )

            rows, metadata, prediction_source = load_and_combine_model_chunks(
                model_root, model_alias
            )

        counts = [int(row["flip_count_per_event"]) for row in rows]
        self.assertEqual(counts.count(0), 1)
        self.assertEqual(tuple(sorted(set(counts))), EXPECTED_FLIP_COUNTS)
        self.assertEqual(len(rows), 1 + REPEATS * (len(EXPECTED_FLIP_COUNTS) - 1))
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata["chunk_count"], len(FLIP_COUNT_CHUNKS))
        self.assertEqual(prediction_source.name, "clean_predictions.npy")

    def test_plot_aggregation_includes_diagnostic_metrics(self):
        model_alias, registry_name = MODELS[0]
        rows = [
            make_result_row(
                model_alias,
                registry_name,
                count,
                77_300,
                repeat,
                42 + repeat,
                fake_metrics(),
            )
            for count in (0, 1)
            for repeat in range(2)
        ]
        values = _aggregate(rows)
        self.assertIn("balanced_repair_gain_mean", values)
        self.assertIn("balanced_damage_loss_mean", values)
        self.assertIn("recall_class_2_mean", values)


if __name__ == "__main__":
    unittest.main()
