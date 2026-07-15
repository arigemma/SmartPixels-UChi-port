from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from lgnStudies.bitFlipStudy.campaign import FLIP_COUNT_CHUNKS, MODELS
from lgnStudies.bitFlipTrainingStudy.campaign import (
    NUMBER_EVALUATION_TASKS,
    NUMBER_TRAINING_TASKS,
    evaluation_configuration,
    training_configuration,
)
from lgnStudies.bitFlipTrainingStudy.fault_injection import (
    FaultAwareTrainingModel,
    apply_bernoulli_faults,
    validate_fault_probability,
)
from lgnStudies.bitFlipTrainingStudy.s08_percentage_sweep import (
    MAX_EVALUATION_PERCENTAGE,
    S08_TOTAL_FAULT_SITES,
    evaluation_configuration as s08_evaluation_configuration,
    evaluation_flip_count_chunks,
    evaluation_flip_counts,
    parse_training_percentages,
    training_configuration as s08_training_configuration,
)
from lgnStudies.bitFlipTrainingStudy.s08_fault_scope_sweep import (
    S08_INPUT_FAULT_SITES,
    evaluation_configuration as s08_scope_evaluation_configuration,
    input_evaluation_flip_count_chunks,
    input_evaluation_flip_counts,
    training_configuration as s08_scope_training_configuration,
)
from lgnStudies.bitFlipTrainingStudy.train import plot_training_history


class FakeLogicDense(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.in_dim = width
        self.out_dim = width

    def forward(self, values):
        return values


class FakeDenseLGN(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.bin = torch.nn.Identity()
        self.layers = torch.nn.ModuleList(
            [FakeLogicDense(width), FakeLogicDense(width)]
        )
        self.group_sum = torch.nn.Identity()

    def forward(self, values):
        values = self.bin(values)
        for layer in self.layers:
            values = layer(values)
        return self.group_sum(values)


class FaultAwareTrainingTest(unittest.TestCase):
    def test_probability_validation(self):
        self.assertEqual(validate_fault_probability(0), 0.0)
        self.assertEqual(validate_fault_probability(1), 1.0)
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                validate_fault_probability(value)

    def test_fault_operation_is_out_of_place_and_differentiable(self):
        values = torch.tensor([[0.2, 0.8]], requires_grad=True)
        original = values.detach().clone()
        result = apply_bernoulli_faults(values, 1.0)
        self.assertTrue(torch.equal(values.detach(), original))
        self.assertTrue(torch.allclose(result, 1 - values))
        result.sum().backward()
        self.assertTrue(torch.equal(values.grad, -torch.ones_like(values)))

    def test_faults_are_enabled_only_during_training(self):
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        model = FaultAwareTrainingModel(FakeDenseLGN(), fault_probability=1.0)
        self.assertEqual(model.total_fault_sites, 12)
        self.assertEqual(model.expected_flips_per_event, 12)

        model.train()
        # Three complemented stages (binarizer plus two layers) leave a
        # complemented output.
        self.assertTrue(torch.equal(model(values), 1 - values))

        model.eval()
        self.assertTrue(torch.equal(model(values), values))

    def test_input_scope_faults_only_binarized_inputs(self):
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        model = FaultAwareTrainingModel(
            FakeDenseLGN(),
            fault_probability=1.0,
            fault_scope="input",
        ).train()
        self.assertEqual(model.fault_scope, "input")
        self.assertEqual(model.total_fault_sites, 4)
        self.assertEqual(model.expected_flips_per_event, 4)
        self.assertTrue(torch.equal(model(values), 1 - values))

        with self.assertRaises(ValueError):
            FaultAwareTrainingModel(
                FakeDenseLGN(),
                fault_probability=0.1,
                fault_scope="weights",
            )

    def test_zero_probability_matches_base_model(self):
        base = FakeDenseLGN(width=8)
        model = FaultAwareTrainingModel(base, fault_probability=0.0).train()
        values = torch.rand(5, 8)
        self.assertTrue(torch.equal(model(values), base(values)))

    def test_campaign_maps_all_models_and_evaluation_chunks(self):
        self.assertEqual(NUMBER_TRAINING_TASKS, len(MODELS))
        self.assertEqual(
            [training_configuration(index) for index in range(NUMBER_TRAINING_TASKS)],
            list(MODELS),
        )
        observed = [
            evaluation_configuration(index)
            for index in range(NUMBER_EVALUATION_TASKS)
        ]
        self.assertEqual(
            {(model, chunk) for model, _, chunk, _ in observed},
            {
                (model, chunk)
                for model, _ in MODELS
                for chunk in range(len(FLIP_COUNT_CHUNKS))
            },
        )
        with self.assertRaises(ValueError):
            training_configuration(NUMBER_TRAINING_TASKS)
        with self.assertRaises(ValueError):
            evaluation_configuration(NUMBER_EVALUATION_TASKS)

        root = Path(__file__).parent
        self.assertIn(
            f"#SBATCH --array=0-{NUMBER_TRAINING_TASKS - 1}",
            (root / "run.slurm").read_text(),
        )
        self.assertIn(
            f"#SBATCH --array=0-{NUMBER_EVALUATION_TASKS - 1}",
            (root / "evaluate.slurm").read_text(),
        )

    def test_training_plot_contains_balanced_accuracy_and_class_recalls(self):
        rows = [
            {
                "step": step,
                "training_fault_probability": 0.01,
                "clean_balanced_accuracy": 0.5 + step / 100_000,
                "clean_recall_class_0": 0.4,
                "clean_recall_class_1": 0.5,
                "clean_recall_class_2": 0.6,
            }
            for step in (0, 5_000)
        ]
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "training_history"
            plot_training_history(rows, "fake_model", output)
            self.assertTrue(output.with_suffix(".png").is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())

    def test_s08_percentage_sweep_count_grid_ends_at_5p3_percent(self):
        counts = evaluation_flip_counts()
        chunks = evaluation_flip_count_chunks()
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], round(S08_TOTAL_FAULT_SITES * 0.053))
        self.assertAlmostEqual(
            100 * counts[-1] / S08_TOTAL_FAULT_SITES,
            MAX_EVALUATION_PERCENTAGE,
            places=3,
        )
        self.assertEqual(tuple(count for chunk in chunks for count in chunk), counts[1:])
        self.assertTrue(all(left < right for left, right in zip(counts, counts[1:])))

    def test_s08_percentage_sweep_task_mapping(self):
        percentages = parse_training_percentages("3.0 0.1 1.0")
        self.assertEqual(percentages, (0.1, 1.0, 3.0))
        self.assertEqual(
            s08_training_configuration(1, percentages)[:2],
            (1.0, "train_1pct"),
        )
        condition, tag, percentage_text, model, registry, chunk, counts = (
            s08_evaluation_configuration(0, percentages)
        )
        self.assertEqual(condition, "inference_only")
        self.assertEqual(tag, "inference_only")
        self.assertEqual(percentage_text, "none")
        self.assertEqual(chunk, 0)
        self.assertEqual(tuple(counts), evaluation_flip_count_chunks()[0])

        condition, tag, percentage_text, model, registry, chunk, counts = (
            s08_evaluation_configuration(len(evaluation_flip_count_chunks()), percentages)
        )
        self.assertEqual(condition, "fault_aware_training")
        self.assertEqual(tag, "train_0p1pct")
        self.assertEqual(percentage_text, "0.1")

    def test_s08_input_scope_grid_ends_at_5p3_percent(self):
        counts = input_evaluation_flip_counts()
        chunks = input_evaluation_flip_count_chunks()
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], round(S08_INPUT_FAULT_SITES * 0.053))
        self.assertAlmostEqual(
            100 * counts[-1] / S08_INPUT_FAULT_SITES,
            MAX_EVALUATION_PERCENTAGE,
            places=1,
        )
        self.assertEqual(tuple(count for chunk in chunks for count in chunk), counts[1:])
        self.assertTrue(all(left < right for left, right in zip(counts, counts[1:])))

    def test_s08_fault_scope_task_mapping(self):
        percentages = parse_training_percentages("3.0 0.1 1.0")
        self.assertEqual(
            s08_scope_training_configuration(1, percentages),
            (
                1.0,
                "train_1pct",
                "s08_p1202M_tau40",
                "lgn-dense-2-dense-100_128_-model2lgnFull_s08_p1202M_40",
            ),
        )
        condition, scope, tag, percentage_text, model, registry, chunk, counts = (
            s08_scope_evaluation_configuration(0, percentages)
        )
        self.assertEqual(condition, "inference_only")
        self.assertEqual(scope, "input_only")
        self.assertEqual(tag, "inference_only")
        self.assertEqual(percentage_text, "none")
        self.assertEqual(chunk, 0)
        self.assertEqual(tuple(counts), input_evaluation_flip_count_chunks()[0])

        condition, scope, tag, percentage_text, model, registry, chunk, counts = (
            s08_scope_evaluation_configuration(
                len(input_evaluation_flip_count_chunks()),
                percentages,
            )
        )
        self.assertEqual(condition, "fault_aware_training")
        self.assertEqual(scope, "input_only")
        self.assertEqual(tag, "train_0p1pct")
        self.assertEqual(percentage_text, "0.1")


if __name__ == "__main__":
    unittest.main()
