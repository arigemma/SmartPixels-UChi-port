import unittest

import torch

from lgnStudies.bitFlipStudy.fault_injection import (
    ExactCountBitFlipModel,
    apply_stage_faults,
    sample_unique_fault_indices,
    validate_flip_count,
    validate_fault_operation,
    validate_fault_scope,
)


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


class FaultInjectionTest(unittest.TestCase):
    def test_sampler_returns_exact_unique_count_for_every_event(self):
        torch.manual_seed(12)
        plan = sample_unique_fault_indices(
            batch_size=128,
            total_sites=1000,
            flip_count=64,
            device="cpu",
        )
        self.assertEqual(plan.shape, (128, 64))
        self.assertTrue(torch.all(plan >= 0))
        self.assertTrue(torch.all(plan < 1000))
        for event_plan in plan:
            self.assertEqual(len(torch.unique(event_plan)), 64)

    def test_zero_count_has_empty_plan(self):
        plan = sample_unique_fault_indices(7, 100, 0, "cpu")
        self.assertEqual(plan.shape, (7, 0))

    def test_invalid_count_is_rejected(self):
        for count in (-1, 11, 1.5, True):
            with self.assertRaises(ValueError):
                validate_flip_count(count, 10)

    def test_invalid_fault_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_fault_scope("weights")

    def test_invalid_fault_operation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_fault_operation("stuck_at_zero")

    def test_stage_applies_only_its_global_indices(self):
        values = torch.zeros((2, 4))
        plan = torch.tensor([[0, 5, 9], [4, 6, 11]])
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        actual = apply_stage_faults(values, plan, stage_offset=4)
        self.assertTrue(torch.equal(actual, expected))

    def test_random_set_uses_targets_and_can_leave_a_site_unchanged(self):
        values = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        plan = torch.tensor([[0, 1, 3]])
        targets = torch.tensor([[0, 0, 1]])
        actual = apply_stage_faults(
            values.clone(),
            plan,
            stage_offset=0,
            fault_operation="random_set",
            fault_values=targets,
        )
        self.assertTrue(torch.equal(actual, torch.tensor([[0.0, 0.0, 0.0, 1.0]])))

    def test_zero_fault_wrapper_matches_original_model(self):
        base = FakeDenseLGN()
        wrapped = ExactCountBitFlipModel(base, flip_count=0)
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(wrapped(values), base(values)))
        self.assertEqual(wrapped.total_fault_sites, 12)
        self.assertEqual(wrapped.fault_stages, 3)
        self.assertEqual(wrapped.fault_scope, "all")

    def test_input_scope_counts_only_binarized_inputs(self):
        model = ExactCountBitFlipModel(
            FakeDenseLGN(),
            flip_count=4,
            fault_scope="input",
        )
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        expected = 1 - values
        self.assertEqual(model.total_fault_sites, 4)
        self.assertEqual(model.fault_stages, 1)
        self.assertEqual(model.fault_scope, "input")
        self.assertTrue(torch.equal(model(values), expected))

        with self.assertRaises(ValueError):
            model.set_flip_count(5)

    def test_internal_scope_excludes_binarized_inputs(self):
        model = ExactCountBitFlipModel(
            FakeDenseLGN(),
            flip_count=8,
            fault_scope="internal",
        )
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        # Two internal complements cancel; the input stage is untouched.
        self.assertEqual(model.total_fault_sites, 8)
        self.assertEqual(model.fault_stages, 2)
        self.assertTrue(torch.equal(model(values), values))

    def test_random_set_seed_reproduces_indices_and_targets(self):
        model = ExactCountBitFlipModel(
            FakeDenseLGN(width=16),
            flip_count=5,
            fault_scope="internal",
            fault_operation="random_set",
        )
        values = torch.zeros((20, 16))
        torch.manual_seed(987)
        first = model(values)
        torch.manual_seed(987)
        second = model(values)
        self.assertTrue(torch.equal(first, second))

    def test_flipping_every_site_complements_each_stage(self):
        model = ExactCountBitFlipModel(FakeDenseLGN(), flip_count=12)
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        expected = 1 - values
        # Binarizer plus two logic-layer outputs gives three complements.
        self.assertTrue(torch.equal(model(values), expected))

    def test_seed_reproduces_faults(self):
        model = ExactCountBitFlipModel(FakeDenseLGN(width=16), flip_count=5)
        values = torch.zeros((20, 16))
        torch.manual_seed(123)
        first = model(values)
        torch.manual_seed(123)
        second = model(values)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
