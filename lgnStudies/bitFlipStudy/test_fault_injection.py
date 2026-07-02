import unittest

import torch

from lgnStudies.bitFlipStudy.fault_injection import (
    ExactCountBitFlipModel,
    apply_stage_faults,
    sample_unique_fault_indices,
    validate_flip_count,
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

    def test_stage_applies_only_its_global_indices(self):
        values = torch.zeros((2, 4))
        plan = torch.tensor([[0, 5, 9], [4, 6, 11]])
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        actual = apply_stage_faults(values, plan, stage_offset=4)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_fault_wrapper_matches_original_model(self):
        base = FakeDenseLGN()
        wrapped = ExactCountBitFlipModel(base, flip_count=0)
        values = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(wrapped(values), base(values)))
        self.assertEqual(wrapped.total_fault_sites, 12)
        self.assertEqual(wrapped.fault_stages, 3)

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
