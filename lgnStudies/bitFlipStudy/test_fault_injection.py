import unittest

import torch

from lgnStudies.bitFlipStudy.fault_injection import (
    ActivationBitFlipModel,
    random_bit_flip,
)


class FakeDenseLGN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bin = torch.nn.Identity()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Identity(), torch.nn.Identity()]
        )
        self.group_sum = torch.nn.Identity()


class FaultInjectionTest(unittest.TestCase):
    def test_probability_zero_is_identity(self):
        values = torch.tensor([[0.0, 0.25, 1.0]])
        self.assertTrue(torch.equal(random_bit_flip(values, 0.0), values))

    def test_probability_one_is_complement(self):
        values = torch.tensor([[0.0, 0.25, 1.0]])
        expected = torch.tensor([[1.0, 0.75, 0.0]])
        self.assertTrue(torch.equal(random_bit_flip(values, 1.0), expected))

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            random_bit_flip(torch.zeros(1), -0.01)
        with self.assertRaises(ValueError):
            ActivationBitFlipModel(FakeDenseLGN(), 1.01)

    def test_wrapper_flips_after_each_binary_stage(self):
        model = ActivationBitFlipModel(FakeDenseLGN(), probability=1.0)
        values = torch.tensor([[0.0, 1.0]])
        # Three stages (binarization + two layers) means three complements.
        self.assertTrue(torch.equal(model(values), 1 - values))
        self.assertEqual(model.fault_locations, 3)

    def test_relaxed_flip_remains_differentiable(self):
        values = torch.tensor([0.25, 0.75], requires_grad=True)
        random_bit_flip(values, 1.0).sum().backward()
        self.assertTrue(torch.equal(values.grad, -torch.ones_like(values)))

    def test_seed_reproduces_fault_mask(self):
        values = torch.zeros(100)
        torch.manual_seed(123)
        first = random_bit_flip(values, 0.2)
        torch.manual_seed(123)
        second = random_bit_flip(values, 0.2)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
