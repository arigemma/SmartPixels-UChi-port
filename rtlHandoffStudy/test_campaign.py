from __future__ import annotations

import unittest

import torch

from models.registry import REGISTRY
from models.rtl_handoff import (
    OrderedLearnableBinarization,
    RTL_HANDOFF_CONFIGS,
    RTL_HANDOFF_BY_ALIAS,
    _reduction_widths,
)
from rtlHandoffStudy.campaign import validate_campaign


class RtlHandoffCampaignTest(unittest.TestCase):
    def test_manifest_and_registry(self):
        validate_campaign()
        self.assertEqual(len(RTL_HANDOFF_CONFIGS), 20)
        self.assertEqual(
            sum(config["family"] == "dense" for config in RTL_HANDOFF_CONFIGS), 10
        )
        for config in RTL_HANDOFF_BY_ALIAS.values():
            self.assertIn(config["registry_name"], REGISTRY)

    def test_ordered_thresholds_are_consistent_across_modes(self):
        initial = torch.tensor(
            [[-1.0, -0.25, 0.5, 1.5], [0.0, 0.1, 0.4, 2.0]],
            dtype=torch.float32,
        )
        layer = OrderedLearnableBinarization(initial)
        layer.train()
        training_thresholds = layer.get_thresholds().detach().clone()
        layer.eval()
        evaluation_thresholds = layer.get_thresholds().detach().clone()
        self.assertTrue(torch.allclose(training_thresholds, evaluation_thresholds))
        self.assertTrue(torch.all(torch.diff(evaluation_thresholds, dim=-1) > 0))
        self.assertTrue(torch.allclose(initial, evaluation_thresholds, atol=2e-6))

    def test_hard_output_is_feature_major_threshold_minor(self):
        thresholds = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
        layer = OrderedLearnableBinarization(thresholds).eval()
        values = torch.tensor([[1.5, 25.0]])
        expected = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(layer(values), expected))

    def test_eflx_io_screening_values(self):
        dense = REGISTRY[
            RTL_HANDOFF_BY_ALIAS["dense_s01_t63_fixed_g128"]["registry_name"]
        ]["class"].deployment_config()
        conv15 = REGISTRY[
            RTL_HANDOFF_BY_ALIAS["conv_s02_t15_fixed_g64"]["registry_name"]
        ]["class"].deployment_config()
        conv63 = REGISTRY[
            RTL_HANDOFF_BY_ALIAS["conv_s01_t63_fixed_g100"]["registry_name"]
        ]["class"].deployment_config()
        self.assertEqual(dense["compact_input_bits"], 84)
        self.assertEqual(conv15["compact_input_bits"], 420)
        self.assertEqual(conv63["compact_input_bits"], 630)

    def test_group_reduction_paths_cover_every_input(self):
        for in_dim, out_dim in ((1269, 384), (1914, 192), (1200, 192)):
            current = in_dim
            for width in _reduction_widths(in_dim, out_dim):
                self.assertGreaterEqual(2 * width, current)
                current = width
            self.assertEqual(current, out_dim)


if __name__ == "__main__":
    unittest.main()
