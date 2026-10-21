import unittest

import numpy as np

from maiaStudies.campaign import (
    ALL_CONFIGS,
    CAMPAIGN_CONFIGS,
    N_CONV_INPUT_FEATURES,
    N_DENSE_INPUT_FEATURES,
    N_SLICES,
    TAU_EXTENSION_CONFIGS,
    get_config,
)
from maiaStudies.metrics import evaluate_binary_scores
from maiaStudies.models import resource_proxies


class CampaignTest(unittest.TestCase):
    def test_grid_and_binary_groups(self):
        self.assertEqual(N_DENSE_INPUT_FEATURES, 14)
        self.assertEqual(N_CONV_INPUT_FEATURES, 105)
        self.assertEqual(N_SLICES, 8)
        self.assertEqual(len(CAMPAIGN_CONFIGS), 30)
        self.assertEqual(sum(row.family == "dense" for row in CAMPAIGN_CONFIGS), 15)
        self.assertEqual(sum(row.family == "conv" for row in CAMPAIGN_CONFIGS), 15)
        self.assertEqual({row.tau for row in CAMPAIGN_CONFIGS}, {10.0, 20.0, 40.0})
        for row in CAMPAIGN_CONFIGS:
            row.validate()
            self.assertEqual(row.lut_rank, 2)
            self.assertEqual(row.final_logic_width % 2, 0)
            self.assertGreater(resource_proxies(row)["logic_node_proxy"], 0)
        self.assertEqual(CAMPAIGN_CONFIGS[0].dense_widths, (1269, 1269, 1270))
        self.assertEqual(CAMPAIGN_CONFIGS[15].conv_kernels, (16, 32, 64, 60))
        self.assertEqual(CAMPAIGN_CONFIGS[15].dense_widths, (1200, 1200, 600, 300))

    def test_tau_extension_reuses_every_architecture_at_tau_one_and_three(self):
        self.assertEqual(len(TAU_EXTENSION_CONFIGS), 20)
        self.assertEqual(len(ALL_CONFIGS), 50)
        self.assertEqual(
            {row.tau for row in TAU_EXTENSION_CONFIGS},
            {1.0, 3.0},
        )
        self.assertEqual(
            get_config(0, "tau_extension").alias,
            "dense_s01_tau1",
        )
        self.assertEqual(
            get_config(19, "tau_extension").alias,
            "conv_s05_tau3",
        )
        self.assertEqual(len({row.alias for row in ALL_CONFIGS}), 50)
        self.assertEqual([row.task_index for row in ALL_CONFIGS], list(range(50)))

    def test_physics_metrics(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([-3.0, -2.0, 1.0, -1.0, 2.0, 3.0])
        result = evaluate_binary_scores(labels, scores, labels, scores)
        self.assertAlmostEqual(result["argmax"]["balanced_accuracy"], 2.0 / 3.0)
        self.assertGreater(result["roc_auc"], 0.5)
        self.assertIn(
            "signal_efficiency_90pct", result["validation_derived_operating_points"]
        )


if __name__ == "__main__":
    unittest.main()
