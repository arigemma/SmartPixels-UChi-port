import unittest

import numpy as np

from maiaStudies.campaign import CAMPAIGN_CONFIGS
from maiaStudies.metrics import evaluate_binary_scores
from maiaStudies.models import resource_proxies


class CampaignTest(unittest.TestCase):
    def test_grid_and_binary_groups(self):
        self.assertEqual(len(CAMPAIGN_CONFIGS), 30)
        self.assertEqual(sum(row.family == "dense" for row in CAMPAIGN_CONFIGS), 15)
        self.assertEqual(sum(row.family == "conv" for row in CAMPAIGN_CONFIGS), 15)
        self.assertEqual({row.tau for row in CAMPAIGN_CONFIGS}, {10.0, 20.0, 40.0})
        for row in CAMPAIGN_CONFIGS:
            row.validate()
            self.assertEqual(row.lut_rank, 2)
            self.assertEqual(row.final_logic_width % 2, 0)
            self.assertGreater(resource_proxies(row)["logic_node_proxy"], 0)

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
