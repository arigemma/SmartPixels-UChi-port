import tempfile
import unittest
from pathlib import Path

from hgqStudies.campaign import BETA_BY_PRECISION, MODEL_CONFIGS, config_for_task, manifest


class CampaignTest(unittest.TestCase):
    def test_registry_matches_qkeras_curve(self):
        self.assertEqual(len(MODEL_CONFIGS), 21)
        self.assertEqual([model.index for model in MODEL_CONFIGS], list(range(21)))
        self.assertEqual(len({model.alias for model in MODEL_CONFIGS}), 21)
        self.assertEqual(sum(model.input_type == "y-profile-adc2a" for model in MODEL_CONFIGS), 5)
        self.assertEqual(MODEL_CONFIGS[0].qkeras_peer, "qkeras-model-2-w2a4-adc2a")
        self.assertEqual(MODEL_CONFIGS[-1].hidden_units, (768, 256, 128))
        self.assertEqual(MODEL_CONFIGS[-1].beta0, BETA_BY_PRECISION["w5a10"])

    def test_reference_resource_accounting(self):
        first = config_for_task(0)
        self.assertEqual(first.reference_resources, 14250 + 5255)
        self.assertEqual(first.reference_balanced_accuracy, 0.720)
        self.assertEqual(manifest()["model_count"], 21)

    def test_invalid_task_is_clear(self):
        for task in (-1, 21):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "outside"):
                config_for_task(task)


if __name__ == "__main__":
    unittest.main()
