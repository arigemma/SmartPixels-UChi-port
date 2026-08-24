import unittest
from pathlib import Path

from models.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
TAUS = (10, 20, 40)
CONFIGS = (
    ("s01_p003M", (16, 32, 64, 60), 300, 0, 2_964_000),
    ("s02_p007M", (24, 48, 96, 90), 450, 0, 6_568_200),
    ("s03_p012M", (32, 64, 128, 120), 600, 0, 11_587_200),
    ("s04_p018M", (40, 80, 160, 150), 750, 0, 18_021_000),
    ("s05_p026M", (48, 96, 192, 180), 900, 0, 25_869_600),
    ("s06_p039M", (48, 96, 192, 180), 900, 1, 38_829_600),
    ("s07_p069M", (64, 128, 256, 240), 1200, 1, 68_851_200),
    ("s08_p110M", (96, 192, 384, 360), 1500, 1, 110_220_000),
    ("s09_p213M", (128, 256, 512, 480), 1800, 2, 213_177_600),
    ("s10_p385M", (192, 384, 768, 720), 2400, 2, 384_979_200),
)


def connectivity_proxy(model_class) -> int:
    k1, k2, k3, k4 = (
        model_class.k1,
        model_class.k2,
        model_class.k3,
        model_class.k4,
    )
    fc_base = model_class.fc_base
    h1 = 4 * fc_base
    h2 = 2 * fc_base
    fc_in = model_class.n_bits + 6 * k4
    convolutional = 9 * (
        model_class.n_bits * k1 + k1 * k2 + k2 * k3 + k3 * k4
    )
    dense = (
        fc_in * h1
        + h1 * h1
        + h1 * h2
        + h2 * fc_base
        + model_class.extra_dense_layers * h1 * h1
    )
    return convolutional + dense


class Model3FrontierCampaignTest(unittest.TestCase):
    def test_complete_registry_and_architectures(self):
        observed_proxies = []
        for size, widths, fc_base, extra_layers, expected_proxy in CONFIGS:
            for tau in TAUS:
                name = f"lgn-conv-3-100-orpool-model3Frontier_{size}_{tau}"
                self.assertIn(name, REGISTRY)
                entry = REGISTRY[name]
                self.assertEqual(entry["input_type"], "y-profile-timing")
                self.assertEqual(entry["task_type"], "classification")
                model_class = entry["class"]
                self.assertEqual(
                    (model_class.k1, model_class.k2, model_class.k3, model_class.k4),
                    widths,
                )
                self.assertEqual(model_class.fc_base, fc_base)
                self.assertEqual(model_class.extra_dense_layers, extra_layers)
                self.assertEqual(model_class.tau, float(tau))
                self.assertEqual(connectivity_proxy(model_class), expected_proxy)
            observed_proxies.append(expected_proxy)
        self.assertEqual(observed_proxies, sorted(observed_proxies))

    def test_slurm_grids_reference_every_size(self):
        training = (ROOT / "train_model3_frontier.slurm").read_text()
        evaluation = (ROOT / "eval_model3_frontier.slurm").read_text()
        for size, *_rest in CONFIGS:
            self.assertIn(size, training)
            self.assertIn(size, evaluation)
        self.assertIn("#SBATCH --array=0-29%6", training)
        self.assertIn("--candidate-forward discrete", evaluation)


if __name__ == "__main__":
    unittest.main()
