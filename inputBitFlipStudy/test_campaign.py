from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from inputBitFlipStudy.campaign import (
    EVALUATION_PERCENTAGES,
    MODELS,
    TRAINING_PERCENTAGES,
    evaluation_flip_count_chunks,
    evaluation_flip_counts,
    evaluation_tasks,
    training_tasks,
)
from lgnStudies.bitFlipStudy.fault_injection import (
    sample_unique_fault_indices as sample_lgn_indices,
)
from lgnStudies.bitFlipStudy.plots import _configure_x_axis
from qkerasStudies.fault_injection import (
    sample_unique_fault_indices as sample_qkeras_indices,
)


class SelectedInputFaultCampaignTest(unittest.TestCase):
    def test_requested_training_and_evaluation_grids(self):
        self.assertEqual(TRAINING_PERCENTAGES, tuple(float(i) for i in range(1, 11)))
        self.assertEqual(EVALUATION_PERCENTAGES, tuple(float(i) for i in range(0, 61, 5)))
        self.assertEqual(len(training_tasks()), 22)
        self.assertEqual(len(evaluation_tasks()), 88)

        for model in MODELS:
            counts = evaluation_flip_counts(model.input_fault_sites)
            self.assertEqual(counts[0], 0)
            self.assertAlmostEqual(
                100 * counts[-1] / model.input_fault_sites,
                60.0,
                delta=0.5,
            )
            self.assertEqual(len(evaluation_flip_count_chunks(model.input_fault_sites)), 4)

    def test_exact_model_selection(self):
        self.assertEqual(MODELS[0].alias, "s02_p163M_tau20")
        self.assertTrue(MODELS[0].registry_name.endswith("s02_p163M_20"))
        self.assertEqual(MODELS[1].alias, "qkeras_y256_64_w4a8")
        self.assertEqual(MODELS[0].input_fault_sites, 1400)
        self.assertEqual(MODELS[1].input_fault_sites, 112)

    def test_high_fraction_samplers_are_unique(self):
        lgn = sample_lgn_indices(4, 100, 60, "cpu")
        self.assertEqual(tuple(lgn.shape), (4, 60))
        for row in lgn:
            self.assertEqual(len(torch.unique(row)), 60)

        qkeras = sample_qkeras_indices(4, 100, 60, np.random.default_rng(42))
        self.assertEqual(qkeras.shape, (4, 60))
        for row in qkeras:
            self.assertEqual(len(np.unique(row)), 60)

    def test_linear_percentage_axis_is_not_log_scaled(self):
        figure, axis = plt.subplots()
        _configure_x_axis(
            axis,
            np.asarray(EVALUATION_PERCENTAGES),
            "percentage",
            linear=True,
        )
        self.assertEqual(axis.get_xscale(), "linear")
        plt.close(figure)

    def test_slurm_array_sizes_match_campaign_tasks(self):
        root = Path(__file__).resolve().parent
        training_slurm = (root / "run_training.slurm").read_text()
        evaluation_slurm = (root / "run_evaluation.slurm").read_text()
        self.assertIn(
            f"#SBATCH --array=0-{len(training_tasks()) - 1}",
            training_slurm,
        )
        self.assertIn(
            f"#SBATCH --array=0-{len(evaluation_tasks()) - 1}",
            evaluation_slurm,
        )


if __name__ == "__main__":
    unittest.main()
