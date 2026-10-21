import unittest

from maiaStudies.campaign import CAMPAIGN_CONFIGS
from maiaStudies.summarize import (
    _assert_campaign_learned,
    _best_rows,
    _flatten,
    _legacy_native_configs,
    _wide_rows,
)


def _point(target, rejection):
    acceptance = 1.0 / rejection
    return {
        "threshold": 0.25,
        "signal_efficiency": target,
        "bib_acceptance": acceptance,
        "bib_rejection": rejection,
        "bib_rejection_fraction": 1.0 - acceptance,
    }


def _metrics(balanced_accuracy, rejection):
    return {
        "n_examples": 200,
        "n_signal": 100,
        "n_bib": 100,
        "roc_auc": 0.91,
        "average_precision": 0.90,
        "argmax": {
            **_point(0.82, rejection),
            "balanced_accuracy": balanced_accuracy,
        },
        "validation_derived_operating_points": {
            "signal_efficiency_90pct": _point(0.90, rejection),
            "signal_efficiency_95pct": _point(0.95, rejection - 1.0),
            "signal_efficiency_99pct": _point(0.99, rejection - 2.0),
        },
    }


class SummaryTest(unittest.TestCase):
    def test_chance_level_campaign_is_rejected(self):
        rows = [
            {"subset": "full_test", "balanced_accuracy": 0.50, "roc_auc": 0.49},
            {"subset": "full_test", "balanced_accuracy": 0.51, "roc_auc": 0.52},
        ]
        with self.assertRaisesRegex(RuntimeError, "chance level"):
            _assert_campaign_learned(rows)
        _assert_campaign_learned(
            [{"subset": "full_test", "balanced_accuracy": 0.70, "roc_auc": 0.80}]
        )

    def test_flatten_and_wide_table_preserve_physics_metrics(self):
        config = CAMPAIGN_CONFIGS[0]
        result = {
            "input_schema": "maia_model2_yprofile_model3_yprofile_timing8_v3",
            "checkpoint_step": 20_000,
            "trainable_parameters": 12_345,
            "logic_node_proxy": 1_950,
            "group_sum_inputs": config.final_logic_width,
            "thermometer_input_bits": 1_300,
            "checkpoint": "/tmp/best_model.pth",
            "timing_cut": {
                "test_signal_acceptance": 0.99,
                "test_bib_acceptance": 0.70,
            },
            "full_test": _metrics(0.78, 20.0),
            "timing_cut_test": _metrics(0.80, 25.0),
        }
        rows = [
            _flatten(config, result, "full_test"),
            _flatten(config, result, "timing_cut_test"),
        ]
        self.assertEqual(
            rows[0]["family_label"], "Canonical dense LNN (14 inputs)"
        )
        self.assertEqual(rows[0]["trainable_parameters"], 12_345)
        self.assertEqual(rows[0]["bib_rejection_at_90pct_signal_op"], 20.0)
        self.assertAlmostEqual(
            rows[0]["bib_rejection_fraction_at_99pct_signal_op"],
            1.0 - 1.0 / 18.0,
        )
        wide = _wide_rows(rows)
        self.assertEqual(len(wide), 1)
        self.assertEqual(wide[0]["full_test_balanced_accuracy"], 0.78)
        self.assertEqual(wide[0]["timing_cut_test_balanced_accuracy"], 0.80)

    def test_best_table_is_separated_by_model_family(self):
        rows = []
        for config, balanced_accuracy in (
            (CAMPAIGN_CONFIGS[0], 0.75),
            (CAMPAIGN_CONFIGS[1], 0.77),
            (CAMPAIGN_CONFIGS[15], 0.80),
        ):
            result = {
                "input_schema": "maia_model2_yprofile_model3_yprofile_timing8_v3",
                "checkpoint_step": 20_000,
                "trainable_parameters": 10_000 + config.task_index,
                "logic_node_proxy": 1_000 + config.task_index,
                "group_sum_inputs": config.final_logic_width,
                "thermometer_input_bits": (
                    config.raw_input_features * config.thresholds_per_feature
                ),
                "checkpoint": "/tmp/best_model.pth",
                "timing_cut": {
                    "test_signal_acceptance": 0.99,
                    "test_bib_acceptance": 0.70,
                },
                "full_test": _metrics(
                    balanced_accuracy, 20.0 + config.task_index
                ),
                "timing_cut_test": _metrics(
                    balanced_accuracy + 0.01, 22.0 + config.task_index
                ),
            }
            rows.extend(
                (
                    _flatten(config, result, "full_test"),
                    _flatten(config, result, "timing_cut_test"),
                )
            )
        best = _best_rows(rows)
        dense_balanced = next(
            row
            for row in best
            if row["family"] == "dense"
            and row["criterion"] == "highest balanced accuracy"
            and row["rank"] == 1
        )
        conv_balanced = next(
            row
            for row in best
            if row["family"] == "conv"
            and row["criterion"] == "highest balanced accuracy"
            and row["rank"] == 1
        )
        self.assertEqual(dense_balanced["alias"], CAMPAIGN_CONFIGS[1].alias)
        self.assertEqual(conv_balanced["alias"], CAMPAIGN_CONFIGS[15].alias)
        self.assertTrue(
            any(
                row["criterion"]
                == "highest BIB rejection fraction at 99% signal OP"
                for row in best
            )
        )

    def test_legacy_native_plot_configs_cannot_collide_with_canonical_aliases(self):
        configs = _legacy_native_configs()
        self.assertEqual(len(configs), 30)
        self.assertEqual(configs[0].alias, "dense_s01_tau10")
        self.assertEqual(configs[0].raw_input_features, 37)
        self.assertEqual(configs[15].alias, "conv_s01_tau10")
        self.assertEqual(configs[15].raw_input_features, 276)
        self.assertTrue(all(not row.alias.startswith("dense14_") for row in configs))
        self.assertTrue(all(not row.alias.startswith("conv105_") for row in configs))


if __name__ == "__main__":
    unittest.main()
