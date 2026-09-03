from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np

from internalBitOverwriteStudy.campaign import (
    COUNTS_PER_CHUNK,
    DEFAULT_TRAINING_PERCENTAGES,
    EVALUATION_PERCENTAGES,
    FAULT_OPERATION,
    FAULT_SCOPE,
    MODELS,
    evaluation_chunks,
    evaluation_counts,
    evaluation_tasks,
    combine_campaign,
    parse_training_percentages,
    training_tasks,
    write_campaign_config,
)
from qkerasStudies.fault_injection import (
    QuantizationRange,
    _apply_exact_stage_bit_faults,
    _stage_ranges,
    stage_metadata,
)
from qkerasStudies.models import get_config


class InternalOverwriteCampaignTest(unittest.TestCase):
    def test_selected_models_and_internal_site_counts(self):
        lnn, qkeras = MODELS
        self.assertEqual(lnn.alias, "s02_p163M_tau20")
        self.assertTrue(lnn.registry_name.endswith("s02_p163M_20"))
        self.assertEqual(lnn.internal_fault_sites, 34_650)
        self.assertEqual(qkeras.alias, "qkeras_y256_64_w4a8")
        self.assertEqual(qkeras.internal_fault_sites, 2_560)
        metadata = stage_metadata(get_config(qkeras.alias), "internal")
        self.assertEqual(tuple(metadata["eligible_model_stage_indices"]), (1, 2))
        self.assertEqual(tuple(metadata["stage_widths"]), (256, 64))
        ranges = _stage_ranges(
            get_config(qkeras.alias), QuantizationRange(-2.0, 3.0)
        )
        self.assertEqual(ranges[0], QuantizationRange(-2.0, 3.0))
        self.assertEqual(ranges[1].minimum, 0.0)
        self.assertEqual(ranges[1].maximum, 255 / 256)

    def test_default_task_grid(self):
        self.assertEqual(DEFAULT_TRAINING_PERCENTAGES, (0.0, 1.0, 3.0, 10.0))
        self.assertEqual(len(training_tasks()), 8)
        chunks_per_condition = (len(EVALUATION_PERCENTAGES) - 1) // COUNTS_PER_CHUNK
        self.assertEqual(chunks_per_condition, 5)
        self.assertEqual(len(evaluation_tasks()), 40)
        for model in MODELS:
            counts = evaluation_counts(model.internal_fault_sites)
            self.assertEqual(len(counts), len(EVALUATION_PERCENTAGES))
            self.assertEqual(counts[0], 0)
            self.assertEqual(len(evaluation_chunks(model.internal_fault_sites)), 5)

    def test_training_grid_parser_is_extension_ready(self):
        self.assertEqual(parse_training_percentages("0, 1 2.5 10"), (0.0, 1.0, 2.5, 10.0))
        with self.assertRaises(ValueError):
            parse_training_percentages("1 3 10")

    def test_campaign_config_records_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_campaign_config(root, "abc123")
            text = (root / "campaign_config.json").read_text()
            self.assertIn(f'"fault_scope": "{FAULT_SCOPE}"', text)
            self.assertIn(f'"fault_operation": "{FAULT_OPERATION}"', text)
            self.assertIn('"training_task_count": 8', text)
            self.assertIn('"evaluation_task_count": 40', text)

    def test_qkeras_random_set_assigns_requested_target_bits(self):
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow is not installed")
        values = tf.constant([[0.0]], dtype=tf.float32)
        # With two bits in [0,1], uint 0 has both bits clear. Selecting both,
        # then assigning targets [0,1], must yield uint 2 => 2/3.
        actual = _apply_exact_stage_bit_faults(
            values,
            tf.constant([[0, 1]], dtype=tf.int64),
            stage_offset=0,
            bits=2,
            value_range=QuantizationRange(0.0, 1.0),
            fault_operation="random_set",
            fault_values=tf.constant([[0, 1]], dtype=tf.int32),
        )
        self.assertTrue(np.allclose(actual.numpy(), [[2 / 3]], atol=1e-7))

    def test_complete_synthetic_campaign_combines_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            write_campaign_config(root, "abc123")
            for model in MODELS:
                counts = evaluation_counts(model.internal_fault_sites)
                for percentage in DEFAULT_TRAINING_PERCENTAGES:
                    tag = "baseline_clean" if percentage == 0 else f"train_{percentage:g}pct"
                    condition = root / model.family / tag / model.alias
                    training = condition / "training"
                    training.mkdir(parents=True)
                    checkpoint = training / (
                        "final_model.pth" if model.family == "lnn" else "final_model.keras"
                    )
                    checkpoint.touch()
                    status = {
                        "complete": True,
                        "model_alias": model.alias,
                        "registry_name": model.registry_name,
                        "training_fault_scope": FAULT_SCOPE,
                        "fault_operation": FAULT_OPERATION,
                        "total_fault_sites_per_event": model.internal_fault_sites,
                        "training_fault_percentage": percentage,
                    }
                    (training / "training_complete.json").write_text(
                        json.dumps(status)
                    )
                    for chunk_index, chunk_counts in enumerate(
                        evaluation_chunks(model.internal_fault_sites)
                    ):
                        chunk = condition / "chunks" / f"chunk_{chunk_index:02d}"
                        chunk.mkdir(parents=True)
                        rows = []
                        for count in (0, *chunk_counts):
                            repeats = 1 if count == 0 else 5
                            for repeat in range(repeats):
                                rows.append(
                                    {
                                        "model_alias": model.alias,
                                        "registry_name": model.registry_name,
                                        "flip_count_per_event": count,
                                        "total_fault_sites_per_event": model.internal_fault_sites,
                                        "equivalent_fault_fraction": count / model.internal_fault_sites,
                                        "equivalent_fault_percentage": 100 * count / model.internal_fault_sites,
                                        "repeat": repeat,
                                        "fault_seed": 42 + repeat,
                                        "balanced_accuracy": 0.8 - 0.3 * count / model.internal_fault_sites,
                                    }
                                )
                        with (chunk / "bit_flip_results.csv").open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                        metadata = {
                            "complete": True,
                            "model_alias": model.alias,
                            "registry_name": model.registry_name,
                            "fault_scope": FAULT_SCOPE,
                            "fault_operation": FAULT_OPERATION,
                            "total_fault_sites_per_event": model.internal_fault_sites,
                            "flip_counts_per_event": [0, *chunk_counts],
                            "repeats_per_nonzero_count": 5,
                        }
                        (chunk / "bit_flip_metadata.json").write_text(
                            json.dumps(metadata)
                        )
                        np.save(chunk / "clean_predictions.npy", np.asarray([0, 1, 2], dtype=np.uint8))

            combine_campaign(root)
            self.assertTrue((root / "lnn_vs_qkeras_internal_random_overwrite.png").is_file())
            self.assertTrue((root / "lnn_vs_qkeras_internal_random_overwrite.pdf").is_file())
            with (root / "internal_random_overwrite_plot_points.csv").open() as handle:
                points = list(csv.DictReader(handle))
            self.assertEqual(
                len(points),
                len(MODELS)
                * len(DEFAULT_TRAINING_PERCENTAGES)
                * len(EVALUATION_PERCENTAGES),
            )
            self.assertEqual(
                {row["fault_scope"] for row in points},
                {"internal"},
            )
            self.assertEqual(
                {row["fault_operation"] for row in points},
                {"random_set"},
            )


if __name__ == "__main__":
    unittest.main()
