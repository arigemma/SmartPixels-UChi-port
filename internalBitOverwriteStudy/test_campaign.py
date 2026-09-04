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
    qkeras_retry_task_text,
    training_tasks,
    write_campaign_config,
    write_qkeras_retry_manifest,
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

            qkeras = next(model for model in MODELS if model.family == "qkeras")
            for percentage in DEFAULT_TRAINING_PERCENTAGES:
                tag = (
                    "baseline_clean"
                    if percentage == 0
                    else f"train_{percentage:g}pct"
                )
                for chunk_index in (3, 4):
                    metadata_path = (
                        root
                        / qkeras.family
                        / tag
                        / qkeras.alias
                        / "chunks"
                        / f"chunk_{chunk_index:02d}"
                        / "bit_flip_metadata.json"
                    )
                    metadata = json.loads(metadata_path.read_text())
                    metadata["complete"] = False
                    metadata_path.write_text(json.dumps(metadata))

                    csv_path = metadata_path.with_name("bit_flip_results.csv")
                    with csv_path.open(newline="") as handle:
                        timeout_rows = list(csv.DictReader(handle))
                    for row in timeout_rows:
                        selected = int(row["flip_count_per_event"])
                        if selected > 0:
                            row["fault_seed"] = str(
                                42
                                + 1_000_003 * selected
                                + int(row["repeat"])
                            )
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=list(timeout_rows[0]),
                        )
                        writer.writeheader()
                        writer.writerows(timeout_rows)

                # A timed-out artifact may contain two complete points and a
                # partially written third point. Only the former are eligible
                # for the explicitly preliminary plot.
                chunk_four = (
                    root
                    / qkeras.family
                    / tag
                    / qkeras.alias
                    / "chunks"
                    / "chunk_04"
                    / "bit_flip_results.csv"
                )
                with chunk_four.open(newline="") as handle:
                    partial_rows = list(csv.DictReader(handle))
                counts = sorted(
                    {
                        int(row["flip_count_per_event"])
                        for row in partial_rows
                        if int(row["flip_count_per_event"]) > 0
                    }
                )
                partial_rows = [
                    row
                    for row in partial_rows
                    if int(row["flip_count_per_event"]) != counts[-1]
                    or int(row["repeat"]) < 2
                ]
                with chunk_four.open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=list(partial_rows[0]),
                    )
                    writer.writeheader()
                    writer.writerows(partial_rows)

            combine_campaign(root, preliminary=True)
            preliminary_png = (
                root / "lnn_vs_qkeras_internal_random_overwrite_preliminary.png"
            )
            preliminary_points = (
                root / "internal_random_overwrite_preliminary_plot_points.csv"
            )
            self.assertTrue(preliminary_png.is_file())
            with preliminary_points.open() as handle:
                partial_points = list(csv.DictReader(handle))
            self.assertEqual(
                len(partial_points),
                len(DEFAULT_TRAINING_PERCENTAGES)
                * (len(EVALUATION_PERCENTAGES) + len(EVALUATION_PERCENTAGES) - 1),
            )
            self.assertTrue(
                any(
                    row["family"] == "qkeras"
                    and row["source_artifact_complete"] == "False"
                    for row in partial_points
                )
            )

            manifest_path = write_qkeras_retry_manifest(root)
            with manifest_path.open(newline="") as handle:
                retry_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(retry_rows), 24)
            self.assertIn(qkeras.alias, qkeras_retry_task_text(root, 0))

            for manifest_row in retry_rows:
                count = int(manifest_row["selected_site_count"])
                condition = (
                    root
                    / "qkeras"
                    / manifest_row["tag"]
                    / manifest_row["model_alias"]
                )
                chunk = (
                    condition
                    / "chunks"
                    / (
                        f"chunk_retry_{manifest_row['task_index']}_"
                        f"count_{count}"
                    )
                )
                chunk.mkdir(parents=True)
                rows = []
                for selected_count in (0, count):
                    repeats = 1 if selected_count == 0 else 5
                    for repeat in range(repeats):
                        rows.append(
                            {
                                "model_alias": qkeras.alias,
                                "registry_name": qkeras.registry_name,
                                "flip_count_per_event": selected_count,
                                "total_fault_sites_per_event": qkeras.internal_fault_sites,
                                "equivalent_fault_fraction": selected_count
                                / qkeras.internal_fault_sites,
                                "equivalent_fault_percentage": 100
                                * selected_count
                                / qkeras.internal_fault_sites,
                                "repeat": repeat,
                                "fault_seed": 42 + repeat,
                                "balanced_accuracy": 0.8
                                - 0.3
                                * selected_count
                                / qkeras.internal_fault_sites,
                            }
                        )
                with (chunk / "bit_flip_results.csv").open(
                    "w", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                metadata = {
                    "complete": True,
                    "model_alias": qkeras.alias,
                    "registry_name": qkeras.registry_name,
                    "fault_scope": FAULT_SCOPE,
                    "fault_operation": FAULT_OPERATION,
                    "total_fault_sites_per_event": qkeras.internal_fault_sites,
                    "flip_counts_per_event": [0, count],
                    "repeats_per_nonzero_count": 5,
                }
                (chunk / "bit_flip_metadata.json").write_text(
                    json.dumps(metadata)
                )
                np.save(
                    chunk / "clean_predictions.npy",
                    np.asarray([0, 1, 2], dtype=np.uint8),
                )

            combine_campaign(root)


if __name__ == "__main__":
    unittest.main()
