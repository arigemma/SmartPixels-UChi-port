from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from inputBitFlipStudy.campaign import (
    EVALUATION_PERCENTAGES,
    MODELS,
    PAPER_REFINED_EVALUATION_PERCENTAGES,
    PAPER_REFINEMENT_PERCENTAGES,
    REFINED_EVALUATION_PERCENTAGES,
    REFINEMENT_PERCENTAGES,
    TRAINING_PERCENTAGES,
    _combine_refined_source,
    _combine_qkeras_source,
    combine_refined_campaign,
    combine_paper_refined_campaign,
    evaluation_flip_count_chunks,
    evaluation_flip_counts,
    evaluation_tasks,
    refinement_flip_counts,
    refinement_tasks,
    paper_refinement_flip_counts,
    paper_refinement_tasks,
    percentage_tag,
    training_tasks,
)
from lgnStudies.bitFlipStudy.fault_injection import (
    sample_unique_fault_indices as sample_lgn_indices,
)
from lgnStudies.bitFlipStudy.plots import _configure_x_axis
from lgnStudies.bitFlipStudy.study import SUPPORTED_MODEL_HIDDEN_DIMS
from models.registry import REGISTRY
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

    def test_low_percentage_refinement_grid(self):
        self.assertEqual(REFINEMENT_PERCENTAGES, (2.5, 7.5))
        self.assertEqual(
            REFINED_EVALUATION_PERCENTAGES[:5],
            (0.0, 2.5, 5.0, 7.5, 10.0),
        )
        self.assertEqual(len(refinement_tasks()), 22)
        self.assertEqual(refinement_flip_counts(1400), (35, 105))
        self.assertEqual(refinement_flip_counts(112), (3, 8))
        for model in MODELS:
            self.assertTrue(
                set(refinement_flip_counts(model.input_fault_sites)).isdisjoint(
                    evaluation_flip_counts(model.input_fault_sites)
                )
            )

    def test_five_point_paper_refinement_grid(self):
        self.assertEqual(
            PAPER_REFINEMENT_PERCENTAGES,
            (1.25, 3.75, 12.5, 17.5, 22.5),
        )
        self.assertEqual(len(paper_refinement_tasks()), 22)
        self.assertEqual(
            PAPER_REFINED_EVALUATION_PERCENTAGES[:7],
            (0.0, 1.25, 2.5, 3.75, 5.0, 7.5, 10.0),
        )
        self.assertEqual(
            paper_refinement_flip_counts(1400),
            (18, 52, 175, 245, 315),
        )
        self.assertEqual(
            paper_refinement_flip_counts(112),
            (1, 4, 14, 20, 25),
        )
        for model in MODELS:
            previous = {
                *evaluation_flip_counts(model.input_fault_sites),
                *refinement_flip_counts(model.input_fault_sites),
            }
            self.assertTrue(
                set(
                    paper_refinement_flip_counts(model.input_fault_sites)
                ).isdisjoint(previous)
            )

    def test_refinement_merge_preserves_base_and_adds_only_new_points(self):
        model = MODELS[0]

        def rows_for_counts(counts):
            rows = []
            for count in counts:
                for repeat in (range(1) if count == 0 else range(5)):
                    rows.append(
                        {
                            "model_alias": model.alias,
                            "registry_name": model.registry_name,
                            "flip_count_per_event": str(count),
                            "total_fault_sites_per_event": str(
                                model.input_fault_sites
                            ),
                            "equivalent_fault_percentage": str(
                                100 * count / model.input_fault_sites
                            ),
                            "repeat": str(repeat),
                            "fault_seed": str(42 + count + repeat),
                            "balanced_accuracy": "0.8",
                        }
                    )
            return rows

        def write_csv(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = (
                root / model.family / percentage_tag(None) / model.alias
            )
            base_csv = model_root / "bit_flip_results.csv"
            refinement_root = model_root / "refinement_0to10pct"
            refinement_csv = refinement_root / "bit_flip_results.csv"
            base_counts = evaluation_flip_counts(model.input_fault_sites)
            extra_counts = refinement_flip_counts(model.input_fault_sites)
            write_csv(base_csv, rows_for_counts(base_counts))
            write_csv(
                refinement_csv,
                rows_for_counts((0, *extra_counts)),
            )
            predictions = np.asarray([0, 1, 2, 1], dtype=np.int64)
            np.save(model_root / "clean_predictions.npy", predictions)
            np.save(refinement_root / "clean_predictions.npy", predictions)

            merged = _combine_refined_source(
                root,
                model,
                percentage_tag(None),
            )

            observed_counts = {
                int(row["flip_count_per_event"]) for row in merged
            }
            self.assertEqual(
                observed_counts,
                set((*base_counts, *extra_counts)),
            )
            self.assertEqual(len(merged), len(rows_for_counts(base_counts)) + 10)
            self.assertTrue(
                (model_root / "bit_flip_results_refined_0to10pct.csv").is_file()
            )
            with base_csv.open(newline="") as handle:
                self.assertEqual(
                    len(list(csv.DictReader(handle))),
                    len(rows_for_counts(base_counts)),
                )

    def test_original_qkeras_combiner_still_delegates_and_returns(self):
        source = inspect.getsource(_combine_qkeras_source)
        self.assertIn("return write_combined_artifacts(", source)

    def test_complete_refinement_combine_writes_new_plots(self):
        def rows_for_counts(model, counts):
            rows = []
            for count in counts:
                for repeat in (range(1) if count == 0 else range(5)):
                    rows.append(
                        {
                            "model_alias": model.alias,
                            "registry_name": model.registry_name,
                            "flip_count_per_event": str(count),
                            "total_fault_sites_per_event": str(
                                model.input_fault_sites
                            ),
                            "equivalent_fault_percentage": str(
                                100 * count / model.input_fault_sites
                            ),
                            "repeat": str(repeat),
                            "fault_seed": str(42 + count + repeat),
                            "balanced_accuracy": str(0.8 - count / 100_000),
                            "prediction_change_fraction": "0.1",
                            "balanced_repair_gain": "0.01",
                            "balanced_damage_loss": "0.02",
                            "recall_class_0": "0.79",
                            "recall_class_1": "0.80",
                            "recall_class_2": "0.81",
                        }
                    )
            return rows

        def write_csv(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input_bit_flip_campaign_summary.json").write_text("{}\n")
            predictions = np.asarray([0, 1, 2, 1], dtype=np.int64)
            for model in MODELS:
                base_counts = evaluation_flip_counts(model.input_fault_sites)
                extra_counts = refinement_flip_counts(model.input_fault_sites)
                for percentage in (None, *TRAINING_PERCENTAGES):
                    model_root = (
                        root
                        / model.family
                        / percentage_tag(percentage)
                        / model.alias
                    )
                    refinement_root = model_root / "refinement_0to10pct"
                    write_csv(
                        model_root / "bit_flip_results.csv",
                        rows_for_counts(model, base_counts),
                    )
                    write_csv(
                        refinement_root / "bit_flip_results.csv",
                        rows_for_counts(model, (0, *extra_counts)),
                    )
                    np.save(model_root / "clean_predictions.npy", predictions)
                    np.save(
                        refinement_root / "clean_predictions.npy",
                        predictions,
                    )

            combine_refined_campaign(root)

            self.assertTrue(
                (
                    root
                    / "input_bit_flip_campaign_refined_0to10pct_summary.json"
                ).is_file()
            )
            for model in MODELS:
                stem = root / (
                    f"{model.alias}_input_bit_flip_robustness_linear_"
                    "refined_0to10pct"
                )
                self.assertTrue(stem.with_suffix(".png").is_file())
                self.assertTrue(stem.with_suffix(".pdf").is_file())

            for model in MODELS:
                paper_counts = paper_refinement_flip_counts(
                    model.input_fault_sites
                )
                for percentage in (None, *TRAINING_PERCENTAGES):
                    model_root = (
                        root
                        / model.family
                        / percentage_tag(percentage)
                        / model.alias
                    )
                    paper_root = model_root / "paper_refinement_0to25pct"
                    write_csv(
                        paper_root / "bit_flip_results.csv",
                        rows_for_counts(model, (0, *paper_counts)),
                    )
                    np.save(
                        paper_root / "clean_predictions.npy",
                        predictions,
                    )

            combine_paper_refined_campaign(root)

            self.assertTrue(
                (
                    root
                    / "input_bit_flip_campaign_paper_0to50pct_summary.json"
                ).is_file()
            )
            for model in MODELS:
                stem = root / (
                    f"{model.alias}_input_bit_flip_robustness_"
                    "paper_0to50pct"
                )
                self.assertTrue(stem.with_suffix(".png").is_file())
                self.assertTrue(stem.with_suffix(".pdf").is_file())
                points_path = stem.with_name(stem.name + "_points.json")
                self.assertTrue(points_path.is_file())
                payload = json.loads(points_path.read_text())
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(payload["model"]["alias"], model.alias)
                self.assertEqual(payload["axes"]["x"]["range"], [0.0, 50.0])
                self.assertEqual(len(payload["curves"]), 11)
                self.assertEqual(
                    payload["curves"][0]["style"],
                    {"color": "#333333", "linestyle": "-", "marker": "o"},
                )
                self.assertEqual(
                    payload["curves"][1]["training_fault_percentage"],
                    1.0,
                )
                self.assertTrue(
                    all(
                        point["x_percent"] <= 50.0
                        for curve in payload["curves"]
                        for point in curve["points"]
                    )
                )

    def test_exact_model_selection(self):
        self.assertEqual(MODELS[0].alias, "s02_p163M_tau20")
        self.assertTrue(MODELS[0].registry_name.endswith("s02_p163M_20"))
        self.assertEqual(MODELS[1].alias, "qkeras_y256_64_w4a8")
        self.assertEqual(MODELS[0].input_fault_sites, 1400)
        self.assertEqual(MODELS[1].input_fault_sites, 112)

    def test_selected_lgn_is_supported_by_inference_evaluator(self):
        model_class = REGISTRY[MODELS[0].registry_name]["class"]
        self.assertIn(tuple(model_class.hidden_dims), SUPPORTED_MODEL_HIDDEN_DIMS)

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
        resume_launcher = (root / "resume_failed_evaluation.sh").read_text()
        self.assertIn(
            f"#SBATCH --array=0-{len(training_tasks()) - 1}",
            training_slurm,
        )
        self.assertIn(
            f"#SBATCH --array=0-{len(evaluation_tasks()) - 1}",
            evaluation_slurm,
        )
        self.assertIn('FAILED_TASKS="${2:-0-43,84-87}"', resume_launcher)


if __name__ == "__main__":
    unittest.main()
