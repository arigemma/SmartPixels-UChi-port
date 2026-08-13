import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lgnStudies.select_best_tau_checkpoints import (
    read_evaluation_rows,
    select_best_rows,
    stage_selected_rows,
)


SIZES = ("s01_p319M", "s02_p465M")
TAUS = (20, 40, 60)


class SelectBestTauCheckpointsTest(unittest.TestCase):
    def _write_grid(self, root: Path, scores: dict[tuple[str, int], float]) -> Path:
        summary = root / "balanced_accuracy_summary.tsv"
        with summary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["label", "input_type", "balanced_acc", "path"],
                delimiter="\t",
            )
            writer.writeheader()
            for size in SIZES:
                for tau in TAUS:
                    checkpoint = root / "results" / size / f"tau_{tau}" / "best_model.pth"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(f"{size}-{tau}".encode())
                    relative = checkpoint.relative_to(root)
                    writer.writerow(
                        {
                            "label": f"{size}/tau_{tau} (discrete)",
                            "input_type": "y-profile-timing",
                            "balanced_acc": scores[(size, tau)],
                            "path": str(relative),
                        }
                    )
        return summary

    def test_selects_one_best_tau_per_size_and_stages_full_checkpoints(self):
        scores = {
            ("s01_p319M", 20): 0.81,
            ("s01_p319M", 40): 0.83,
            ("s01_p319M", 60): 0.82,
            ("s02_p465M", 20): 0.84,
            ("s02_p465M", 40): 0.82,
            ("s02_p465M", 60): 0.83,
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self._write_grid(root, scores)
            rows = read_evaluation_rows(summary, root)
            selected = select_best_rows(rows, SIZES, TAUS)
            self.assertEqual([(row.size, row.tau) for row in selected], [
                ("s01_p319M", 40),
                ("s02_p465M", 20),
            ])

            stage = stage_selected_rows(
                selected,
                summary,
                root / "transfer_ready",
                overwrite=False,
            )
            first = stage / "s01_p319M_tau40" / "best_model.pth"
            second = stage / "s02_p465M_tau20" / "best_model.pth"
            self.assertEqual(first.read_bytes(), b"s01_p319M-40")
            self.assertEqual(second.read_bytes(), b"s02_p465M-20")
            self.assertTrue((first.parent / "SHA256SUMS").is_file())
            self.assertTrue((second.parent / "SHA256SUMS").is_file())
            self.assertTrue((stage / "best_by_size.tsv").is_file())
            self.assertTrue((stage / "balanced_accuracy_summary.tsv").is_file())
            manifest = json.loads((stage / "selection_manifest.json").read_text())
            self.assertEqual(manifest["selected_count"], 2)

            # The transfer bundle must not share mutable file data with the
            # original training result, and a rerun must replace stale output.
            first.write_bytes(b"changed-transfer-copy")
            self.assertEqual(
                selected[0].resolved_checkpoint.read_bytes(), b"s01_p319M-40"
            )
            stage_selected_rows(
                selected,
                summary,
                stage,
                overwrite=True,
            )
            self.assertEqual(first.read_bytes(), b"s01_p319M-40")

    def test_rejects_an_incomplete_evaluation_grid(self):
        scores = {
            (size, tau): 0.8 for size in SIZES for tau in TAUS
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self._write_grid(root, scores)
            rows = read_evaluation_rows(summary, root)
            incomplete = [row for row in rows if not (row.size == SIZES[0] and row.tau == 60)]
            with self.assertRaisesRegex(ValueError, "missing"):
                select_best_rows(incomplete, SIZES, TAUS)

    def test_exact_score_tie_selects_smaller_tau(self):
        scores = {
            (size, tau): 0.8 for size in SIZES for tau in TAUS
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = self._write_grid(root, scores)
            selected = select_best_rows(
                read_evaluation_rows(summary, root), SIZES, TAUS
            )
            self.assertEqual([row.tau for row in selected], [20, 20])


if __name__ == "__main__":
    unittest.main()
