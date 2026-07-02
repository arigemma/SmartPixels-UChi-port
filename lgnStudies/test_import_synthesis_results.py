import unittest

from lgnStudies.import_synthesis_results import join_stage_results


class ImportSynthesisResultsTest(unittest.TestCase):
    def test_joins_only_actual_vivado_rows(self):
        training = [
            {"step": "0", "balanced_accuracy": "0.4"},
            {"step": "200", "balanced_accuracy": "0.8"},
        ]
        synthesis = [
            {
                "model_name": "s09_p1460M_tau40_step000000",
                "lut": "1000",
                "ff": "200",
                "dsp": "0",
                "source": "vivado_fast",
                "clock_ns": "4.16",
            },
            {
                "model_name": "s09_p1460M_tau40_step000200",
                "lut": "1500",
                "ff": "300",
                "dsp": "0",
                "source": "vivado",
                "clock_ns": "4.16",
            },
        ]
        joined = join_stage_results(training, synthesis, "s09_p1460M_tau40")
        self.assertEqual([row["vivado_lut"] for row in joined], [1000, 1500])

    def test_rejects_estimates_and_missing_steps(self):
        training = [{"step": "0", "balanced_accuracy": "0.4"}]
        synthesis = [
            {
                "model_name": "s09_p1460M_tau40_step000000",
                "lut": "1000",
                "source": "da4ml_report",
            }
        ]
        with self.assertRaises(ValueError):
            join_stage_results(training, synthesis, "s09_p1460M_tau40")


if __name__ == "__main__":
    unittest.main()
