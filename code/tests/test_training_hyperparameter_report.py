import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reports.common import report_paths
from reports.training_hyperparameters import (
    best_improvement_event,
    choose_best_epoch,
    epoch_from_checkpoint_name,
)


class TrainingHyperparameterReportTests(unittest.TestCase):
    def test_report_paths_create_a_report_specific_subdirectory(self):
        with TemporaryDirectory() as temporary_directory:
            paths = report_paths("example_report", Path(temporary_directory))

            self.assertTrue(all(path.parent.name == "example_report" for path in paths))
            self.assertEqual(
                [path.name for path in paths],
                [
                    "example_report.json",
                    "example_report.csv",
                    "example_report.tex",
                ],
            )

    def test_best_improvement_uses_strict_last_improvement(self):
        history = [
            {"epoch": 0, "value": 5.0, "global_step": 10},
            {"epoch": 1, "value": 3.0, "global_step": 20},
            {"epoch": 2, "value": 3.0, "global_step": 30},
            {"epoch": 3, "value": 4.0, "global_step": 40},
        ]

        self.assertEqual(best_improvement_event(history)["epoch"], 1)

    def test_tensorboard_epoch_takes_precedence_and_is_verified(self):
        tensorboard = {
            "events": [
                {"epoch": 0, "value": 5.0, "global_step": 10},
                {"epoch": 1, "value": 3.0, "global_step": 20},
                {"epoch": 2, "value": 4.0, "global_step": 30},
            ]
        }
        checkpoint = {
            "loaded_checkpoint_epoch": None,
            "metadata_filename_epoch": 1,
            "stopped_epoch": None,
        }

        result = choose_best_epoch(tensorboard, checkpoint, 10, 50)

        self.assertEqual(result["value"], 1)
        self.assertEqual(result["source"], "tensorboard_validation_history")
        self.assertTrue(result["direct_sources_agree"])

    def test_early_stopping_arithmetic_is_last_resort(self):
        result = choose_best_epoch(
            {"events": []},
            {
                "loaded_checkpoint_epoch": None,
                "metadata_filename_epoch": None,
                "stopped_epoch": 17,
            },
            patience=10,
            max_epochs=50,
        )

        self.assertEqual(result["value"], 7)
        self.assertEqual(result["source"], "early_stopping_fallback")

    def test_checkpoint_filename_epoch_is_zero_based(self):
        path = r"C:\checkpoints\run\best-07-123.4567.ckpt"
        self.assertEqual(epoch_from_checkpoint_name(path), 7)


if __name__ == "__main__":
    unittest.main()
