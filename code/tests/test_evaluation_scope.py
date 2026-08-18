import sys
import unittest
from unittest.mock import patch

from core.utils import validate_fine_tuned_model_set
from evaluation.evaluation import parse_args as parse_evaluation_args
from evaluation.visited_nodes_analysis import (
    get_embedding_strategy_items,
    parse_args as parse_analysis_args,
    validate_expected_analysis_results,
)


BASE_MODELS = (
    "org/model-a",
    "org/model-b",
)


def fine_tuned_path(base_model, *, activation="relu"):
    model_name = base_model.rsplit("/", 1)[-1]
    return (
        f"/models/{model_name}_{activation}_cosine_nonorm_"
        "matryoshka_v4_finetuned"
    )


class FineTunedScopeTests(unittest.TestCase):
    def test_complete_model_set_requires_exactly_one_model_per_family(self):
        validate_fine_tuned_model_set(
            [fine_tuned_path(model) for model in BASE_MODELS],
            BASE_MODELS,
            "v4",
        )

        with self.assertRaisesRegex(ValueError, "missing=.*model-b"):
            validate_fine_tuned_model_set(
                [fine_tuned_path(BASE_MODELS[0])],
                BASE_MODELS,
                "v4",
            )

        with self.assertRaisesRegex(ValueError, "duplicates=.*model-a"):
            validate_fine_tuned_model_set(
                [
                    fine_tuned_path(BASE_MODELS[0]),
                    fine_tuned_path(BASE_MODELS[0], activation="gelu"),
                    fine_tuned_path(BASE_MODELS[1]),
                ],
                BASE_MODELS,
                "v4",
            )

    def test_analysis_scope_flags_are_opt_in(self):
        with patch.object(
            sys,
            "argv",
            ["visited_nodes_analysis", "dataset.json", "--run-suffix", "v4"],
        ):
            default_args = parse_analysis_args()

        self.assertFalse(default_args.fine_tuned_only)
        self.assertFalse(default_args.skip_dijkstra)
        self.assertFalse(default_args.skip_baselines)

        with patch.object(
            sys,
            "argv",
            [
                "visited_nodes_analysis",
                "dataset.json",
                "--run-suffix",
                "v4",
                "--fine-tuned-only",
                "--skip-dijkstra",
                "--skip-baselines",
            ],
        ):
            scoped_args = parse_analysis_args()

        self.assertTrue(scoped_args.fine_tuned_only)
        self.assertTrue(scoped_args.skip_dijkstra)
        self.assertTrue(scoped_args.skip_baselines)

    def test_skip_dijkstra_keeps_only_astar(self):
        self.assertEqual(
            [name for name, _ in get_embedding_strategy_items(True)],
            ["A*"],
        )
        self.assertEqual(
            [name for name, _ in get_embedding_strategy_items(False)],
            ["A*", "Dijkstra"],
        )

    def test_analysis_completeness_check_detects_missing_rows(self):
        expected = {
            ("model-a_v4_finetuned", 2, "A*"),
            ("model-a_v4_finetuned", 4, "A*"),
        }
        results = [
            {
                "model": "model-a_v4_finetuned",
                "dimension": 2,
                "analysis": {"strategy": "A*"},
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "missing.*4"):
            validate_expected_analysis_results(results, expected)

        results.append(
            {
                "model": "model-a_v4_finetuned",
                "dimension": 4,
                "analysis": {"strategy": "A*"},
            }
        )
        validate_expected_analysis_results(results, expected)

    def test_validation_fine_tuned_only_flag_is_opt_in(self):
        with patch.object(
            sys,
            "argv",
            ["evaluation", "dataset.json", "--run-suffix", "v4"],
        ):
            default_args = parse_evaluation_args()
        self.assertFalse(default_args.fine_tuned_only)

        with patch.object(
            sys,
            "argv",
            [
                "evaluation",
                "dataset.json",
                "--run-suffix",
                "v4",
                "--fine-tuned-only",
            ],
        ):
            scoped_args = parse_evaluation_args()
        self.assertTrue(scoped_args.fine_tuned_only)


if __name__ == "__main__":
    unittest.main()
