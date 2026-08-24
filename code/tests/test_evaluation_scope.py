import sys
import unittest
from unittest.mock import patch

from core.config import (
    DEFAULT_RUN_SUFFIX,
    DEFAULT_TEST_GRAPHS,
    VALIDATION_SELECTED_FINETUNED_MODELS,
)
from core.graph_config import GRAPH_CONFIGS
from core.utils import validate_fine_tuned_model_set
from evaluation.evaluation import (
    get_validation_dimensions,
    parse_args as parse_evaluation_args,
)
from evaluation.f1_significance_test import DEFAULT_REFERENCE_DIMENSION
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
    def test_v4_current_generation_defaults(self):
        self.assertEqual(DEFAULT_RUN_SUFFIX, "v4")
        self.assertEqual(DEFAULT_REFERENCE_DIMENSION, 64)
        self.assertEqual(
            DEFAULT_TEST_GRAPHS,
            ("causenet", "causenet_full", "ceg"),
        )
        self.assertEqual(
            [
                (
                    config["checkpoint_name"],
                    config["embedding_dimension"],
                    config["existing_validation_budget"],
                )
                for config in VALIDATION_SELECTED_FINETUNED_MODELS
            ],
            [
                (
                    "all-mpnet-base-v2_relu_cosine_nonorm_"
                    "matryoshka_v4_finetuned",
                    64,
                    14,
                ),
                (
                    "bge-large-en-v1.5_relu_cosine_nonorm_"
                    "matryoshka_v4_finetuned",
                    8,
                    27,
                ),
                (
                    "mxbai-embed-large-v1_relu_cosine_nonorm_"
                    "matryoshka_v4_finetuned",
                    8,
                    20,
                ),
                (
                    "Qwen3-Embedding-0.6B_relu_cosine_nonorm_"
                    "matryoshka_v4_finetuned",
                    768,
                    6,
                ),
                (
                    "granite-embedding-english-r2_relu_euclid_nonorm_"
                    "matryoshka_v4_finetuned",
                    64,
                    23,
                ),
            ],
        )
        for graph_name in ("causenet", "causenet_full", "ceg"):
            self.assertEqual(GRAPH_CONFIGS[graph_name]["bfs_p95_cap"], 1_316)
            self.assertIn("/v4 ", GRAPH_CONFIGS[graph_name]["bfs_p95_cap_source"])

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
        self.assertFalse(default_args.pretrained_native_only)

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
        self.assertFalse(scoped_args.pretrained_native_only)

        with patch.object(
            sys,
            "argv",
            [
                "evaluation",
                "dataset.json",
                "--run-suffix",
                "v4",
                "--pretrained-native-only",
            ],
        ):
            pretrained_args = parse_evaluation_args()
        self.assertFalse(pretrained_args.fine_tuned_only)
        self.assertTrue(pretrained_args.pretrained_native_only)

    def test_validation_model_scope_flags_are_mutually_exclusive(self):
        incompatible_flags = (
            ("--fine-tuned-only", "--pretrained-native-only"),
            ("--fine-tuned-only", "--ablation"),
            ("--pretrained-native-only", "--ablation"),
        )

        for first_flag, second_flag in incompatible_flags:
            with self.subTest(first_flag=first_flag, second_flag=second_flag):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "evaluation",
                        "dataset.json",
                        "--run-suffix",
                        "v4",
                        first_flag,
                        second_flag,
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        parse_evaluation_args()

    def test_pretrained_native_scope_uses_only_full_dimension(self):
        self.assertEqual(get_validation_dimensions(768, True), [768])
        self.assertEqual(get_validation_dimensions(1024, True), [1024])
        self.assertEqual(
            get_validation_dimensions(768),
            [768, 512, 256, 128, 64, 32, 16, 8, 4, 2],
        )


if __name__ == "__main__":
    unittest.main()
