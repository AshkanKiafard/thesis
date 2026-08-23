import unittest
from pathlib import Path

from core import model_registry
from core.constants import EMBEDDINGS_DIR
from core.utils import (
    get_dimension_embedding_cache_path,
    get_embedding_cache_path,
    get_embedding_cache_vectors_path,
)
from evaluation.run_budget_tradeoff import MODEL_CONFIGS, build_v4_model_configs


class EmbeddingCachePathTest(unittest.TestCase):
    def test_cache_path_helpers_match_persisted_filename_convention(self):
        base_cache = get_embedding_cache_path(
            Path("embeddings"),
            Path("models") / "checkpoint_finetuned",
        )
        dimension_cache = get_dimension_embedding_cache_path(base_cache, 128)

        self.assertEqual(
            base_cache,
            Path("embeddings/checkpoint_finetuned_embeddings.npy"),
        )
        self.assertEqual(
            get_embedding_cache_vectors_path(base_cache),
            Path("embeddings/checkpoint_finetuned_embeddings_vectors.npy"),
        )
        self.assertEqual(
            get_embedding_cache_vectors_path(dimension_cache),
            Path(
                "embeddings/"
                "checkpoint_finetuned_dim128_embeddings_vectors.npy"
            ),
        )

    def test_budget_tradeoff_paths_match_runtime_dimension_selection(self):
        for config in MODEL_CONFIGS:
            with self.subTest(model=config.model):
                registered_model = model_registry.get_embedding_model(
                    config.checkpoint_name
                )
                self.assertIsNotNone(registered_model)

                runtime_cache = get_embedding_cache_path(
                    EMBEDDINGS_DIR,
                    config.checkpoint_path,
                )
                if config.embedding_dimension < registered_model.full_dimension:
                    runtime_cache = get_dimension_embedding_cache_path(
                        runtime_cache,
                        config.embedding_dimension,
                    )
                runtime_vectors = get_embedding_cache_vectors_path(runtime_cache)

                self.assertEqual(config.embedding_path, runtime_vectors)

    def test_v4_tradeoff_configs_come_from_validation_family_summaries(self):
        family_inputs = (
            ("MPNet", "all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v4_finetuned", 128, 31),
            ("BGE", "bge-large-en-v1.5_gelu_euclid_nonorm_matryoshka_v4_finetuned", 256, 47),
            ("MXBAI", "mxbai-embed-large-v1_relu_euclid_nonorm_matryoshka_v4_finetuned", 512, 59),
            ("Qwen", "Qwen3-Embedding-0.6B_gelu_cosine_nonorm_matryoshka_v4_finetuned", 64, 23),
            ("Granite", "granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned", 32, 19),
        )
        family_summaries = []
        selected = None
        for family, checkpoint, dimension, budget in family_inputs:
            candidate = {
                "model_path": f"data/models/lightning/{checkpoint}",
                "family": family,
                "dimension": dimension,
                "astar_max_visits": budget,
            }
            if family == "BGE":
                selected = {
                    **candidate,
                    "dimension": 512,
                    "astar_max_visits": 53,
                }
            family_summaries.append(
                {
                    "family": family,
                    "tradeoff_candidate": candidate,
                }
            )

        configs = build_v4_model_configs(
            {
                "best": selected,
                "family_summaries": family_summaries,
            }
        )

        self.assertEqual(len(configs), 5)
        self.assertTrue(
            all(config.checkpoint_name.endswith("_v4_finetuned") for config in configs)
        )
        self.assertEqual(
            [config.existing_validation_budget for config in configs],
            [31, 53, 59, 23, 19],
        )
        self.assertEqual(configs[1].embedding_dimension, 512)
        self.assertEqual(configs[1].activation_function, "GELU")
        self.assertEqual(configs[1].distance_metric, "Euclidean")


if __name__ == "__main__":
    unittest.main()
