import unittest
from pathlib import Path

from core import model_registry
from core.constants import EMBEDDINGS_DIR
from core.utils import (
    get_dimension_embedding_cache_path,
    get_embedding_cache_path,
    get_embedding_cache_vectors_path,
)
from evaluation.run_budget_tradeoff import MODEL_CONFIGS


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


if __name__ == "__main__":
    unittest.main()
