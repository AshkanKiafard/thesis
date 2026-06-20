import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.utils import (
    CAUSENET_FULL_NODE_UNIVERSE,
    MERGED_NODE_UNIVERSE,
    get_node_universe_path,
    write_node_universe,
)
from core.embeddings import (
    EmbeddingCacheValidationError,
    _close_memmap,
    get_embedding_cache_status,
    load_st_embedding_cache,
    load_st_embedding_cache_index,
    save_st_embedding_cache,
)


def split_cache_paths(cache_file, node_universe=MERGED_NODE_UNIVERSE):
    cache_stem = cache_file.with_suffix("")
    return {
        "texts": get_node_universe_path(cache_file.parent, node_universe),
        "legacy_texts": cache_stem.with_name(f"{cache_stem.name}_texts.jsonl"),
        "vectors": cache_stem.with_name(f"{cache_stem.name}_vectors.npy"),
    }


class EmbeddingCacheJsonlMmapTests(unittest.TestCase):
    def test_save_and_load_shared_universal_jsonl_mmap_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            node_order = ["alpha", "beta"]
            cache = {
                "alpha": np.array([1.0, 2.0, 3.0], dtype="float32"),
                "beta": np.array([4.0, 5.0, 6.0], dtype="float32"),
            }

            save_st_embedding_cache(
                cache_file,
                cache,
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=node_order,
            )
            paths = split_cache_paths(cache_file)

            self.assertFalse(cache_file.exists())
            self.assertFalse(paths["legacy_texts"].exists())
            self.assertTrue(paths["texts"].exists())
            self.assertTrue(paths["vectors"].exists())

            with open(paths["texts"], encoding="utf-8") as file:
                self.assertEqual(
                    [json.loads(line) for line in file],
                    node_order,
                )

            loaded = load_st_embedding_cache(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            self.assertEqual(set(loaded), {"alpha", "beta"})
            np.testing.assert_array_equal(loaded["alpha"], cache["alpha"])
            np.testing.assert_array_equal(loaded["beta"], cache["beta"])

            in_memory_cache, text_to_idx, vectors = load_st_embedding_cache_index(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            try:
                self.assertEqual(in_memory_cache, {})
                self.assertEqual(text_to_idx, {"alpha": 0, "beta": 1})
                self.assertIsInstance(vectors, np.memmap)
                self.assertEqual(vectors.shape, (2, 3))
                np.testing.assert_array_equal(vectors[1], cache["beta"])
            finally:
                _close_memmap(vectors)

    def test_multiple_models_share_one_universal_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_cache_file = Path(tmp_dir) / "model_a_embeddings.npy"
            second_cache_file = Path(tmp_dir) / "model_b_dim2_embeddings.npy"
            node_order = ["alpha", "beta"]

            save_st_embedding_cache(
                first_cache_file,
                {
                    "alpha": np.array([1.0, 2.0], dtype="float32"),
                    "beta": np.array([3.0, 4.0], dtype="float32"),
                },
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=node_order,
            )
            save_st_embedding_cache(
                second_cache_file,
                {
                    "alpha": np.array([5.0, 6.0], dtype="float32"),
                    "beta": np.array([7.0, 8.0], dtype="float32"),
                },
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=node_order,
            )

            self.assertTrue(
                get_node_universe_path(tmp_dir, MERGED_NODE_UNIVERSE).exists()
            )
            self.assertEqual(
                list(Path(tmp_dir).glob("*_texts.jsonl")),
                [],
            )

    def test_save_reuses_existing_rows_in_fixed_universal_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            node_order = ["alpha", "beta", "gamma"]
            first_cache = {
                "alpha": np.array([1.0, 2.0], dtype="float32"),
                "beta": np.array([3.0, 4.0], dtype="float32"),
                "gamma": np.array([5.0, 6.0], dtype="float32"),
            }
            save_st_embedding_cache(
                cache_file,
                first_cache,
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=node_order,
            )

            _, existing_text_to_idx, existing_vectors = load_st_embedding_cache_index(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            try:
                save_st_embedding_cache(
                    cache_file,
                    {"gamma": np.array([7.0, 8.0], dtype="float32")},
                    existing_text_to_idx=existing_text_to_idx,
                    existing_vectors=existing_vectors,
                    node_universe=MERGED_NODE_UNIVERSE,
                )
            finally:
                _close_memmap(existing_vectors)

            _, text_to_idx, vectors = load_st_embedding_cache_index(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            try:
                self.assertEqual(
                    text_to_idx,
                    {"alpha": 0, "beta": 1, "gamma": 2},
                )
                self.assertEqual(vectors.shape, (3, 2))
                np.testing.assert_array_equal(
                    vectors,
                    np.array(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                            [7.0, 8.0],
                        ],
                        dtype="float32",
                    ),
                )
            finally:
                _close_memmap(vectors)

    def test_causenet_full_uses_separate_universal_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            normal_cache = Path(tmp_dir) / "toy_model_embeddings.npy"
            full_cache = Path(tmp_dir) / "toy_model_causenet_full_embeddings.npy"

            save_st_embedding_cache(
                normal_cache,
                {
                    "alpha": np.array([1.0], dtype="float32"),
                    "beta": np.array([2.0], dtype="float32"),
                },
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=["alpha", "beta"],
            )
            save_st_embedding_cache(
                full_cache,
                {
                    "full alpha": np.array([3.0], dtype="float32"),
                    "full beta": np.array([4.0], dtype="float32"),
                    "full gamma": np.array([5.0], dtype="float32"),
                },
                node_universe=CAUSENET_FULL_NODE_UNIVERSE,
                node_order=["full alpha", "full beta", "full gamma"],
            )

            _, normal_index, normal_vectors = load_st_embedding_cache_index(
                normal_cache,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            _, full_index, full_vectors = load_st_embedding_cache_index(
                full_cache,
                node_universe=CAUSENET_FULL_NODE_UNIVERSE,
            )
            try:
                self.assertEqual(normal_index, {"alpha": 0, "beta": 1})
                self.assertEqual(
                    full_index,
                    {"full alpha": 0, "full beta": 1, "full gamma": 2},
                )
                self.assertEqual(normal_vectors.shape, (2, 1))
                self.assertEqual(full_vectors.shape, (3, 1))
            finally:
                _close_memmap(normal_vectors)
                _close_memmap(full_vectors)

    def test_wrong_node_universe_combination_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            save_st_embedding_cache(
                cache_file,
                {
                    "alpha": np.array([1.0], dtype="float32"),
                    "beta": np.array([2.0], dtype="float32"),
                },
                node_universe=MERGED_NODE_UNIVERSE,
                node_order=["alpha", "beta"],
            )
            write_node_universe(
                get_node_universe_path(tmp_dir, CAUSENET_FULL_NODE_UNIVERSE),
                ["full alpha", "full beta", "full gamma"],
            )

            with self.assertRaisesRegex(
                EmbeddingCacheValidationError,
                "row-count mismatch",
            ):
                load_st_embedding_cache_index(
                    cache_file,
                    node_universe=CAUSENET_FULL_NODE_UNIVERSE,
                    strict=True,
                )

    def test_missing_vector_cache_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            write_node_universe(
                get_node_universe_path(tmp_dir, MERGED_NODE_UNIVERSE),
                ["alpha", "beta"],
            )

            status = get_embedding_cache_status(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )

            self.assertFalse(status["covered"])
            self.assertEqual(status["nodes_count"], 2)
            self.assertIsNone(status["vectors_shape"])
            self.assertIn("missing embedding vectors", status["reason"])

    def test_single_file_pickle_cache_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            np.save(
                cache_file,
                {"alpha": np.array([1.0, 2.0], dtype="float32")},
                allow_pickle=True,
            )

            self.assertEqual(
                load_st_embedding_cache(
                    cache_file,
                    node_universe=MERGED_NODE_UNIVERSE,
                ),
                {},
            )

            in_memory_cache, text_to_idx, vectors = load_st_embedding_cache_index(
                cache_file,
                node_universe=MERGED_NODE_UNIVERSE,
            )
            self.assertEqual(in_memory_cache, {})
            self.assertEqual(text_to_idx, {})
            self.assertIsNone(vectors)


if __name__ == "__main__":
    unittest.main()
