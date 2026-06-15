import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.embeddings import (
    _close_memmap,
    load_st_embedding_cache,
    load_st_embedding_cache_index,
    save_st_embedding_cache,
)


def split_cache_paths(cache_file):
    cache_stem = cache_file.with_suffix("")
    return {
        "texts": cache_stem.with_name(f"{cache_stem.name}_texts.jsonl"),
        "vectors": cache_stem.with_name(f"{cache_stem.name}_vectors.npy"),
    }


class EmbeddingCacheJsonlMmapTests(unittest.TestCase):
    def test_save_and_load_split_jsonl_mmap_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            cache = {
                "alpha": np.array([1.0, 2.0, 3.0], dtype="float32"),
                "beta": np.array([4.0, 5.0, 6.0], dtype="float32"),
            }

            save_st_embedding_cache(cache_file, cache)
            paths = split_cache_paths(cache_file)

            self.assertFalse(cache_file.exists())
            self.assertTrue(paths["texts"].exists())
            self.assertTrue(paths["vectors"].exists())

            with open(paths["texts"], encoding="utf-8") as file:
                self.assertEqual(
                    [json.loads(line) for line in file],
                    ["alpha", "beta"],
                )

            loaded = load_st_embedding_cache(cache_file)
            self.assertEqual(set(loaded), {"alpha", "beta"})
            np.testing.assert_array_equal(loaded["alpha"], cache["alpha"])
            np.testing.assert_array_equal(loaded["beta"], cache["beta"])

            in_memory_cache, text_to_idx, vectors = load_st_embedding_cache_index(
                cache_file
            )
            try:
                self.assertEqual(in_memory_cache, {})
                self.assertEqual(text_to_idx, {"alpha": 0, "beta": 1})
                self.assertIsInstance(vectors, np.memmap)
                self.assertEqual(vectors.shape, (2, 3))
                np.testing.assert_array_equal(vectors[1], cache["beta"])
            finally:
                _close_memmap(vectors)

    def test_save_appends_new_rows_to_existing_mmap_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            first_cache = {
                "alpha": np.array([1.0, 2.0], dtype="float32"),
                "beta": np.array([3.0, 4.0], dtype="float32"),
            }
            save_st_embedding_cache(cache_file, first_cache)

            _, existing_text_to_idx, existing_vectors = load_st_embedding_cache_index(
                cache_file
            )
            try:
                save_st_embedding_cache(
                    cache_file,
                    {"gamma": np.array([5.0, 6.0], dtype="float32")},
                    existing_text_to_idx=existing_text_to_idx,
                    existing_vectors=existing_vectors,
                )
            finally:
                _close_memmap(existing_vectors)

            _, text_to_idx, vectors = load_st_embedding_cache_index(cache_file)
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
                            [5.0, 6.0],
                        ],
                        dtype="float32",
                    ),
                )
            finally:
                _close_memmap(vectors)

    def test_single_file_pickle_cache_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "toy_model_embeddings.npy"
            np.save(
                cache_file,
                {"alpha": np.array([1.0, 2.0], dtype="float32")},
                allow_pickle=True,
            )

            self.assertEqual(load_st_embedding_cache(cache_file), {})

            in_memory_cache, text_to_idx, vectors = load_st_embedding_cache_index(
                cache_file
            )
            self.assertEqual(in_memory_cache, {})
            self.assertEqual(text_to_idx, {})
            self.assertIsNone(vectors)


if __name__ == "__main__":
    unittest.main()
