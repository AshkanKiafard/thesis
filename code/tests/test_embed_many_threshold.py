import unittest

from core.config import EMBEDDING_INDEX_MIN_SUCCESSORS
from traverse_strategies.astar import _embed_many as astar_embed_many
from traverse_strategies.dijkstra import _embed_many as dijkstra_embed_many


class CountingEmbedder:
    def __init__(self, has_index=True):
        self.has_index = has_index
        self.embed_calls = 0
        self.embed_many_calls = 0

    def has_embedding_index(self):
        return self.has_index

    def embed(self, text):
        self.embed_calls += 1
        return f"single:{text}"

    def embed_many(self, texts):
        self.embed_many_calls += 1
        return [f"many:{text}" for text in texts]


class EmbedManyThresholdTests(unittest.TestCase):
    def assert_threshold_behavior(self, helper):
        below_threshold_nodes = [f"n{index}" for index in range(15)]
        at_threshold_nodes = [f"n{index}" for index in range(16)]
        config = {
            "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS
        }

        embedder = CountingEmbedder()
        result = helper(embedder, below_threshold_nodes, config)
        self.assertEqual(result[0], "single:n0")
        self.assertEqual(embedder.embed_calls, 15)
        self.assertEqual(embedder.embed_many_calls, 0)

        embedder = CountingEmbedder()
        result = helper(embedder, at_threshold_nodes, config)
        self.assertEqual(result[0], "many:n0")
        self.assertEqual(embedder.embed_calls, 0)
        self.assertEqual(embedder.embed_many_calls, 1)

        embedder = CountingEmbedder(has_index=False)
        result = helper(embedder, at_threshold_nodes, config)
        self.assertEqual(result[0], "single:n0")
        self.assertEqual(embedder.embed_calls, 16)
        self.assertEqual(embedder.embed_many_calls, 0)

    def test_astar_uses_embed_many_at_selected_threshold(self):
        self.assert_threshold_behavior(astar_embed_many)

    def test_dijkstra_uses_embed_many_at_selected_threshold(self):
        self.assert_threshold_behavior(dijkstra_embed_many)


if __name__ == "__main__":
    unittest.main()
