import unittest

import networkx as nx
import torch

from core.constants import DistanceMetric
from core.embeddings import STEmbedder
from core.indexed_graph import IndexedGraph
from traverse_strategies.astar import _get_distances_pair, astar_traverse


class STEmbedderPairedDistanceTests(unittest.TestCase):
    def make_embedder(self, distance_metric, device="cpu"):
        embedder = object.__new__(STEmbedder)
        embedder.device = torch.device(device)
        embedder.distance_metric = distance_metric
        embedder.matryoshka_dim = None
        return embedder

    def assert_pair_matches_independent_calls(
        self,
        distance_metric,
        source1,
        source2,
        matrix,
        assume_normalized=False,
    ):
        embedder = self.make_embedder(distance_metric)
        expected1 = embedder.get_distances(
            source1,
            matrix,
            assume_normalized=assume_normalized,
        )
        expected2 = embedder.get_distances(
            source2,
            matrix,
            assume_normalized=assume_normalized,
        )

        actual1, actual2 = embedder.get_distances_pair(
            source1,
            source2,
            matrix,
            assume_normalized=assume_normalized,
        )

        self.assertEqual(actual1, expected1)
        self.assertEqual(actual2, expected2)

    def test_euclidean_pair_matches_independent_calls_exactly(self):
        self.assert_pair_matches_independent_calls(
            DistanceMetric.EUCLIDEAN,
            torch.tensor([0.25, -0.5, 1.0, 2.0]),
            torch.tensor([-1.5, 0.75, 0.5, -0.25]),
            torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 2.0, 0.5],
                [-2.0, 0.25, 0.75, 3.0],
            ]),
        )

    def test_cosine_pair_matches_independent_calls_exactly(self):
        self.assert_pair_matches_independent_calls(
            DistanceMetric.COSINE,
            torch.tensor([0.25, -0.5, 1.0, 2.0]),
            torch.tensor([-1.5, 0.75, 0.5, -0.25]),
            torch.tensor([
                [0.5, 1.0, -0.5, 0.25],
                [1.0, -1.0, 2.0, 0.5],
                [-2.0, 0.25, 0.75, 3.0],
            ]),
        )

    def test_normalized_cosine_pair_matches_independent_calls_exactly(self):
        source1 = torch.nn.functional.normalize(
            torch.tensor([0.25, -0.5, 1.0, 2.0]),
            p=2,
            dim=0,
        )
        source2 = torch.nn.functional.normalize(
            torch.tensor([-1.5, 0.75, 0.5, -0.25]),
            p=2,
            dim=0,
        )
        matrix = torch.nn.functional.normalize(
            torch.tensor([
                [0.5, 1.0, -0.5, 0.25],
                [1.0, -1.0, 2.0, 0.5],
                [-2.0, 0.25, 0.75, 3.0],
            ]),
            p=2,
            dim=1,
        )

        self.assert_pair_matches_independent_calls(
            DistanceMetric.COSINE,
            source1,
            source2,
            matrix,
            assume_normalized=True,
        )

    def test_empty_pair_matches_two_empty_results(self):
        embedder = self.make_embedder(DistanceMetric.EUCLIDEAN)

        self.assertEqual(
            embedder.get_distances_pair(
                torch.ones(4),
                torch.zeros(4),
                torch.empty((0, 4)),
            ),
            ([], []),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_euclidean_pair_matches_independent_calls_exactly(self):
        embedder = self.make_embedder(DistanceMetric.EUCLIDEAN, device="cuda")
        source1 = torch.tensor([0.25, -0.5, 1.0, 2.0], device="cuda")
        source2 = torch.tensor([-1.5, 0.75, 0.5, -0.25], device="cuda")
        matrix = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 2.0, 0.5],
            [-2.0, 0.25, 0.75, 3.0],
        ], device="cuda")

        expected = (
            embedder.get_distances(source1, matrix),
            embedder.get_distances(source2, matrix),
        )
        actual = embedder.get_distances_pair(source1, source2, matrix)

        self.assertEqual(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_single_target_pair_matches_independent_calls_exactly(self):
        embedder = self.make_embedder(DistanceMetric.EUCLIDEAN, device="cuda")
        source1 = torch.tensor([0.25, -0.5, 1.0, 2.0], device="cuda")
        source2 = torch.tensor([-1.5, 0.75, 0.5, -0.25], device="cuda")
        matrix = torch.tensor(
            [[1.0, -1.0, 2.0, 0.5]],
            device="cuda",
        )

        expected = (
            embedder.get_distances(source1, matrix),
            embedder.get_distances(source2, matrix),
        )
        actual = embedder.get_distances_pair(source1, source2, matrix)

        self.assertEqual(actual, expected)


class STEmbedderIndexedGatherTests(unittest.TestCase):
    def make_embedder(self, rows=32, dim=8, device="cpu"):
        embedder = object.__new__(STEmbedder)
        embedder.device = device
        embedder.distance_metric = DistanceMetric.EUCLIDEAN
        embedder.matryoshka_dim = None
        matrix = torch.arange(
            rows * dim,
            dtype=torch.float32,
            device=device,
        ).reshape(rows, dim)
        embedder.embedding_table = torch.nn.Embedding.from_pretrained(
            matrix,
            freeze=True,
        )
        embedder.embedding_table_dim = dim
        return embedder

    def assert_gather_is_exact(self, indices, device="cpu"):
        embedder = self.make_embedder(device=device)
        expected_indices = torch.as_tensor(
            indices,
            device=device,
            dtype=torch.long,
        )
        expected = embedder.embedding_table.weight.index_select(
            0,
            expected_indices,
        )
        actual = embedder.embed_indices(indices)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_empty_single_small_and_batched_gathers_are_exact(self):
        for indices in ([], [7], [7, 2, 7, 15], list(range(16))):
            with self.subTest(indices=indices):
                self.assert_gather_is_exact(indices)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_small_gathers_are_bitwise_exact(self):
        for indices in ([7], [7, 2], [7, 2, 7, 15], list(range(15))):
            with self.subTest(indices=indices):
                self.assert_gather_is_exact(indices, device="cuda")

    def test_adaptive_gather_preserves_indexed_astar_path_and_reachability(self):
        nodes = ["start", "alpha", "beta", "merge", "goal"]
        node_to_idx = {node: index for index, node in enumerate(nodes)}
        adjacency = (
            (1, 2),
            (3,),
            (3,),
            (4,),
            (),
        )
        indexed_graph = IndexedGraph(
            node_to_idx=node_to_idx,
            idx_to_node=nodes,
            adjacency=adjacency,
        )
        graph = networkx_graph = nx.DiGraph()
        networkx_graph.add_nodes_from(nodes)
        for source_index, successors in enumerate(adjacency):
            for successor_index in successors:
                networkx_graph.add_edge(
                    nodes[source_index],
                    nodes[successor_index],
                )

        embedder = self.make_embedder(rows=len(nodes), dim=4)
        adaptive_results = []
        for reachability_only in (False, True):
            adaptive_results.append(
                astar_traverse(
                    graph,
                    "start",
                    "goal",
                    embedder,
                    {
                        "_indexed_graph": indexed_graph,
                        "reachability_only": reachability_only,
                    },
                )
            )

        def legacy_index_select(indices):
            index_tensor = torch.as_tensor(indices, dtype=torch.long)
            return embedder.embedding_table.weight.index_select(0, index_tensor)

        embedder.embed_indices = legacy_index_select
        legacy_results = []
        for reachability_only in (False, True):
            legacy_results.append(
                astar_traverse(
                    graph,
                    "start",
                    "goal",
                    embedder,
                    {
                        "_indexed_graph": indexed_graph,
                        "reachability_only": reachability_only,
                    },
                )
            )

        self.assertEqual(adaptive_results, legacy_results)


class LegacyDistanceEmbedder:
    def __init__(self):
        self.embeddings = {
            "start": 0.0,
            "left": 1.0,
            "right": 2.0,
            "goal": 3.0,
        }
        self.distance_calls = 0

    def has_normalized_runtime_embeddings(self):
        return False

    def embed(self, node):
        return self.embeddings[node]

    def get_distances(self, source, targets, assume_normalized=False):
        self.distance_calls += 1
        return [abs(source - target) for target in targets]


class PairedDistanceEmbedder(LegacyDistanceEmbedder):
    def __init__(self):
        super().__init__()
        self.pair_calls = 0

    def get_distances_pair(
        self,
        source1,
        source2,
        targets,
        assume_normalized=False,
    ):
        self.pair_calls += 1
        return (
            [abs(source1 - target) for target in targets],
            [abs(source2 - target) for target in targets],
        )


class AStarPairedDistanceTests(unittest.TestCase):
    def test_helper_falls_back_to_two_legacy_calls(self):
        embedder = LegacyDistanceEmbedder()

        result = _get_distances_pair(
            embedder,
            0.0,
            3.0,
            [1.0, 2.0],
            assume_normalized=False,
        )

        self.assertEqual(result, ([1.0, 2.0], [2.0, 1.0]))
        self.assertEqual(embedder.distance_calls, 2)

    def test_astar_path_and_visit_count_match_legacy_path(self):
        graph = nx.DiGraph([
            ("start", "left"),
            ("start", "right"),
            ("left", "goal"),
            ("right", "goal"),
        ])
        legacy_embedder = LegacyDistanceEmbedder()
        paired_embedder = PairedDistanceEmbedder()

        legacy_result = astar_traverse(
            graph,
            "start",
            "goal",
            legacy_embedder,
        )
        paired_result = astar_traverse(
            graph,
            "start",
            "goal",
            paired_embedder,
        )

        self.assertEqual(paired_result, legacy_result)
        self.assertGreater(paired_embedder.pair_calls, 0)
        self.assertEqual(paired_embedder.distance_calls, 0)
        self.assertEqual(
            legacy_embedder.distance_calls,
            paired_embedder.pair_calls * 2,
        )


if __name__ == "__main__":
    unittest.main()
