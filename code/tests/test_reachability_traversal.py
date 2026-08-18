import unittest
from unittest.mock import patch

import networkx as nx

from core.utils import traverse_graph
from evaluation.evaluation import run_evaluation_loop
from traverse_strategies.astar import astar_traverse
from traverse_strategies.bfs import bfs_traverse


class DummyEmbedder:
    def embed(self, node):
        return node

    def embed_many(self, nodes):
        return list(nodes)

    def get_distances(self, _source, targets, **_kwargs):
        return [1.0 for _ in targets]

    def has_embedding_index(self):
        return False


class DummyIndexedEmbedder(DummyEmbedder):
    def has_embedding_index(self):
        return True

    def embed_index(self, index):
        return index

    def embed_indices(self, indices):
        return list(indices)


class DummyIndexedGraph:
    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.indices = {node: index for index, node in enumerate(self.nodes)}
        self.adjacency = [[] for _ in self.nodes]
        for source, target in edges:
            self.adjacency[self.indices[source]].append(self.indices[target])

    def has_node(self, node):
        return node in self.indices

    def node_index(self, node):
        return self.indices[node]

    def node_text(self, index):
        return self.nodes[index]


class ReachabilityTraversalTests(unittest.TestCase):
    def setUp(self):
        self.embedder = DummyEmbedder()

    def test_bfs_path_mode_is_preserved_and_reachability_returns_on_discovery(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("s", "a"), ("s", "t")])

        path, path_visits = bfs_traverse(graph, "s", "t", None)
        reachable, reachability_visits = bfs_traverse(
            graph,
            "s",
            "t",
            None,
            {"reachability_only": True},
        )

        self.assertEqual(path, ["s", "t"])
        self.assertEqual(path_visits, 2)
        self.assertIs(reachable, True)
        self.assertEqual(reachability_visits, 1)

    def test_astar_path_mode_is_preserved_and_reachability_returns_on_discovery(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("s", "a"), ("s", "t")])

        path, path_visits = astar_traverse(graph, "s", "t", self.embedder)
        reachable, reachability_visits = astar_traverse(
            graph,
            "s",
            "t",
            self.embedder,
            {"reachability_only": True},
        )

        self.assertEqual(path, ["s", "t"])
        self.assertEqual(path_visits, 2)
        self.assertIs(reachable, True)
        self.assertEqual(reachability_visits, 1)

    def test_budget_check_precedes_target_discovery(self):
        graph = nx.DiGraph([("s", "m"), ("m", "t")])

        for strategy, embedder, cap_key in (
            (bfs_traverse, None, "bfs_max_visits"),
            (astar_traverse, self.embedder, "astar_max_visits"),
        ):
            with self.subTest(strategy=strategy.__name__):
                reachable, visits = strategy(
                    graph,
                    "s",
                    "t",
                    embedder,
                    {"reachability_only": True, cap_key: 1},
                )
                self.assertIs(reachable, False)
                self.assertEqual(visits, 2)

                reachable, visits = strategy(
                    graph,
                    "s",
                    "t",
                    embedder,
                    {"reachability_only": True, cap_key: 2},
                )
                self.assertIs(reachable, True)
                self.assertEqual(visits, 2)

    def test_same_node_and_missing_node_results_match_the_selected_mode(self):
        graph = nx.DiGraph()
        graph.add_node("s")

        for strategy, embedder in (
            (bfs_traverse, None),
            (astar_traverse, self.embedder),
        ):
            with self.subTest(strategy=strategy.__name__):
                self.assertEqual(strategy(graph, "s", "s", embedder), (["s"], 0))
                self.assertEqual(
                    strategy(
                        graph,
                        "s",
                        "s",
                        embedder,
                        {"reachability_only": True},
                    ),
                    (True, 0),
                )
                self.assertEqual(
                    traverse_graph(
                        graph,
                        "s",
                        "missing",
                        embedder,
                        strategy,
                        {"reachability_only": True},
                    ),
                    (False, 0),
                )

    def test_in_graph_unreachable_target_returns_false_on_exhaustion(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["s", "t"])

        for strategy, embedder in (
            (bfs_traverse, None),
            (astar_traverse, self.embedder),
        ):
            with self.subTest(strategy=strategy.__name__):
                self.assertEqual(
                    strategy(
                        graph,
                        "s",
                        "t",
                        embedder,
                        {"reachability_only": True},
                    ),
                    (False, 1),
                )

    def test_reachability_does_not_reconstruct_paths(self):
        graph = nx.DiGraph([("s", "t")])

        with patch(
            "traverse_strategies.bfs._reconstruct_path",
            side_effect=AssertionError("path reconstruction used"),
        ), patch(
            "traverse_strategies.astar._reconstruct_path",
            side_effect=AssertionError("path reconstruction used"),
        ):
            self.assertEqual(
                bfs_traverse(
                    graph,
                    "s",
                    "t",
                    None,
                    {"reachability_only": True},
                ),
                (True, 1),
            )
            self.assertEqual(
                astar_traverse(
                    graph,
                    "s",
                    "t",
                    self.embedder,
                    {"reachability_only": True},
                ),
                (True, 1),
            )

    def test_indexed_astar_supports_both_modes(self):
        graph = nx.DiGraph([("s", "t")])
        indexed_graph = DummyIndexedGraph(["s", "t"], [("s", "t")])
        embedder = DummyIndexedEmbedder()

        path, path_visits = astar_traverse(
            graph,
            "s",
            "t",
            embedder,
            {"_indexed_graph": indexed_graph},
        )
        reachable, reachability_visits = astar_traverse(
            graph,
            "s",
            "t",
            embedder,
            {"_indexed_graph": indexed_graph, "reachability_only": True},
        )

        self.assertEqual((path, path_visits), (["s", "t"], 1))
        self.assertEqual((reachable, reachability_visits), (True, 1))

        self.assertEqual(
            astar_traverse(
                graph,
                "s",
                "s",
                embedder,
                {"_indexed_graph": indexed_graph, "reachability_only": True},
            ),
            (True, 0),
        )

    def test_evaluation_uses_boolean_reachability_without_path_metrics(self):
        graph = nx.DiGraph([("s", "t")])

        def boolean_strategy(_graph, _start, _end, _embedder, config):
            self.assertTrue(config["reachability_only"])
            return True, 1

        summary = run_evaluation_loop(
            [{"id": "q1", "cause": "s", "effect": "t", "answer": True}],
            graph,
            None,
            {"BFS": boolean_strategy},
            "reachability-test",
        )["BFS"]

        self.assertEqual(summary["metrics"]["f1_score"], 1.0)
        self.assertEqual(summary["metrics"]["avg_path_length"], 0.0)
        self.assertEqual(summary["metrics"]["num_costed_paths"], 0)
        self.assertEqual(summary["per_example"][0]["path_length"], 0)
        self.assertIsNone(summary["per_example"][0]["path_cost"])


if __name__ == "__main__":
    unittest.main()
