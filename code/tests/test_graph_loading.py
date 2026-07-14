import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from core.indexed_graph import build_indexed_graph
from core.utils import is_ignorable_graph_node, load_causal_graph


class GraphLoadingTests(unittest.TestCase):
    def test_causenet_loader_skips_blank_and_punctuation_only_endpoints(self):
        rows = [
            {
                "causal_relation": {
                    "cause": {"concept": "valid_cause"},
                    "effect": {"concept": "valid_effect"},
                },
                "sources": [],
            },
            {
                "causal_relation": {
                    "cause": {"concept": "blank_effect"},
                    "effect": {"concept": "   "},
                },
                "sources": [],
            },
            {
                "causal_relation": {
                    "cause": {"concept": ""},
                    "effect": {"concept": "blank_cause"},
                },
                "sources": [],
            },
            {
                "causal_relation": {
                    "cause": {"concept": "# ##"},
                    "effect": {"concept": "punctuation_only"},
                },
                "sources": [],
            },
            {
                "causal_relation": {
                    "cause": {"concept": "C#"},
                    "effect": {"concept": "valid_effect"},
                },
                "sources": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "graph.jsonl"
            with open(graph_path, "w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row))
                    file.write("\n")

            graph = load_causal_graph(graph_path)

        self.assertEqual(set(graph.nodes), {"valid cause", "valid effect", "C#"})
        self.assertNotIn("", graph.nodes)
        self.assertNotIn("# ##", graph.nodes)
        self.assertTrue(is_ignorable_graph_node("# ##"))
        self.assertFalse(is_ignorable_graph_node("C#"))

    def test_indexed_graph_ignores_invalid_missing_nodes(self):
        graph = nx.DiGraph()
        graph.add_edge("cause", "effect")
        graph.add_edge("cause", "# ##")
        graph.add_edge("# ##", "effect")

        indexed_graph = build_indexed_graph(
            graph,
            {"cause": 0, "effect": 1},
            progress_every=None,
        )

        self.assertEqual(indexed_graph.successors(0), (1,))
        self.assertEqual(indexed_graph.successors(1), ())


if __name__ == "__main__":
    unittest.main()
