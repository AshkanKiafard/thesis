import json
import tempfile
import unittest
from pathlib import Path

from core.utils import load_causal_graph


class GraphLoadingTests(unittest.TestCase):
    def test_causenet_loader_skips_blank_endpoints(self):
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
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "graph.jsonl"
            with open(graph_path, "w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row))
                    file.write("\n")

            graph = load_causal_graph(graph_path)

        self.assertEqual(set(graph.nodes), {"valid cause", "valid effect"})
        self.assertNotIn("", graph.nodes)


if __name__ == "__main__":
    unittest.main()
