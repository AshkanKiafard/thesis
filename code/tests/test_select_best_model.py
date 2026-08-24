import json
import tempfile
import unittest
from pathlib import Path

from evaluation.select_best_model import (
    build_pareto_front,
    rank_pareto_knee,
    select_best_astar_model,
)


def candidate(name, f1, nodes, runtime, dimension):
    return {
        "model": name,
        "model_path": f"data/models/lightning/{name}_v4_finetuned",
        "family": name,
        "variant": "finetuned",
        "dimension": dimension,
        "f1_score": f1,
        "accuracy": f1,
        "recall": f1,
        "precision": f1,
        "avg_nodes_visited": nodes,
        "avg_time_ms": runtime,
        "num_examples": 159,
        "astar_max_visits": max(1, round(nodes * 2)),
    }


class ParetoKneeSelectionTest(unittest.TestCase):
    def test_default_selection_excludes_uncapped_candidates(self):
        def entry(name, cap, f1, nodes):
            return {
                "model": name,
                "model_path": f"data/models/lightning/{name}",
                "dimension": 64,
                "used_config": {"astar_max_visits": cap},
                "evaluation": {
                    "A*": {
                        "metrics": {
                            "f1_score": f1,
                            "accuracy": f1,
                            "recall": f1,
                            "precision": f1,
                            "avg_nodes_visited": nodes,
                            "avg_time_ms": 1.0,
                            "num_examples": 10,
                        }
                    }
                },
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "evaluation_results.json"
            results_path.write_text(
                json.dumps([
                    entry("capped_finetuned", 23, 0.80, 5.0),
                    entry("uncapped_finetuned", -1, 0.99, 1.0),
                ]),
                encoding="utf-8",
            )

            selection = select_best_astar_model(results_path)

        self.assertEqual(selection["best"]["model"], "capped_finetuned")
        self.assertEqual(selection["budget_mode_filter"], "capped")

    def test_discards_dominated_candidates_and_selects_tradeoff_knee(self):
        efficient = candidate("efficient", 0.70, 1.0, 1.0, 2)
        knee = candidate("knee", 0.85, 4.0, 2.0, 4)
        expensive = candidate("expensive", 0.86, 100.0, 3.0, 8)
        dominated = candidate("dominated", 0.80, 10.0, 1.5, 16)

        frontier = build_pareto_front(
            [efficient, knee, expensive, dominated]
        )
        ranked = rank_pareto_knee(
            [efficient, knee, expensive, dominated]
        )

        self.assertEqual(
            [item["model"] for item in frontier],
            ["efficient", "knee", "expensive"],
        )
        self.assertEqual(ranked[0]["model"], "knee")
        self.assertGreater(ranked[0]["knee_score"], 0)

    def test_two_point_frontier_uses_more_effective_fallback(self):
        efficient = candidate("efficient", 0.80, 2.0, 0.5, 2)
        effective = candidate("effective", 0.85, 20.0, 5.0, 4)

        ranked = rank_pareto_knee([efficient, effective])

        self.assertEqual(ranked[0]["model"], "effective")

    def test_frontier_without_positive_interior_knee_uses_effective_endpoint(self):
        efficient = candidate("efficient", 0.20, 1.0, 0.5, 2)
        weak_tradeoff = candidate("weak_tradeoff", 0.30, 10.0, 1.0, 4)
        effective = candidate("effective", 0.90, 100.0, 5.0, 8)

        ranked = rank_pareto_knee(
            [efficient, weak_tradeoff, effective]
        )

        self.assertEqual(ranked[0]["model"], "effective")

    def test_collinear_frontier_uses_effective_endpoint_despite_roundoff(self):
        efficient = candidate("efficient", 0.20, 1.0, 0.5, 2)
        collinear = candidate("collinear", 0.55, 10.0, 1.0, 4)
        effective = candidate("effective", 0.90, 100.0, 5.0, 8)

        ranked = rank_pareto_knee([efficient, collinear, effective])

        self.assertEqual(ranked[0]["model"], "effective")

    def test_requires_positive_finite_visited_node_measurement(self):
        invalid = candidate("invalid", 0.80, 0.0, 1.0, 2)

        with self.assertRaisesRegex(ValueError, "positive finite"):
            rank_pareto_knee([invalid])


if __name__ == "__main__":
    unittest.main()
