import unittest

from reports.p95_visited_nodes import extract_p95_rows, render_latex


class P95VisitedNodesReportTests(unittest.TestCase):
    def test_extract_rows_uses_finetuned_astar_successful_p95_caps(self):
        entries = [
            {
                "model": "all-mpnet-base-v2",
                "dimension": 2,
                "analysis": {
                    "strategy": "A*",
                    "p95_visited_successful_only": 999.1,
                },
            },
            {
                "model": (
                    "all-mpnet-base-v2_relu_cosine_nonorm_"
                    "matryoshka_v3_finetuned"
                ),
                "dimension": 2,
                "analysis": {
                    "strategy": "A*",
                    "p95_visited_successful_only": 889.7,
                    "num_examples": 1213,
                    "num_successful_paths": 1127,
                },
            },
            {
                "model": (
                    "all-mpnet-base-v2_relu_cosine_nonorm_"
                    "matryoshka_v3_finetuned"
                ),
                "dimension": 4,
                "analysis": {
                    "strategy": "Dijkstra",
                    "p95_visited_successful_only": 100.0,
                },
            },
        ]

        rows = extract_p95_rows(entries, dimensions=(2, 4))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "MPNet")
        self.assertEqual(rows[0]["values"]["2"]["raw"], 889.7)
        self.assertEqual(rows[0]["values"]["2"]["cap"], 890)
        self.assertNotIn("4", rows[0]["values"])

    def test_latex_uses_compact_dimension_header_and_dash_for_missing_values(self):
        latex = render_latex(
            [
                {
                    "model": "Qwen",
                    "values": {
                        "2": {"cap": 1277},
                        "4": {"cap": None},
                    },
                }
            ],
            (2, 4),
            caption="Example",
            label="tab:example",
        )

        self.assertIn(r"\multicolumn{2}{c}{\textbf{Dimension}}", latex)
        self.assertIn(r"$1{,}277$ & --", latex)
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\bottomrule", latex)


if __name__ == "__main__":
    unittest.main()
