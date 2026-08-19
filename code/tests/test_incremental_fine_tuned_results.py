import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluation import save_result as save_evaluation_result
from evaluation.visited_nodes_analysis import save_result as save_analysis_result


class IncrementalFineTunedResultTests(unittest.TestCase):
    def test_visited_node_results_skip_an_existing_model_dimension_strategy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "visited_nodes_analysis.json"
            result = {
                "model": "mpnet_v4_finetuned",
                "dimension": 128,
                "analysis": {"strategy": "A*"},
            }

            save_analysis_result(result, output_file)
            save_analysis_result(result, output_file)

            self.assertEqual(
                json.loads(output_file.read_text(encoding="utf-8")),
                [result],
            )

    def test_evaluation_results_append_only_missing_algorithms(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_json_file = Path(temporary_directory) / "evaluation_results.json"
            output_csv_file = Path(temporary_directory) / "evaluation_results.csv"
            astar_result = {
                "model": "mpnet_v4_finetuned",
                "dimension": 128,
                "evaluation": {"A*": {"metrics": {"f1_score": 0.8}}},
            }
            dijkstra_result = {
                "model": "mpnet_v4_finetuned",
                "dimension": 128,
                "evaluation": {"Dijkstra": {"metrics": {"f1_score": 0.8}}},
            }

            save_evaluation_result(astar_result, output_json_file, output_csv_file)
            save_evaluation_result(astar_result, output_json_file, output_csv_file)
            save_evaluation_result(dijkstra_result, output_json_file, output_csv_file)

            saved_results = json.loads(
                output_json_file.read_text(encoding="utf-8")
            )
            saved_algorithms = [
                algorithm
                for entry in saved_results
                for algorithm in entry["evaluation"]
            ]
            self.assertEqual(saved_algorithms, ["A*", "Dijkstra"])

            with output_csv_file.open(newline="", encoding="utf-8") as file:
                self.assertEqual(
                    [
                        row["algorithm"]
                        for row in csv.DictReader(file, delimiter=";")
                    ],
                    ["A*", "Dijkstra"],
                )
