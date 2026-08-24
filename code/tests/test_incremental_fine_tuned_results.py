import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluation import (
    save_fastest_equivalent_result,
    save_result as save_evaluation_result,
)
from evaluation.visited_nodes_analysis import save_result as save_analysis_result


class IncrementalFineTunedResultTests(unittest.TestCase):
    @staticmethod
    def make_astar_entry(
        runtime,
        *,
        prediction=True,
        visits=3,
        timestamp="old",
        max_visits=23,
    ):
        return {
            "model": "granite_v4_finetuned",
            "model_path": "data/models/lightning/granite_v4_finetuned",
            "dimension": 64,
            "split": "test",
            "run_suffix": "v4",
            "ablation": False,
            "config_source_dataset": "msmarco_train",
            "config_source_graph": "causenet",
            "ablation_shared_max_visits": False,
            "ablation_cap_reference_model": None,
            "embedding_device": "cuda",
            "used_config": {"astar_max_visits": max_visits},
            "timestamp": timestamp,
            "evaluation": {
                "A*": {
                    "metrics": {
                        "accuracy": 1.0 if prediction else 0.0,
                        "f1_score": 1.0 if prediction else 0.0,
                        "recall": 1.0 if prediction else 0.0,
                        "precision": 1.0 if prediction else 0.0,
                        "tp": 1 if prediction else 0,
                        "fn": 0 if prediction else 1,
                        "fp": 0,
                        "tn": 0,
                        "avg_nodes_visited": float(visits),
                        "avg_path_length": 0.0,
                        "avg_time_ms": runtime,
                        "avg_path_cost": 0.0,
                        "avg_cost_per_hop": 0.0,
                        "num_costed_paths": 0,
                        "num_examples": 1,
                    },
                    "per_example": [
                        {
                            "id": "q1",
                            "cause": "a",
                            "effect": "b",
                            "true": True,
                            "pred": prediction,
                            "correct": prediction,
                            "nodes_visited": visits,
                            "path_length": 0,
                            "time_sec": runtime / 1000.0,
                            "time_ms": runtime,
                            "path_cost": None,
                        }
                    ],
                }
            },
        }

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

    def test_fastest_equivalent_result_replaces_only_astar_in_place(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_json_file = Path(temporary_directory) / "evaluation_results.json"
            output_csv_file = Path(temporary_directory) / "evaluation_results.csv"
            existing = self.make_astar_entry(5.0)
            baseline = {
                "model": "BFS_Baseline",
                "dimension": None,
                "evaluation": {
                    "BFS": {"metrics": {"f1_score": 0.75}}
                },
            }
            output_json_file.write_text(
                json.dumps([existing, baseline], indent=4),
                encoding="utf-8",
            )

            candidate = self.make_astar_entry(3.0, timestamp="new")
            replaced = save_fastest_equivalent_result(
                candidate,
                output_json_file,
                output_csv_file,
            )

            saved = json.loads(output_json_file.read_text(encoding="utf-8"))
            self.assertTrue(replaced)
            self.assertEqual([entry["model"] for entry in saved], [
                "granite_v4_finetuned",
                "BFS_Baseline",
            ])
            self.assertEqual(
                saved[0]["evaluation"]["A*"]["metrics"]["avg_time_ms"],
                3.0,
            )
            self.assertEqual(saved[0]["timestamp"], "new")
            self.assertEqual(saved[1], baseline)

    def test_slower_equivalent_result_leaves_outputs_untouched(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_json_file = Path(temporary_directory) / "evaluation_results.json"
            output_csv_file = Path(temporary_directory) / "evaluation_results.csv"
            existing = self.make_astar_entry(3.0)
            save_evaluation_result(existing, output_json_file, output_csv_file)
            json_before = output_json_file.read_bytes()
            csv_before = output_csv_file.read_bytes()

            retained = save_fastest_equivalent_result(
                self.make_astar_entry(5.0, timestamp="new"),
                output_json_file,
                output_csv_file,
            )

            self.assertFalse(retained)
            self.assertEqual(output_json_file.read_bytes(), json_before)
            self.assertEqual(output_csv_file.read_bytes(), csv_before)

    def test_substantive_mismatch_raises_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_json_file = Path(temporary_directory) / "evaluation_results.json"
            output_csv_file = Path(temporary_directory) / "evaluation_results.csv"
            save_evaluation_result(
                self.make_astar_entry(5.0),
                output_json_file,
                output_csv_file,
            )
            json_before = output_json_file.read_bytes()
            csv_before = output_csv_file.read_bytes()

            with self.assertRaisesRegex(ValueError, "aggregate metrics changed"):
                save_fastest_equivalent_result(
                    self.make_astar_entry(3.0, prediction=False),
                    output_json_file,
                    output_csv_file,
                )

            self.assertEqual(output_json_file.read_bytes(), json_before)
            self.assertEqual(output_csv_file.read_bytes(), csv_before)

    def test_capped_and_uncapped_astar_results_coexist_and_retain_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_json_file = Path(temporary_directory) / "evaluation_results.json"
            output_csv_file = Path(temporary_directory) / "evaluation_results.csv"

            capped = self.make_astar_entry(3.0, max_visits=23)
            uncapped = self.make_astar_entry(9.0, max_visits=-1)
            save_evaluation_result(capped, output_json_file, output_csv_file)
            save_fastest_equivalent_result(
                uncapped,
                output_json_file,
                output_csv_file,
            )
            save_fastest_equivalent_result(
                self.make_astar_entry(5.0, max_visits=-1, timestamp="new"),
                output_json_file,
                output_csv_file,
            )

            saved = json.loads(output_json_file.read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 2)
            by_cap = {
                entry["used_config"]["astar_max_visits"]: entry
                for entry in saved
            }
            self.assertEqual(
                by_cap[23]["evaluation"]["A*"]["metrics"]["avg_time_ms"],
                3.0,
            )
            self.assertEqual(
                by_cap[-1]["evaluation"]["A*"]["metrics"]["avg_time_ms"],
                5.0,
            )
