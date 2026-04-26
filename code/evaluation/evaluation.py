# TODO Signifikanztest und Bonferroni Correction

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import traverse_strategies as ts
from core.embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from core.utils import get_matryoshka_dims, load_graph, traverse_graph

GRAPH_PATH = "data/graphs/causenet-precision.jsonl"

base_models = ["all-mpnet-base-v2", "Qwen/Qwen3-Embedding-0.6B"]

lightning_dir = "data/models/lightning"
fine_tuned_models = []

# Pick up all exported fine-tuned models automatically.
if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name)) if name != "old"
    ]

model_queue = base_models + fine_tuned_models

print("Model queue:", model_queue)


def build_output_paths(dataset_path: str):
    dataset_stem = Path(dataset_path).stem
    dataset_name = dataset_stem.replace("_filtered", "")
    output_dir = Path("data/evaluation") / dataset_name
    output_json_file = output_dir / "evaluation_results.json"
    output_csv_file = output_dir / "evaluation_results.csv"
    return dataset_name, output_dir, str(output_json_file), str(output_csv_file)


def compute_embedding_path_cost(path, embeder):
    # No path means no cost can be computed.
    if not path:
        return None

    # A single-node path has zero movement cost.
    if len(path) < 2:
        return 0.0

    if embeder is None:
        return None

    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ea = embeder.embed(a)
        eb = embeder.embed(b)
        total += embeder.get_distance(ea, eb)

    return float(total)


def save_all_results_csv(all_results, output_csv_file):
    fieldnames = [
        "algorithm",
        "model",
        "dimension",
        "timestamp",
        "accuracy",
        "f1_score",
        "recall",
        "precision",
        "tp",
        "fn",
        "fp",
        "tn",
        "avg_nodes_visited",
        "avg_path_length",
        "avg_time_sec",
        "avg_path_cost",
        "avg_cost_per_hop",
        "num_costed_paths",
    ]

    rows = []

    for entry in all_results:
        for algorithm, metrics in entry.get("evaluation", {}).items():
            row = {
                "algorithm": algorithm,
                "model": entry.get("model"),
                "dimension": entry.get("dimension", ""),
                "timestamp": entry.get("timestamp"),
                **metrics,
            }

            # convert floats for German Excel
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = str(v).replace(".", ",")

            rows.append(row)

    with open(output_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV overwritten: {output_csv_file}")


def load_results_file(output_json_file):
    if os.path.exists(output_json_file):
        with open(output_json_file, "r") as f:
            return json.load(f)
    return []


def save_result(result_entry, output_json_file, output_csv_file):
    current_results = load_results_file(output_json_file)
    current_results.append(result_entry)

    # Save JSON (overwrite full file)
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(current_results, f, indent=4)

    # Save CSV (overwrite full file)
    save_all_results_csv(current_results, output_csv_file)

    print(f"Saved results for '{result_entry['model']}'")


def calculate_metrics(y_true, y_pred, nodes_visited, path_lengths, times, path_costs):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    nodes_visited = np.array(nodes_visited, dtype=float)
    path_lengths = np.array(path_lengths, dtype=int)
    times = np.array(times, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    # Ignore empty paths when averaging path length.
    valid_lengths = path_lengths[path_lengths > 0]
    avg_path_len = float(valid_lengths.mean()) if len(valid_lengths) > 0 else 0.0

    # Only count costs that were actually computed.
    valid_costs = [c for c in path_costs if c is not None]
    avg_cost = float(np.mean(valid_costs)) if len(valid_costs) > 0 else 0.0

    # Cost per hop is more comparable than raw path cost when paths differ in length.
    cost_per_hop = []
    for L, c in zip(path_lengths, path_costs):
        if c is not None and L > 1:
            cost_per_hop.append(c / (L - 1))
    avg_cost_per_hop = float(np.mean(cost_per_hop)) if len(cost_per_hop) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "avg_nodes_visited": float(nodes_visited.mean()) if len(nodes_visited) > 0 else 0.0,
        "avg_path_length": float(avg_path_len),
        "avg_time_sec": float(times.mean()) if len(times) > 0 else 0.0,
        "avg_path_cost": float(avg_cost),
        "avg_cost_per_hop": float(avg_cost_per_hop),
        "num_costed_paths": int(len(valid_costs)),
    }


def run_evaluation_loop(data, graph, embeder, strategies, description, config=None):
    # Keep raw outputs for each strategy and summarize at the end.
    results = {
        name: {
            "y_true": [],
            "y_pred": [],
            "nodes_visited": [],
            "path_lengths": [],
            "times": [],
            "path_costs": [],
        }
        for name in strategies.keys()
    }

    print(f"Starting evaluation: {description}")

    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"  Eval {i}/{len(data)}...")

        cause = item["cause"]
        effect = item["effect"]

        # Skip examples that are not covered by the current graph.
        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        true_label = bool(item["answer"])

        for name, strategy in strategies.items():
            start_time = time.time()
            path, visited_nodes = traverse_graph(graph, cause, effect, embeder, strategy, config)
            end_time = time.time()

            elapsed = end_time - start_time
            pred_label = bool(path)

            path_cost = compute_embedding_path_cost(path, embeder)

            results[name]["y_true"].append(true_label)
            results[name]["y_pred"].append(pred_label)
            results[name]["nodes_visited"].append(visited_nodes)
            results[name]["path_lengths"].append(len(path) if path else 0)
            results[name]["times"].append(elapsed)
            results[name]["path_costs"].append(path_cost)

    summary = {}
    for name in strategies.keys():
        metrics = calculate_metrics(
            results[name]["y_true"],
            results[name]["y_pred"],
            results[name]["nodes_visited"],
            results[name]["path_lengths"],
            results[name]["times"],
            results[name]["path_costs"],
        )
        summary[name] = metrics

        print(f"--- {name} Results ---")
        print(
            f"Acc: {metrics['accuracy']:.3f} | "
            f"F1: {metrics['f1_score']:.3f} | "
            f"Avg Nodes: {metrics['avg_nodes_visited']:.1f} | "
            f"Avg Cost: {metrics['avg_path_cost']:.3f} | "
            f"Cost/Hop: {metrics['avg_cost_per_hop']:.3f}"
        )

    return summary


def parse_args():
    # Read the dataset path from the command line so the same script can be
    # reused across normalized datasets without editing constants in the file.
    parser = argparse.ArgumentParser(description="Evaluate a normalized causal QA dataset.")
    parser.add_argument("dataset_path", help="Path to the normalized dataset JSON file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path

    MASTER_CONFIG = {
        "rl_model_path": "data/models/rl/msmarco_evaluation_state_dict.pt",
        "rl_beam_width": 5,
        "rl_max_path_len": -1,
        "rl_max_visits": 446,
        "astar_max_visits": 399,
        "dijkstra_max_visits": 5987,
    }

    dataset_name, output_dir, output_json_file, output_csv_file = build_output_paths(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading normalized dataset...")
    with open(dataset_path) as f:
        valid_data = json.load(f)

    print("Loading causal graph...")
    causal_graph = load_graph(GRAPH_PATH)
    existing_results = load_results_file(output_json_file)

    # -------------------------------------------------------------------------
    # BFS baseline
    # -------------------------------------------------------------------------
    if not any(entry["model"] == "BFS_Baseline" for entry in existing_results):
        print("\n=== Running BFS Baseline ===")

        bfs_summary = run_evaluation_loop(
            valid_data,
            causal_graph,
            None,
            {"BFS": ts.bfs_traverse},
            f"BFS Baseline | {dataset_name}",
            config=MASTER_CONFIG,
        )

        save_result(
            {
                "model": "BFS_Baseline",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": bfs_summary,
            },
            output_json_file,
            output_csv_file,
        )

    # -------------------------------------------------------------------------
    # RL baseline
    # -------------------------------------------------------------------------
    if not any(entry["model"] == "RL_Baseline" for entry in existing_results):
        print("\n=== Running RL Baseline ===")
        try:
            rl_embeder = GloveEmbeder(
                "data/embeddings/glove.6B/glove.6B.300d.txt",
                DistanceMetric.COSINE,
            )

            rl_summary = run_evaluation_loop(
                valid_data,
                causal_graph,
                rl_embeder,
                {"RL": ts.rl_traverse},
                f"RL Baseline | {dataset_name}",
                config=MASTER_CONFIG,
            )

            save_result(
                {
                    "model": "RL_Baseline",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "evaluation": rl_summary,
                },
                output_json_file,
                output_csv_file,
            )

            del rl_embeder
            gc.collect()
        except Exception as e:
            print(f"Failed RL: {e}")

    # -------------------------------------------------------------------------
    # Semantic search strategies
    # -------------------------------------------------------------------------
    semantic_strategies = {
        "A*": ts.astar_traverse,
        "Dijkstra": ts.dijkstra_traverse,
    }

    for model_path in model_queue:
        model_name = model_path.split("/")[-1]

        print(f"\nEVALUATING: {model_path}")

        try:
            main_embeder = STEmbedder(
                model_path=model_path,
                distance_metric=DistanceMetric.COSINE
            )

            model_dim = main_embeder.get_model_dim()

            dims = get_matryoshka_dims(model_dim)

            for dim in dims:
                if any(entry.get("model") == model_name and
                    (entry.get("dimension") == dim)
                    for entry in existing_results):
                    print(f"Skipping {model_name} dim {dim} (already evaluated)")
                    continue

                print(f"--- Dim: {dim} ---")
                main_embeder.set_matryoshka_dim(dim)

                main_summary = run_evaluation_loop(
                    valid_data,
                    causal_graph,
                    main_embeder,
                    semantic_strategies,
                    f"{model_path} | {dataset_name}",
                    config=MASTER_CONFIG,
                )

                save_result(
                    {
                        "model": model_name,
                        "dimension": dim,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "evaluation": main_summary,
                    },
                    output_json_file,
                    output_csv_file,
                )

            del main_embeder
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error: {e}")

    print("\nAll evaluations complete.")
