import gc
import json
import os
import time

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
from embeddings import STEmbeder, DistanceMetric
from utils import get_concept, load_graph, traverse_graph


GRAPH_PATH = "data/graphs/causenet-precision.jsonl"
TEST = False
if TEST:
    OUTPUT_FILE = "data/evaluation/evaluation_results_test.json"
    VALID_DATA_PATH = "data/datasets/msmarco_test.json"
else:
    OUTPUT_FILE = "data/evaluation/evaluation_results_valid.json"
    VALID_DATA_PATH = "data/datasets/msmarco_valid.json"

MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]

base_models = [
    "all-mpnet-base-v2",
    # "all-MiniLM-L12-v2",
    # "multi-qa-mpnet-base-cos-v1"
]

lightning_dir = "data/models/lightning"
fine_tuned_models = []

if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name))
    ]
    print(f"Found {len(fine_tuned_models)} fine-tuned models in {lightning_dir}")
else:
    print(f"Warning: Directory {lightning_dir} not found.")

model_queue = base_models + fine_tuned_models


def load_results_file():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    return []


def save_result(result_entry):
    current_results = load_results_file()
    current_results.append(result_entry)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(current_results, f, indent=4)
    print(f"Saved results for '{result_entry['model']}'")


def calculate_metrics(y_true, y_pred, nodes_visited, path_lengths, times):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    nodes_visited = np.array(nodes_visited, dtype=int)
    path_lengths = np.array(path_lengths, dtype=int)
    times = np.array(times, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    valid_paths = path_lengths[path_lengths > 0]
    avg_path_len = valid_paths.mean() if len(valid_paths) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "avg_nodes_visited": float(nodes_visited.mean()),
        "avg_path_length": float(avg_path_len),
        "avg_time_sec": float(times.mean())
    }


def run_evaluation_loop(data, graph, embeder, strategies, description):
    results = {
        name: {"y_true": [], "y_pred": [], "nodes_visited": [], "path_lengths": [], "times": []}
        for name in strategies.keys()
    }

    print(f"Starting evaluation: {description}")

    for i, item in enumerate(data):
        if i % 100 == 0: print(f"  Eval {i}/{len(data)}...")

        cause = get_concept(item, 0)
        effect = get_concept(item, 1)
        true_label = item['answer:Extracted'][0] == 'Yes'

        for name, strategy in strategies.items():
            start_time = time.time()
            path, visited_nodes = traverse_graph(graph, cause, effect, embeder, strategy)
            end_time = time.time()

            elapsed = end_time - start_time
            pred_label = bool(path)  # True if path found, False otherwise

            results[name]["y_true"].append(true_label)
            results[name]["y_pred"].append(pred_label)
            results[name]["nodes_visited"].append(visited_nodes)
            results[name]["path_lengths"].append(len(path))
            results[name]["times"].append(elapsed)

    summary = {}
    for name in strategies.keys():
        metrics = calculate_metrics(
            results[name]["y_true"],
            results[name]["y_pred"],
            results[name]["nodes_visited"],
            results[name]["path_lengths"],
            results[name]["times"]
        )
        summary[name] = metrics

        print(f"--- {name} Results ---")
        print(
            f"Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f} | Avg Nodes: {metrics['avg_nodes_visited']:.1f}")

    return summary



if __name__ == "__main__":
    print("Loading test data...")
    with open(VALID_DATA_PATH) as f:
        valid_data = json.load(f)

    print("Loading causal graph...")
    causal_graph = load_graph(GRAPH_PATH)

    existing_results = load_results_file()
    bfs_done = any(entry['model'] == "BFS_Baseline" for entry in existing_results)

    if not bfs_done:
        print("\n=== Running BFS Baseline (One-off) ===")
        bfs_strategies = {"BFS": ts.bfs_traverse}
        bfs_summary = run_evaluation_loop(valid_data, causal_graph, None, bfs_strategies, "BFS Baseline")

        save_result({
            "model": "BFS_Baseline",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation": bfs_summary
        })
    else:
        print("\n=== BFS Baseline already exists. Skipping. ===")

    semantic_strategies = {
        "A*": ts.astar_traverse,
        "Dijkstra": ts.dijkstra_traverse
    }

    for model_path in model_queue:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'=' * 60}")

        existing_results = load_results_file()
        model_name = model_path.split("/")[-1]

        if any(entry['model'] == model_name for entry in existing_results):
            print(f"Skipping {model_name} (already in results).")
            continue

        try:
            main_embeder = STEmbeder(model_path=model_path, distance_metric=DistanceMetric.COSINE)

            for dim in MATRYOSHKA_DIMS:
                print(f"\n--- Evaluating Dim: {dim} ---")

                main_embeder.set_matryoshka_dim(dim)

                main_summary = run_evaluation_loop(valid_data, causal_graph, main_embeder, semantic_strategies, model_path)

                save_result({
                    "model": model_name,
                    "dimension": dim,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "evaluation": main_summary
                })

            print(f"Cleaning up memory for {model_path}...")
            del main_embeder
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            import traceback

            traceback.print_exc()

    print("\nAll evaluations complete.")