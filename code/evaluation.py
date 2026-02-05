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
from embeddings import STEmbeder, GloveEmbeder, DistanceMetric
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

base_models = ["all-mpnet-base-v2"]

lightning_dir = "data/models/lightning"
fine_tuned_models = []

if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name))
    ]

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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nodes_visited, path_lengths, times = np.array(nodes_visited), np.array(path_lengths), np.array(times)
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


def run_evaluation_loop(data, graph, embeder, strategies, description, config=None):
    results = {
        name: {"y_true": [], "y_pred": [], "nodes_visited": [], "path_lengths": [], "times": []}
        for name in strategies.keys()
    }

    print(f"Starting evaluation: {description}")

    for i, item in enumerate(data):
        if i % 100 == 0: print(f"  Eval {i}/{len(data)}...")
        cause = get_concept(item, 0)
        effect = get_concept(item, 1)
        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        true_label = item['answer:Extracted'][0] == 'Yes'

        for name, strategy in strategies.items():
            start_time = time.time()
            path, visited_nodes = traverse_graph(graph, cause, effect, embeder, strategy, config)
            end_time = time.time()

            elapsed = end_time - start_time
            pred_label = bool(path)

            results[name]["y_true"].append(true_label)
            results[name]["y_pred"].append(pred_label)
            results[name]["nodes_visited"].append(visited_nodes)
            results[name]["path_lengths"].append(len(path))
            results[name]["times"].append(elapsed)

    summary = {}
    for name in strategies.keys():
        metrics = calculate_metrics(
            results[name]["y_true"], results[name]["y_pred"],
            results[name]["nodes_visited"], results[name]["path_lengths"],
            results[name]["times"]
        )
        summary[name] = metrics
        print(f"--- {name} Results ---")
        print(
            f"Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f} | Avg Nodes: {metrics['avg_nodes_visited']:.1f}")
    return summary


if __name__ == "__main__":
    MASTER_CONFIG = {
        'rl_model_path': "data/models/rl/msmarco_evaluation_state_dict.pt",
        'rl_beam_width': 5,
        'rl_max_path_len': -1,
        'rl_max_visits': 445,
        'astar_max_visits': 399,
        'dijkstra_max_visits': 5987
    }

    print("Loading test data...")
    with open(VALID_DATA_PATH) as f:
        valid_data = json.load(f)

    print("Loading causal graph...")
    causal_graph = load_graph(GRAPH_PATH)
    existing_results = load_results_file()

    # BFS Baseline
    if not any(entry['model'] == "BFS_Baseline" for entry in existing_results):
        print("\n=== Running BFS Baseline ===")
        bfs_summary = run_evaluation_loop(valid_data, causal_graph, None, {"BFS": ts.bfs_traverse}, "BFS Baseline",
                                          config=MASTER_CONFIG)
        save_result(
            {"model": "BFS_Baseline", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "evaluation": bfs_summary})

    # RL Baseline
    if not any(entry['model'] == "RL_Baseline" for entry in existing_results):
        print("\n=== Running RL Baseline ===")
        try:
            rl_embeder = GloveEmbeder('data/embeddings/glove.6B/glove.6B.300d.txt', DistanceMetric.COSINE)
            rl_summary = run_evaluation_loop(valid_data, causal_graph, rl_embeder, {"RL": ts.rl_traverse},
                                             "RL Baseline", config=MASTER_CONFIG)
            save_result(
                {"model": "RL_Baseline", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "evaluation": rl_summary})
            del rl_embeder
            gc.collect()
        except Exception as e:
            print(f"Failed RL: {e}")

    # Semantic Strategies (A* and Dijkstra)
    semantic_strategies = {"A*": ts.astar_traverse, "Dijkstra": ts.dijkstra_traverse}

    for model_path in model_queue:
        model_name = model_path.split("/")[-1]
        if any(entry['model'] == model_name for entry in existing_results):
            continue

        print(f"\nEVALUATING: {model_path}")
        try:
            main_embeder = STEmbeder(model_path=model_path, distance_metric=DistanceMetric.COSINE)
            for dim in MATRYOSHKA_DIMS:
                print(f"--- Dim: {dim} ---")
                main_embeder.set_matryoshka_dim(dim)
                main_summary = run_evaluation_loop(valid_data, causal_graph, main_embeder, semantic_strategies,
                                                   model_path, config=MASTER_CONFIG)
                save_result({"model": model_name, "dimension": dim, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "evaluation": main_summary})

            del main_embeder
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error: {e}")

    print("\nAll evaluations complete.")
