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
from core.utils import get_model_distance_metric, get_matryoshka_dims, load_graph, traverse_graph
from evaluation.select_best_model import select_best_astar_model

GRAPH_PATH = "data/graphs/causenet-precision.jsonl"

base_models = [
    "sentence-transformers/all-mpnet-base-v2",
    # "BAAI/bge-base-en-v1.5",
    # "ibm-granite/granite-embedding-small-english-r2",
    "BAAI/bge-large-en-v1.5",
    # "ibm-granite/granite-embedding-english-r2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen3-Embedding-0.6B",
    # "Qwen/Qwen3-Embedding-4B",
]

lightning_dir = "data/models/lightning"
fine_tuned_models = []

# Pick up all exported fine-tuned models automatically.
if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name)) and name != "old"
    ]

model_queue = base_models + fine_tuned_models

print("Model queue:", model_queue)


def detect_split(dataset_name: str):
    """
    Detect train/valid/test from the dataset name.
    """
    name = dataset_name.lower()

    if "train" in name:
        return "train"
    if "valid" in name:
        return "valid"
    if "test" in name:
        return "test"

    return "unknown"


def get_config_source_dataset_name(dataset_name: str):
    """
    Use traversal caps from the previous split.

    valid evaluation -> train p95
    test evaluation  -> valid p95
    """
    split = detect_split(dataset_name)

    if split == "valid":
        return dataset_name.replace("valid", "train")

    if split == "test":
        return dataset_name.replace("test", "valid")

    raise ValueError(
        f"No previous split available for dataset '{dataset_name}'. "
        f"Run evaluation only on valid/test with p95 configs."
    )


def build_output_paths(dataset_path: str):
    """
    Build evaluation output paths from dataset name.

    Example:
    data/datasets/msmarco_valid_filtered.json
    ->
    data/evaluation/msmarco_valid/evaluation_results.json
    data/evaluation/msmarco_valid/evaluation_results.csv
    """
    dataset_stem = Path(dataset_path).stem
    dataset_name = dataset_stem.replace("_filtered", "")

    output_dir = Path("data/evaluation") / dataset_name
    output_json_file = output_dir / "evaluation_results.json"
    output_csv_file = output_dir / "evaluation_results.csv"

    return dataset_name, output_dir, str(output_json_file), str(output_csv_file)


def load_p95_configs(eval_dataset_name: str):
    """
    Load per-model traversal caps from the previous split's
    visited_nodes_analysis.json.

    We use p95_visited_successful_only.

    Rationale:
    If 95% of successful searches finish below X visits,
    searching much longer than X is usually wasted compute.
    """
    config_source_dataset_name = get_config_source_dataset_name(eval_dataset_name)

    analysis_file = (
        Path("data/evaluation")
        / config_source_dataset_name
        / "visited_nodes_analysis.json"
    )

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Missing {analysis_file}. "
            f"Run visited_nodes_analysis.py first on '{config_source_dataset_name}'."
        )

    print(f"Loading p95 configs from: {analysis_file}")

    with open(analysis_file, "r", encoding="utf-8") as file:
        analysis_results = json.load(file)

    p95_map = {}

    for entry in analysis_results:
        model = entry.get("model")
        dimension = entry.get("dimension")

        p95_value = entry.get("analysis", {}).get(
            "p95_visited_successful_only"
        )

        if p95_value is None:
            continue

        # Max visits must be an integer.
        p95_map[(model, dimension)] = int(np.ceil(p95_value))

    print("\nLoaded p95 configs:")
    for key, value in p95_map.items():
        print(f"{key}: {value}")

    return p95_map, config_source_dataset_name


def get_astar_max_visits(p95_configs, model_name, dim, full_dim):
    """
    Get A* max visits for this model/dimension.

    Preferred:
    - exact model/dim p95

    Fallback:
    - full-dim p95

    This fallback is useful if visited_nodes_analysis.py was run only
    for the full model dimension.
    """
    if (model_name, dim) in p95_configs:
        return p95_configs[(model_name, dim)]

    if (model_name, full_dim) in p95_configs:
        print(
            f"Warning: no p95 for {model_name} dim {dim}. "
            f"Using full-dim p95 from dim {full_dim}."
        )
        return p95_configs[(model_name, full_dim)]

    raise KeyError(
        f"No p95 config found for {model_name} dim {dim} "
        f"or full dim {full_dim}."
    )


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
        "split",
        "config_source_dataset",
        "used_max_visits",
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
        "num_examples",
    ]

    rows = []

    for entry in all_results:
        used_config = entry.get("used_config", {})

        used_max_visits = (
            used_config.get("bfs_max_visits")
            or used_config.get("rl_max_visits")
            or used_config.get("astar_max_visits")
        )

        for algorithm, strategy_result in entry.get("evaluation", {}).items():
            metrics = strategy_result["metrics"]

            row = {
                "algorithm": algorithm,
                "model": entry.get("model"),
                "dimension": entry.get("dimension", ""),
                "split": entry.get("split", ""),
                "config_source_dataset": entry.get("config_source_dataset", ""),
                "used_max_visits": used_max_visits,
                "timestamp": entry.get("timestamp"),
                **metrics,
            }

            # Convert floats for German Excel.
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = str(v).replace(".", ",")

            rows.append(row)

    with open(output_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV overwritten: {output_csv_file}")


def load_results_file(output_json_file):
    if os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_result(result_entry, output_json_file, output_csv_file):
    current_results = load_results_file(output_json_file)
    current_results.append(result_entry)

    # Save JSON.
    with open(output_json_file, "w", encoding="utf-8") as file:
        json.dump(current_results, file, indent=4)

    # Save CSV.
    save_all_results_csv(current_results, output_csv_file)

    print(f"Saved results for '{result_entry['model']}'")


def calculate_metrics(y_true, y_pred, nodes_visited, path_lengths, times, path_costs):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    nodes_visited = np.array(nodes_visited, dtype=float)
    path_lengths = np.array(path_lengths, dtype=int)
    times = np.array(times, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    ).ravel()

    # Ignore empty paths when averaging path length.
    valid_lengths = path_lengths[path_lengths > 0]
    avg_path_len = float(valid_lengths.mean()) if len(valid_lengths) > 0 else 0.0

    # Only count costs that were actually computed.
    valid_costs = [c for c in path_costs if c is not None]
    avg_cost = float(np.mean(valid_costs)) if len(valid_costs) > 0 else 0.0

    # Cost per hop is more comparable than raw path cost when paths differ in length.
    cost_per_hop = []
    for path_length, path_cost in zip(path_lengths, path_costs):
        if path_cost is not None and path_length > 1:
            cost_per_hop.append(path_cost / (path_length - 1))

    avg_cost_per_hop = float(np.mean(cost_per_hop)) if cost_per_hop else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "avg_nodes_visited": float(nodes_visited.mean()) if len(nodes_visited) else 0.0,
        "avg_path_length": float(avg_path_len),
        "avg_time_sec": float(times.mean()) if len(times) else 0.0,
        "avg_path_cost": float(avg_cost),
        "avg_cost_per_hop": float(avg_cost_per_hop),
        "num_costed_paths": int(len(valid_costs)),
        "num_examples": int(len(y_true)),
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
            # Store per-example predictions so that significance tests can be
            # performed later on paired model outputs.
            "per_example": [],
        }
        for name in strategies.keys()
    }

    print(f"Starting evaluation: {description}")

    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Eval {i}/{len(data)}...")

        cause = item["cause"]
        effect = item["effect"]

        # Skip examples that are not covered by the current graph.
        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        example_id = item.get("id", i)
        true_label = bool(item["answer"])

        for name, strategy in strategies.items():
            start_time = time.time()

            path, visited_nodes = traverse_graph(
                graph,
                cause,
                effect,
                embeder,
                strategy,
                config,
            )

            elapsed = time.time() - start_time
            pred_label = bool(path)

            path_length = len(path) if path else 0
            path_cost = compute_embedding_path_cost(path, embeder)

            results[name]["y_true"].append(true_label)
            results[name]["y_pred"].append(pred_label)
            results[name]["nodes_visited"].append(visited_nodes)
            results[name]["path_lengths"].append(path_length)
            results[name]["times"].append(elapsed)
            results[name]["path_costs"].append(path_cost)

            results[name]["per_example"].append(
                {
                    "id": example_id,
                    "cause": cause,
                    "effect": effect,
                    "true": true_label,
                    "pred": pred_label,
                    "correct": pred_label == true_label,
                    "nodes_visited": int(visited_nodes),
                    "path_length": int(path_length),
                    "time_sec": float(elapsed),
                    "path_cost": path_cost,
                }
            )

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

        summary[name] = {
            "metrics": metrics,
            "per_example": results[name]["per_example"],
        }

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
    # Read dataset path from the command line so the same script can be reused.
    parser = argparse.ArgumentParser(
        description="Evaluate normalized causal dataset."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to normalized dataset JSON.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path

    dataset_name, output_dir, output_json_file, output_csv_file = build_output_paths(
        dataset_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    current_split = detect_split(dataset_name)

    print(f"Current evaluation dataset: {dataset_name}")
    print(f"Current split: {current_split}")

    selected_test_model = None

    if current_split == "test":
        selected_test_model = select_best_astar_model(dataset_name)

        print("\nTest split detected.")
        print("Ignoring full model queue for A*.")
        print("Only evaluating validation-selected model and dimension.")

    # -------------------------------------------------------------------------
    # Load p95 caps from the previous split.
    #
    # valid evaluation -> train p95
    # test evaluation  -> valid p95
    # -------------------------------------------------------------------------
    p95_configs, config_source_dataset_name = load_p95_configs(dataset_name)

    print(f"Using traversal caps from: {config_source_dataset_name}")

    print("Loading dataset...")
    with open(dataset_path, encoding="utf-8") as file:
        valid_data = json.load(file)

    print("Loading graph...")
    causal_graph = load_graph(GRAPH_PATH)

    existing_results = load_results_file(output_json_file)

    # -------------------------------------------------------------------------
    # BFS baseline
    # -------------------------------------------------------------------------
    if not any(entry["model"] == "BFS_Baseline" for entry in existing_results):
        bfs_config = {
            "bfs_max_visits": p95_configs[("BFS_Baseline", None)]
        }

        bfs_summary = run_evaluation_loop(
            valid_data,
            causal_graph,
            None,
            {"BFS": ts.bfs_traverse},
            f"BFS Baseline | {dataset_name}",
            config=bfs_config,
        )

        save_result(
            {
                "model": "BFS_Baseline",
                "dimension": None,
                "split": current_split,
                "config_source_dataset": config_source_dataset_name,
                "used_config": bfs_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": bfs_summary,
            },
            output_json_file,
            output_csv_file,
        )
    else:
        print("Skipping BFS_Baseline because it already exists.")

    existing_results = load_results_file(output_json_file)

    # -------------------------------------------------------------------------
    # RL baseline
    # -------------------------------------------------------------------------
    if not any(entry["model"] == "RL_Baseline" for entry in existing_results):
        rl_embeder = GloveEmbeder(
            "data/embeddings/glove.6B/glove.6B.300d.txt",
            DistanceMetric.COSINE,
        )

        rl_config = {
            "rl_model_path": "data/models/rl/msmarco_evaluation_state_dict.pt",
            "rl_beam_width": 5,
            "rl_max_path_len": -1,
            "rl_max_visits": p95_configs[("RL_Baseline", None)],
        }

        rl_summary = run_evaluation_loop(
            valid_data,
            causal_graph,
            rl_embeder,
            {"RL": ts.rl_traverse},
            f"RL Baseline | {dataset_name}",
            config=rl_config,
        )

        save_result(
            {
                "model": "RL_Baseline",
                "dimension": None,
                "split": current_split,
                "config_source_dataset": config_source_dataset_name,
                "used_config": rl_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": rl_summary,
            },
            output_json_file,
            output_csv_file,
        )

        del rl_embeder
        gc.collect()
    else:
        print("Skipping RL_Baseline because it already exists.")

    existing_results = load_results_file(output_json_file)

    # -------------------------------------------------------------------------
    # valid evaluation:
    # - evaluate all queued base/fine-tuned models
    # - evaluate all Matryoshka dimensions
    #
    # test evaluation:
    # - ignore the full model queue
    # - evaluate only the model/dimension selected from validation
    # -------------------------------------------------------------------------
    if current_split == "test":
        astar_model_queue = [selected_test_model["model_path"]]
        selected_test_dimension = selected_test_model["dimension"]
    else:
        astar_model_queue = model_queue
        selected_test_dimension = None

    for model_path in astar_model_queue:
        model_name = model_path.split("/")[-1]

        print(f"\nEVALUATING: {model_path}")

        try:
            distance_metric = get_model_distance_metric(model_path)
            print(f"Distance metric: {distance_metric}")
            main_embeder = STEmbedder(
                model_path=model_path,
                distance_metric=distance_metric,
            )

            full_dim = main_embeder.get_model_dim()

            if current_split == "test":
                dims = [selected_test_dimension]
            else:
                dims = get_matryoshka_dims(full_dim)

            for dim in dims:
                existing_results = load_results_file(output_json_file)

                if any(
                    entry.get("model") == model_name
                    and entry.get("dimension") == dim
                    for entry in existing_results
                ):
                    print(f"Skipping {model_name} dim {dim}")
                    continue

                print(f"--- Dim: {dim} ---")
                main_embeder.set_matryoshka_dim(dim)

                astar_max_visits = get_astar_max_visits(
                    p95_configs=p95_configs,
                    model_name=model_name,
                    dim=dim,
                    full_dim=full_dim,
                )

                astar_config = {
                    "astar_max_visits": astar_max_visits
                }

                main_summary = run_evaluation_loop(
                    valid_data,
                    causal_graph,
                    main_embeder,
                    {"A*": ts.astar_traverse},
                    f"{model_path} | dim {dim}",
                    config=astar_config,
                )

                save_result(
                    {
                        "model": model_name,
                        "model_path": model_path,
                        "dimension": dim,
                        "split": current_split,
                        "config_source_dataset": config_source_dataset_name,
                        "used_config": astar_config,
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
            print(f"Error for {model_path}: {e}")

    print("\nAll evaluations complete.")