import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import traverse_strategies as ts
from core.embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from core.utils import load_graph, traverse_graph, get_model_distance_metric

# -------------------------------------------------------------------------
# Global paths
# -------------------------------------------------------------------------

GRAPH_PATH = "data/graphs/causenet-precision.jsonl"

# Base models evaluated with A*
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

# Fine-tuned models are discovered automatically
lightning_dir = "data/models/lightning"
fine_tuned_models = []

if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name)) and name != "old"
    ]

model_queue = base_models + fine_tuned_models

print("Model queue:", model_queue)


def build_output_paths(dataset_path: str):
    """
    Build output path from dataset name.

    Example:
    data/datasets/msmarco_valid_filtered.json
    ->
    data/evaluation/msmarco_valid/visited_nodes_analysis.json
    """
    dataset_stem = Path(dataset_path).stem
    dataset_name = dataset_stem.replace("_filtered", "")

    # Detect split automatically
    if "train" in dataset_name.lower():
        split = "train"
    elif "valid" in dataset_name.lower():
        split = "valid"
    elif "test" in dataset_name.lower():
        split = "test"
    else:
        split = "unknown"

    output_dir = Path("data/evaluation") / dataset_name
    output_json_file = output_dir / "visited_nodes_analysis.json"

    return dataset_name, split, output_dir, str(output_json_file)


def load_results_file(output_json_file):
    """
    Load already collected results.

    This makes the script resumable.
    """
    if os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_result(result_entry, output_json_file):
    """
    Append one finished result entry to the JSON file.
    """
    current_results = load_results_file(output_json_file)
    current_results.append(result_entry)

    with open(output_json_file, "w", encoding="utf-8") as file:
        json.dump(current_results, file, indent=4)

    print(f"Saved visited-node results for '{result_entry['model']}'")


def already_done(existing_results, model, dimension=None):
    """
    Check whether this model/dimension was already collected.
    """
    return any(
        entry.get("model") == model
        and entry.get("dimension") == dimension
        for entry in existing_results
    )


def percentile(values, p):
    """
    Return percentile as float.
    """
    if not values:
        return 0.0

    return float(np.percentile(values, p))


def run_visited_nodes_loop(
        data,
        graph,
        embeder,
        strategy,
        strategy_name,
        description,
        config=None,
):
    """
    Run one traversal strategy and store raw visited-node data.

    This script is intentionally uncapped because it is used
    to determine p95 traversal limits later.
    """
    print(f"Starting visited-node collection: {description}")

    per_example = []
    successful_visited_counts = []
    all_visited_counts = []

    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"  Example {i}/{len(data)}...")

        cause = item["cause"]
        effect = item["effect"]

        # Skip examples not covered by graph
        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        example_id = item.get("id", i)
        true_label = bool(item["answer"])

        start_time = time.time()

        path, visited_nodes = traverse_graph(
            graph,
            cause,
            effect,
            embeder,
            strategy,
            config,
        )

        end_time = time.time()

        pred_label = bool(path)
        path_length = len(path) if path else 0
        elapsed = end_time - start_time

        all_visited_counts.append(int(visited_nodes))

        if pred_label:
            successful_visited_counts.append(int(visited_nodes))

        per_example.append(
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
                "path": path if path else [],
            }
        )

    summary = {
        "strategy": strategy_name,
        "num_examples": len(per_example),
        "num_successful_paths": len(successful_visited_counts),

        # Raw values for later plotting
        "visited_counts_all": all_visited_counts,
        "visited_counts_successful_only": successful_visited_counts,

        # Stats used later for config creation
        "max_visited_all": max(all_visited_counts) if all_visited_counts else 0,
        "max_visited_successful_only": (
            max(successful_visited_counts)
            if successful_visited_counts
            else 0
        ),
        "p95_visited_all": percentile(all_visited_counts, 95),
        "p95_visited_successful_only": percentile(
            successful_visited_counts,
            95
        ),

        # Full raw examples for later analysis
        "per_example": per_example,
    }

    print(
        f"Finished {strategy_name} | "
        f"Examples: {summary['num_examples']} | "
        f"Successful paths: {summary['num_successful_paths']} | "
        f"Max visited: {summary['max_visited_all']} | "
        f"P95 successful: {summary['p95_visited_successful_only']:.1f}"
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect uncapped visited-node distributions."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to normalized dataset JSON file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path

    # RL still needs these parameters
    RL_ANALYSIS_CONFIG = {
        "rl_model_path": "data/models/rl/msmarco_evaluation_state_dict.pt",
        "rl_beam_width": 5,
        "rl_max_path_len": -1,
    }

    dataset_name, split, output_dir, output_json_file = build_output_paths(
        dataset_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")

    print("Loading dataset...")
    with open(dataset_path, encoding="utf-8") as file:
        data = json.load(file)

    print("Loading graph...")
    causal_graph = load_graph(GRAPH_PATH)

    existing_results = load_results_file(output_json_file)

    # -------------------------------------------------------------------------
    # BFS baseline
    # -------------------------------------------------------------------------
    if not already_done(existing_results, "BFS_Baseline"):
        print("\n=== Running BFS Baseline ===")

        bfs_result = run_visited_nodes_loop(
            data=data,
            graph=causal_graph,
            embeder=None,
            strategy=ts.bfs_traverse,
            strategy_name="BFS",
            description=f"BFS | {dataset_name}",
            config=None,
        )

        save_result(
            {
                "model": "BFS_Baseline",
                "dimension": None,
                "dataset": dataset_name,
                "split": split,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": bfs_result,
            },
            output_json_file,
        )
    else:
        print("\n=== skipping BFS Baseline ===")

    # -------------------------------------------------------------------------
    # RL baseline
    # -------------------------------------------------------------------------
    if not already_done(existing_results, "RL_Baseline"):
        print("\n=== Running RL Baseline ===")

        try:
            rl_embeder = GloveEmbeder(
                "data/embeddings/glove.6B/glove.6B.300d.txt",
                DistanceMetric.COSINE,
            )

            rl_result = run_visited_nodes_loop(
                data=data,
                graph=causal_graph,
                embeder=rl_embeder,
                strategy=ts.rl_traverse,
                strategy_name="RL",
                description=f"RL | {dataset_name}",
                config=RL_ANALYSIS_CONFIG,
            )

            save_result(
                {
                    "model": "RL_Baseline",
                    "dimension": None,
                    "dataset": dataset_name,
                    "split": split,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis": rl_result,
                },
                output_json_file,
            )

            del rl_embeder
            gc.collect()

        except Exception as e:
            print(f"Failed RL: {e}")
    else:
        print("\n=== skipping RL Baseline ===")

    # -------------------------------------------------------------------------
    # A* models
    # -------------------------------------------------------------------------
    for model_path in model_queue:
        model_name = model_path.split("/")[-1]

        print(f"\nCOLLECTING A*: {model_path}")

        try:
            distance_metric = get_model_distance_metric(model_path)
            print(f"Distance metric: {distance_metric}")
            main_embeder = STEmbedder(
                model_path=model_path,
                distance_metric=distance_metric,
            )

            # Only full dimension
            model_dim = main_embeder.get_model_dim()

            existing_results = load_results_file(output_json_file)

            if already_done(existing_results, model_name, model_dim):
                print(
                    f"Skipping {model_name} dim {model_dim}"
                )
            else:
                main_embeder.set_matryoshka_dim(model_dim)

                astar_result = run_visited_nodes_loop(
                    data=data,
                    graph=causal_graph,
                    embeder=main_embeder,
                    strategy=ts.astar_traverse,
                    strategy_name="A*",
                    description=f"{model_name} | {dataset_name}",
                    config=None,
                )

                save_result(
                    {
                        "model": model_name,
                        "model_path": model_path,
                        "dimension": model_dim,
                        "dataset": dataset_name,
                        "split": split,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "analysis": astar_result,
                    },
                    output_json_file,
                )

            del main_embeder
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error for {model_path}: {e}")

    print("\nVisited-node collection complete.")
    print(f"Output JSON: {output_json_file}")
