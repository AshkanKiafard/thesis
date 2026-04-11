import csv
import json
import os
import time
from typing import Any, Dict

import torch

import traverse_strategies as ts
from embeddings import STEmbeder, DistanceMetric
from utils import get_concept, load_graph, traverse_graph

GRAPH_PATH = "../data/graphs/causenet-precision.jsonl"
DATA_PATH = "../data/datasets/msmarco_valid.json"
OUTPUT_DIR = "../data/evaluation/histogram_exports"

TARGET_MODELS = {
    "relu_cosine": "../data/models/lightning/all-mpnet-base-v2_relu_cosine_v2_finetuned",
    "relu_euclid": "../data/models/lightning/all-mpnet-base-v2_relu_euclid_v2_finetuned",
    "gelu_cosine": "../data/models/lightning/all-mpnet-base-v2_gelu_cosine_v2_finetuned",
    "gelu_euclid": "../data/models/lightning/all-mpnet-base-v2_gelu_euclid_v2_finetuned",
}

TARGET_DIMS = [64, 768]

MASTER_CONFIG = {
    "astar_max_visits": 399,
}


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def export_astar_details_for_model_dim(
    data: list[dict[str, Any]],
    graph,
    embeder: STEmbeder,
    model_label: str,
    dimension: int,
    output_path: str,
    config: Dict[str, Any] | None = None,
) -> None:
    if config is None:
        config = {}

    rows = []
    total = len(data)
    used_examples = 0
    skipped_examples = 0

    print(f"\nStarting export for {model_label}, dim={dimension}")
    print(f"Writing to: {output_path}")

    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"  {i}/{total}")

        cause = get_concept(item, 0)
        effect = get_concept(item, 1)

        # Keep behavior aligned with evaluation.py
        if cause not in graph.nodes or effect not in graph.nodes:
            skipped_examples += 1
            continue

        used_examples += 1

        true_label = item["answer:Extracted"][0] == "Yes"

        path, visited_nodes = traverse_graph(
            graph,
            cause,
            effect,
            embeder,
            ts.astar_traverse,
            config,
        )

        path_found = bool(path)
        hop_count = max(len(path) - 1, 0) if path else 0

        rows.append(
            {
                "model": model_label,
                "dimension": dimension,
                "cause": cause,
                "effect": effect,
                "true_label": true_label,
                "path_found": path_found,
                "hop_count": hop_count,
                "nodes_visited": visited_nodes,
            }
        )

    fieldnames = [
        "model",
        "dimension",
        "cause",
        "effect",
        "true_label",
        "path_found",
        "hop_count",
        "nodes_visited",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {len(rows)} rows written")
    print(f"Used examples: {used_examples}")
    print(f"Skipped examples (not in graph): {skipped_examples}")


if __name__ == "__main__":
    start_all = time.time()

    ensure_directory(OUTPUT_DIR)

    print("Loading validation data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    print("Loading causal graph...")
    causal_graph = load_graph(GRAPH_PATH)

    for short_name, model_path in TARGET_MODELS.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping missing model path: {model_path}")
            continue

        distance_metric = (
            DistanceMetric.EUCLIDEAN if "euclid" in short_name else DistanceMetric.COSINE
        )

        print(f"\n=== Loading model: {short_name} ===")
        print(f"Path: {model_path}")
        print(f"Distance: {distance_metric}")

        try:
            embeder = STEmbeder(
                model_path=model_path,
                distance_metric=distance_metric,
            )

            for dim in TARGET_DIMS:
                embeder.set_matryoshka_dim(dim)

                output_file = os.path.join(
                    OUTPUT_DIR,
                    f"{short_name}_dim{dim}_astar_details.csv"
                )

                export_astar_details_for_model_dim(
                    data=valid_data,
                    graph=causal_graph,
                    embeder=embeder,
                    model_label=short_name,
                    dimension=dim,
                    output_path=output_file,
                    config=MASTER_CONFIG,
                )

            del embeder
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error while processing {short_name}: {e}")

    end_all = time.time()
    print(f"\nAll exports complete in {end_all - start_all:.2f} seconds.")