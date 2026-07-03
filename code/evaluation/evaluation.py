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
from core.config import (
    BASE_MODELS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_DIR,
    EMBEDDING_INDEX_MIN_SUCCESSORS,
    GLOVE_300D_PATH,
    LIGHTNING_DIR,
)
from core.constants import (
    BFS_CAPPED_BASELINE_MODEL,
    BFS_UNCAPPED_BASELINE_MODEL,
    RL_BASELINE_MODEL,
)
from core.embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from core.graph_config import (
    DEFAULT_GRAPH_NAME,
    get_graph_label,
    get_graph_path,
    graph_choices,
)
from core.embedding_preload import preload_graph_embeddings, preload_rl_embeddings
from core.utils import (
    get_ablation_fine_tuned_models,
    get_ablation_model_names,
    get_ablation_reference_model_name,
    get_embedding_cache_suffix,
    get_fine_tuned_models,
    get_model_distance_metric,
    get_matryoshka_dims,
    get_node_universe_for_graph,
    sort_model_queue,
    load_causal_graph,
    load_rl_graph,
    traverse_graph,
)
from evaluation.select_best_model import select_best_astar_model, print_selection

EVALUATION_OUTPUT_ROOT = Path("data/evaluation")


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
    Use traversal caps from the train split.

    valid evaluation -> train p95
    test evaluation  -> train p95
    """
    split = detect_split(dataset_name)

    if split == "valid":
        return dataset_name.replace("valid", "train")

    if split == "test":
        return dataset_name.replace("test", "train")

    raise ValueError(
        f"No train split cap source available for dataset '{dataset_name}'. "
        f"Run evaluation only on valid/test with p95 configs."
    )


def get_evaluation_output_root(ablation: bool = False):
    return EVALUATION_OUTPUT_ROOT / "ablation" if ablation else EVALUATION_OUTPUT_ROOT


def build_output_paths(
    dataset_path: str,
    run_suffix: str,
    graph_name: str,
    ablation: bool = False,
):
    """
    Build evaluation output paths from dataset name and run suffix.

    Example:
    data/datasets/msmarco_valid_filtered.json + v3
    ->
    data/evaluation/causenet/msmarco_valid/v3/evaluation_results.json
    data/evaluation/causenet/msmarco_valid/v3/evaluation_results.csv
    """
    dataset_stem = Path(dataset_path).stem
    dataset_name = dataset_stem.replace("_filtered", "")

    output_dir = get_evaluation_output_root(ablation) / graph_name / dataset_name / run_suffix
    output_json_file = output_dir / "evaluation_results.json"
    output_csv_file = output_dir / "evaluation_results.csv"

    return dataset_name, output_dir, str(output_json_file), str(output_csv_file)


def get_model_name(model_path: str):
    return Path(model_path).name


def get_p95_analysis_file(
    dataset_name: str,
    run_suffix: str,
    graph_name: str,
    ablation: bool = False,
):
    return (
        get_evaluation_output_root(ablation)
        / graph_name
        / dataset_name
        / run_suffix
        / "visited_nodes_analysis.json"
    )


def get_evaluation_results_file(
    dataset_name: str,
    run_suffix: str,
    graph_name: str,
    ablation: bool = False,
):
    return (
        get_evaluation_output_root(ablation)
        / graph_name
        / dataset_name
        / run_suffix
        / "evaluation_results.json"
    )


def load_p95_configs(
    eval_dataset_name: str,
    run_suffix: str,
    graph_name: str,
    config_source_dataset_name: str = None,
    config_source_graph_name: str = None,
    fallback_config_source_dataset_name: str = None,
    fallback_config_source_graph_name: str = DEFAULT_GRAPH_NAME,
    ablation: bool = False,
):
    """
    Load per-model traversal caps from visited_nodes_analysis.json.

    We use p95_visited_successful_only.

    default source -> data/evaluation/<graph>/<train_dataset>/<run_suffix>/visited_nodes_analysis.json
    fallback source -> data/evaluation/<fallback_graph>/<fallback_dataset>/<run_suffix>/visited_nodes_analysis.json

    --config-source-dataset forces a specific source.
    --config-source-graph selects the graph namespace for that explicit source.
    --fallback-config-source-dataset and --fallback-config-source-graph are used
    only if the default source is missing.
    """
    if config_source_dataset_name:
        explicit_graph_name = config_source_graph_name or graph_name
        candidate_sources = [
            (config_source_dataset_name, explicit_graph_name, "explicit")
        ]
    else:
        default_source_dataset_name = get_config_source_dataset_name(eval_dataset_name)
        candidate_sources = [(default_source_dataset_name, graph_name, "default")]

        seen_sources = {(default_source_dataset_name, graph_name)}

        if (
                fallback_config_source_dataset_name
                and (
                    fallback_config_source_dataset_name,
                    fallback_config_source_graph_name,
                )
                not in seen_sources
        ):
            candidate_sources.append(
                (
                    fallback_config_source_dataset_name,
                    fallback_config_source_graph_name,
                    "fallback",
                )
            )
            seen_sources.add(
                (
                    fallback_config_source_dataset_name,
                    fallback_config_source_graph_name,
                )
            )

        if ("msmarco_train", fallback_config_source_graph_name) not in seen_sources:
            candidate_sources.append(
                ("msmarco_train", fallback_config_source_graph_name, "fallback")
            )

    missing_files = []
    selected_source_dataset_name = None
    selected_source_graph_name = None
    analysis_file = None

    for candidate_source_dataset_name, candidate_source_graph_name, source_kind in (
        candidate_sources
    ):
        candidate_file = get_p95_analysis_file(
            candidate_source_dataset_name,
            run_suffix,
            candidate_source_graph_name,
            ablation=ablation,
        )

        if candidate_file.exists():
            selected_source_dataset_name = candidate_source_dataset_name
            selected_source_graph_name = candidate_source_graph_name
            analysis_file = candidate_file

            if source_kind == "fallback":
                print(
                    "Default p95 config source missing. "
                    f"Falling back to: "
                    f"{selected_source_graph_name}/{selected_source_dataset_name}"
                )

            break

        if analysis_file is not None:
            break

        missing_files.append(candidate_file)

    if analysis_file is None:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Missing p95 config source for '{eval_dataset_name}' "
            f"with run suffix '{run_suffix}'. Tried:\n{missing_text}"
        )

    print(f"Loading p95 configs from: {analysis_file}")

    with open(analysis_file, "r", encoding="utf-8") as file:
        analysis_results = json.load(file)

    p95_map = {}

    for entry in analysis_results:
        model = entry.get("model")
        dimension = entry.get("dimension")
        strategy = entry.get("analysis", {}).get("strategy")

        p95_value = entry.get("analysis", {}).get(
            "p95_visited_successful_only"
        )

        if p95_value is None:
            continue

        if strategy is None:
            raise ValueError(
                f"Missing strategy in visited-node entry for "
                f"{model} dim {dimension}. Regenerate visited_nodes_analysis.json."
            )

        p95_map[(model, dimension, strategy)] = int(np.ceil(p95_value))

    print("\nLoaded p95 configs:")
    for key, value in p95_map.items():
        print(f"{key}: {value}")

    return p95_map, selected_source_dataset_name, selected_source_graph_name


def get_p95_cap(p95_configs, model, dimension, strategy):
    strategy_key = (model, dimension, strategy)
    if strategy_key in p95_configs:
        return p95_configs[strategy_key]

    raise KeyError(f"Missing p95 config for {strategy_key}")


def use_shared_ablation_caps(
    p95_configs,
    reference_model_name,
    ablation_model_paths,
    dimension,
    include_dijkstra=True,
):
    """
    Apply the selected main model's traversal budget to every ablation model.

    This keeps the ablation comparison fixed-budget: activation/distance changes,
    but the search budget stays the same as the selected reference model.
    """
    strategies = ["A*"]
    if include_dijkstra:
        strategies.append("Dijkstra")

    shared_configs = {}
    ablation_model_names = [get_model_name(path) for path in ablation_model_paths]

    print("\nUsing shared ablation traversal caps:")
    print(f"Reference model: {reference_model_name}")
    print(f"Reference dim: {dimension}")

    for strategy in strategies:
        reference_cap = get_p95_cap(
            p95_configs,
            reference_model_name,
            dimension,
            strategy,
        )
        print(f"{strategy}: {reference_cap}")

        for model_name in ablation_model_names:
            shared_configs[(model_name, dimension, strategy)] = reference_cap

    return shared_configs


def get_bfs_max_visits_cap(p95_configs):
    """
    The capped BFS evaluation uses the p95 from an uncapped BFS analysis.

    Older visited_nodes_analysis.json files stored that source run as
    BFS_Baseline. Newer files use BFS_Uncapped_Baseline.
    """
    for model in (BFS_UNCAPPED_BASELINE_MODEL, BFS_CAPPED_BASELINE_MODEL):
        strategy_key = (model, None, "BFS")
        if strategy_key in p95_configs:
            return p95_configs[strategy_key]

    raise KeyError(
        "Missing p95 config for uncapped BFS source. Tried "
        f"{BFS_UNCAPPED_BASELINE_MODEL} and {BFS_CAPPED_BASELINE_MODEL}."
    )


def compute_embedding_path_cost(path, embeder):
    if not path:
        return None

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


def run_warmup_traversal(data, graph, embeder, strategy, strategy_name, config=None):
    """
    Run one untimed traversal before evaluation.

    This removes startup artifacts from timing:
    - CUDA/model lazy initialization
    - first embedding/cache access
    - first graph traversal overhead

    The result is ignored and not stored.
    """
    for item in data:
        cause = item["cause"]
        effect = item["effect"]

        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        strategy_config = config
        if strategy_name == "RL":
            strategy_config = dict(config) if config is not None else {}
            strategy_config["question"] = item.get(
                "question",
                f"can {cause} cause {effect}?"
            )

        traverse_graph(
            graph,
            cause,
            effect,
            embeder,
            strategy,
            strategy_config,
        )

        break


def save_all_results_csv(all_results, output_csv_file):
    fieldnames = [
        "algorithm",
        "model",
        "dimension",
        "split",
        "run_suffix",
        "config_source_dataset",
        "embedding_device",
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
        "avg_time_ms",
        "avg_path_cost",
        "avg_cost_per_hop",
        "num_costed_paths",
        "num_examples",
    ]

    rows = []

    for entry in all_results:
        used_config = entry.get("used_config", {})

        for algorithm, strategy_result in entry.get("evaluation", {}).items():
            metrics = strategy_result["metrics"]
            used_max_visits = get_used_max_visits(used_config, algorithm)

            row = {
                "algorithm": algorithm,
                "model": entry.get("model"),
                "dimension": entry.get("dimension", ""),
                "split": entry.get("split", ""),
                "run_suffix": entry.get("run_suffix", ""),
                "config_source_dataset": entry.get("config_source_dataset", ""),
                "embedding_device": entry.get("embedding_device", ""),
                "used_max_visits": used_max_visits,
                "timestamp": entry.get("timestamp"),
                **metrics,
            }

            rows.append(row)

    with open(output_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV overwritten: {output_csv_file}")


def get_used_max_visits(used_config, algorithm):
    if algorithm == "BFS":
        return used_config.get("bfs_max_visits")

    if algorithm == "RL":
        return used_config.get("rl_max_visits")

    if algorithm == "Dijkstra":
        return used_config.get("dijkstra_max_visits")

    if algorithm == "A*":
        return used_config.get("astar_max_visits")

    return None


def strip_runtime_config(config):
    if config is None:
        return {}

    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def load_results_file(output_json_file):
    if os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def save_result(
    result_entry,
    output_json_file,
    output_csv_file,
    replace_existing=None,
):
    current_results = load_results_file(output_json_file)

    if replace_existing is not None:
        original_count = len(current_results)
        current_results = [
            entry for entry in current_results
            if not replace_existing(entry)
        ]
        removed_count = original_count - len(current_results)

        if removed_count:
            print(
                f"Removed {removed_count} existing result(s) before saving "
                f"'{result_entry['model']}'."
            )

    current_results.append(result_entry)

    with open(output_json_file, "w", encoding="utf-8") as file:
        json.dump(current_results, file, indent=4)

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

    valid_lengths = path_lengths[path_lengths > 0]
    avg_path_len = float(valid_lengths.mean()) if len(valid_lengths) > 0 else 0.0

    valid_costs = [c for c in path_costs if c is not None]
    avg_cost = float(np.mean(valid_costs)) if len(valid_costs) > 0 else 0.0

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
        "avg_time_ms": float(times.mean() * 1000) if len(times) else 0.0,
        "avg_path_cost": float(avg_cost),
        "avg_cost_per_hop": float(avg_cost_per_hop),
        "num_costed_paths": int(len(valid_costs)),
        "num_examples": int(len(y_true)),
    }


def run_evaluation_loop(data, graph, embeder, strategies, description, config=None):
    results = {
        name: {
            "y_true": [],
            "y_pred": [],
            "nodes_visited": [],
            "path_lengths": [],
            "times": [],
            "path_costs": [],
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

        if cause not in graph.nodes or effect not in graph.nodes:
            continue

        example_id = item.get("id", i)
        true_label = bool(item["answer"])

        for name, strategy in strategies.items():
            strategy_config = config
            if name == "RL":
                strategy_config = dict(config) if config is not None else {}
                strategy_config["question"] = item.get(
                    "question",
                    f"can {cause} cause {effect}?"
                )

            start_time = time.time()

            path, visited_nodes = traverse_graph(
                graph,
                cause,
                effect,
                embeder,
                strategy,
                strategy_config,
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
                    "time_ms": float(elapsed * 1000),
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
            f"Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"Avg Nodes: {metrics['avg_nodes_visited']:.1f} | "
            f"Avg Time: {metrics['avg_time_ms']:.2f} ms"
        )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate normalized causal dataset."
    )
    parser.add_argument(
        "dataset_path",
        help="Path to normalized dataset JSON.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        required=True,
        help="Final-training run suffix, e.g. v3.",
    )
    parser.add_argument(
        "--graph",
        choices=graph_choices(),
        default=DEFAULT_GRAPH_NAME,
        help="Graph to evaluate with. Defaults to CauseNet.",
    )
    parser.add_argument(
        "--config-source-dataset",
        type=str,
        default=None,
        help=(
            "Force traversal caps from this dataset's visited_nodes_analysis.json, "
            "e.g. msmarco_train."
        ),
    )
    parser.add_argument(
        "--config-source-graph",
        choices=graph_choices(),
        default=None,
        help=(
            "Graph namespace for --config-source-dataset. Defaults to the "
            "evaluated graph if omitted."
        ),
    )
    parser.add_argument(
        "--fallback-config-source-dataset",
        type=str,
        default=None,
        help=(
            "Use this dataset's traversal caps only if the default cap source "
            "is missing, e.g. msmarco_train for sem_test."
        ),
    )
    parser.add_argument(
        "--fallback-config-source-graph",
        choices=graph_choices(),
        default=DEFAULT_GRAPH_NAME,
        help=(
            "Graph namespace for fallback traversal caps. Defaults to CauseNet, "
            "so CausalBank test runs can reuse CauseNet msmarco_train p95 caps."
        ),
    )
    parser.add_argument(
        "--ablation-cap-source-dataset",
        type=str,
        default="msmarco_train",
        help=(
            "In --ablation mode, use this normal evaluation dataset's "
            "visited-node caps from the reference model. Default: msmarco_train."
        ),
    )
    parser.add_argument(
        "--ablation-cap-source-graph",
        choices=graph_choices(),
        default=DEFAULT_GRAPH_NAME,
        help=(
            "In --ablation mode, use this normal evaluation graph namespace "
            "for reference-model visited-node caps. Default: causenet."
        ),
    )
    parser.add_argument(
        "--ablation-cap-reference-model",
        type=str,
        default=None,
        help=(
            "Reference model name whose p95 caps are shared by all ablation "
            "models. Defaults to the main Granite relu/euclid finetuned model "
            "for --run-suffix."
        ),
    )
    parser.add_argument(
        "--skip-embedding-preload",
        action="store_true",
        help=(
            "Do not load the precomputed ST graph-node embedding index before "
            "timed A*/Dijkstra evaluation."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Batch size used when loading or backfilling the ST embedding index.",
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Torch device for ST/GloVe embedding tensors and ST model encoding. "
            "Use cpu to benchmark traversal distance computation without GPU "
            "synchronization overhead. Default: auto."
        ),
    )
    parser.add_argument(
        "--no-save-embedding-cache",
        action="store_true",
        help=f"Do not persist newly preloaded ST embeddings to {EMBEDDINGS_DIR}.",
    )
    parser.add_argument(
        "--skip-bfs-baseline",
        action="store_true",
        help=(
            "Skip the capped BFS_Baseline even if it is missing from the "
            "output file."
        ),
    )
    parser.add_argument(
        "--skip-bfs-uncapped-baseline",
        action="store_true",
        help=(
            "Skip BFS_Uncapped_Baseline even if it is missing from the "
            "output file."
        ),
    )
    parser.add_argument(
        "--skip-rl-baseline",
        action="store_true",
        help="Skip RL_Baseline and avoid loading the separate RL graph.",
    )
    parser.add_argument(
        "--skip-dijkstra",
        action="store_true",
        help="Skip Dijkstra model evaluation while still evaluating A*.",
    )
    parser.add_argument(
        "--skip-dijakstra",
        dest="skip_dijkstra",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-baselines",
        action="store_true",
        help=(
            "Rerun BFS_Uncapped_Baseline, BFS_Baseline, and RL_Baseline even "
            "if they already exist. Existing baseline entries are replaced; "
            "model entries are kept."
        ),
    )
    parser.add_argument(
        "--force-bfs-baseline",
        action="store_true",
        help=(
            "Rerun the capped BFS_Baseline even if it already exists. "
            "Existing BFS_Baseline entries are replaced."
        ),
    )
    parser.add_argument(
        "--force-bfs-uncapped-baseline",
        action="store_true",
        help=(
            "Rerun BFS_Uncapped_Baseline even if it already exists. Existing "
            "BFS_Uncapped_Baseline entries are replaced."
        ),
    )
    parser.add_argument(
        "--force-rl-baseline",
        action="store_true",
        help=(
            "Rerun RL_Baseline even if it already exists. Existing RL_Baseline "
            "entries are replaced."
        ),
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help=(
            "Run only baseline handling, then exit before A*/Dijkstra model "
            "evaluation. Combine with --force-baselines to refresh baselines "
            "without touching model rows."
        ),
    )
    parser.add_argument(
        "--force-model-results",
        action="store_true",
        help=(
            "Rerun A*/Dijkstra model evaluations even if entries already exist. "
            "Existing entries for the same model and dimension are replaced."
        ),
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        default=None,
        help=(
            "Explicit A* model path to evaluate on the test split. "
            "If either best-model argument is omitted, the test split uses "
            "select_best_model.py."
        ),
    )
    parser.add_argument(
        "--best-model-dim",
        type=int,
        default=None,
        help=(
            "Explicit Matryoshka dimension for --best-model-path. "
            "If either best-model argument is omitted, the test split uses "
            "select_best_model.py."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help=(
            "Only evaluate this Matryoshka dimension in --ablation mode, "
            "e.g. 32."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Evaluate the main Granite reference and all three "
            "activation/distance variants together, skip baselines, and "
            "write under data/evaluation/ablation. All four model rows are "
            "refreshed together so runtime results come from one process."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = args.dataset_path
    run_suffix = args.run_suffix
    graph_name = args.graph
    graph_label = get_graph_label(graph_name)
    graph_path = get_graph_path(graph_name)

    if args.dim is not None and args.dim <= 0:
        raise ValueError("--dim must be greater than 0")
    if args.ablation and args.dim is None:
        raise ValueError("--ablation requires --dim, e.g. --dim 32")
    if args.ablation and (args.best_model_path is not None or args.best_model_dim is not None):
        raise ValueError("--ablation cannot be combined with --best-model-path/--best-model-dim")
    if args.ablation:
        args.skip_dijkstra = True
        args.force_model_results = True

    print(f"Run suffix: {run_suffix}")
    print(f"Ablation mode: {args.ablation}")
    print(f"Graph: {graph_label} ({graph_name})")
    print(f"Graph path: {graph_path}")
    print(f"Embedding device: {args.embedding_device}")
    if args.dim is not None:
        print(f"Restricted Matryoshka dim: {args.dim}")
    embedding_cache_suffix = get_embedding_cache_suffix(graph_name)
    node_universe = get_node_universe_for_graph(graph_name)
    if embedding_cache_suffix:
        print(f"Embedding cache suffix: {embedding_cache_suffix}")
    print(f"Embedding node universe: {node_universe}")

    dataset_name, output_dir, output_json_file, output_csv_file = build_output_paths(
        dataset_path,
        run_suffix,
        graph_name,
        ablation=args.ablation,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    current_split = detect_split(dataset_name)

    print(f"Current evaluation dataset: {dataset_name}")
    print(f"Current split: {current_split}")
    print(f"Output directory: {output_dir}")

    model_queue = []
    selected_test_model_path = None
    selected_test_dimension = None

    if args.baselines_only:
        print("\nBaselines-only mode enabled.")
        print("Skipping embedding-guided model selection and model queue.")
    elif args.ablation:
        print("\nAblation mode enabled.")
        print(
            "Evaluating the main reference and all three ablation variants "
            "at the requested dimension in one process."
        )
        print(
            "Refreshing every comparison row to keep runtime measurements "
            "from the same evaluation run."
        )
        model_queue = get_ablation_fine_tuned_models(run_suffix)
        expected_model_names = set(get_ablation_model_names(run_suffix))
        found_model_names = {get_model_name(path) for path in model_queue}
        missing_model_names = sorted(expected_model_names - found_model_names)
        if missing_model_names:
            raise FileNotFoundError(
                "The four-model ablation comparison is incomplete for run "
                f"suffix '{run_suffix}' in {LIGHTNING_DIR}. Missing: "
                f"{missing_model_names}"
            )
        print("Joint ablation comparison model queue:", model_queue)
    elif current_split == "test":
        print("\nTest split detected.")
        print("Ignoring full model queue for embedding-guided strategies.")

        if args.best_model_path is not None and args.best_model_dim is not None:
            selected_test_model_path = args.best_model_path
            selected_test_dimension = args.best_model_dim

            print(
                "Only evaluating explicitly provided model and dimension: "
                f"{selected_test_model_path} | dim {selected_test_dimension}"
            )
        else:
            if args.best_model_path is not None or args.best_model_dim is not None:
                print(
                    "Only one best-model parameter was provided. "
                    "Falling back to select_best_model.py for both values."
                )

            valid_dataset_name = dataset_name.replace("test", "valid")
            valid_results_file = get_evaluation_results_file(
                valid_dataset_name,
                run_suffix,
                graph_name,
            )

            selection = select_best_astar_model(valid_results_file)
            print_selection(selection)

            selected_test_model = selection.get("best")

            if selected_test_model is None:
                raise ValueError(
                    f"No selected A* model found in {valid_results_file}. "
                    "Run validation evaluation first or check select_best_model.py."
                )

            selected_test_model_path = selected_test_model["model_path"]
            selected_test_dimension = selected_test_model["dimension"]

            print(
                "Only evaluating selected model and dimension: "
                f"{selected_test_model_path} | dim {selected_test_dimension}"
            )
    else:
        fine_tuned_models = get_fine_tuned_models(run_suffix)
        model_queue = sort_model_queue(list(BASE_MODELS) + fine_tuned_models, run_suffix)
        print("Model queue:", model_queue)

    (
        p95_configs,
        config_source_dataset_name,
        config_source_graph_name,
    ) = load_p95_configs(
        dataset_name,
        run_suffix,
        graph_name,
        config_source_dataset_name=(
            args.ablation_cap_source_dataset
            if args.ablation
            else args.config_source_dataset
        ),
        config_source_graph_name=(
            args.ablation_cap_source_graph
            if args.ablation
            else args.config_source_graph
        ),
        fallback_config_source_dataset_name=args.fallback_config_source_dataset,
        fallback_config_source_graph_name=args.fallback_config_source_graph,
        ablation=False if args.ablation else args.ablation,
    )

    if args.ablation:
        ablation_reference_model = (
            args.ablation_cap_reference_model
            or get_ablation_reference_model_name(run_suffix)
        )
        p95_configs = use_shared_ablation_caps(
            p95_configs,
            ablation_reference_model,
            model_queue,
            args.dim,
            include_dijkstra=not args.skip_dijkstra,
        )
    else:
        ablation_reference_model = None

    print(
        "Using traversal caps from: "
        f"{config_source_graph_name}/{config_source_dataset_name}/{run_suffix}"
    )

    print("Loading dataset...")
    with open(dataset_path, encoding="utf-8") as file:
        valid_data = json.load(file)

    force_bfs_uncapped_baseline = (
        args.force_baselines or args.force_bfs_uncapped_baseline
    )
    force_bfs_baseline = args.force_baselines or args.force_bfs_baseline
    force_rl_baseline = args.force_baselines or args.force_rl_baseline

    existing_results = load_results_file(output_json_file)
    has_bfs_uncapped_baseline = any(
        entry.get("model") == BFS_UNCAPPED_BASELINE_MODEL
        for entry in existing_results
    )
    has_bfs_baseline = any(
        entry.get("model") == BFS_CAPPED_BASELINE_MODEL
        for entry in existing_results
    )
    has_rl_baseline = any(
        entry.get("model") == RL_BASELINE_MODEL
        for entry in existing_results
    )
    should_run_bfs_uncapped_baseline = (
        not args.skip_bfs_uncapped_baseline
        and not args.ablation
        and (force_bfs_uncapped_baseline or not has_bfs_uncapped_baseline)
    )
    should_run_bfs_baseline = (
        not args.skip_bfs_baseline
        and not args.ablation
        and (force_bfs_baseline or not has_bfs_baseline)
    )
    should_run_rl_baseline = (
        not args.skip_rl_baseline
        and not args.ablation
        and (force_rl_baseline or not has_rl_baseline)
    )

    if args.ablation:
        print("Skipping all baselines in ablation mode.")

    if (
        force_bfs_uncapped_baseline
        and has_bfs_uncapped_baseline
        and should_run_bfs_uncapped_baseline
    ):
        print(f"Force rerun enabled for {BFS_UNCAPPED_BASELINE_MODEL}.")

    if force_bfs_baseline and has_bfs_baseline and should_run_bfs_baseline:
        print(f"Force rerun enabled for {BFS_CAPPED_BASELINE_MODEL}.")

    if force_rl_baseline and has_rl_baseline and should_run_rl_baseline:
        print(f"Force rerun enabled for {RL_BASELINE_MODEL}.")

    print(f"Loading causal graph from: {graph_path}")
    graph_load_start = time.time()
    causal_graph = load_causal_graph(
        graph_path,
        use_inverse=False,
        progress_every=1_000_000,
        progress_label=f"{graph_name} NetworkX graph",
    )
    print(
        "Loaded causal graph: "
        f"{causal_graph.number_of_nodes():,} nodes, "
        f"{causal_graph.number_of_edges():,} edges "
        f"in {time.time() - graph_load_start:.1f}s"
    )

    if should_run_bfs_uncapped_baseline:
        bfs_uncapped_config = {"bfs_max_visits": -1}

        run_warmup_traversal(
            valid_data,
            causal_graph,
            None,
            ts.bfs_traverse,
            "BFS",
            config=bfs_uncapped_config,
        )

        bfs_uncapped_summary = run_evaluation_loop(
            valid_data,
            causal_graph,
            None,
            {"BFS": ts.bfs_traverse},
            f"BFS Uncapped Baseline | {dataset_name} | {run_suffix}",
            config=bfs_uncapped_config,
        )

        save_result(
            {
                "model": BFS_UNCAPPED_BASELINE_MODEL,
                "dimension": None,
                "split": current_split,
                "run_suffix": run_suffix,
                "config_source_dataset": config_source_dataset_name,
                "used_config": bfs_uncapped_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": bfs_uncapped_summary,
            },
            output_json_file,
            output_csv_file,
            replace_existing=(
                (
                    lambda entry: entry.get("model")
                    == BFS_UNCAPPED_BASELINE_MODEL
                )
                if force_bfs_uncapped_baseline
                else None
            ),
        )
    else:
        if args.skip_bfs_uncapped_baseline:
            print(
                f"Skipping {BFS_UNCAPPED_BASELINE_MODEL} because "
                "--skip-bfs-uncapped-baseline was set."
            )
        elif has_bfs_uncapped_baseline:
            print(f"Skipping {BFS_UNCAPPED_BASELINE_MODEL} because it already exists.")
        else:
            print(f"Skipping {BFS_UNCAPPED_BASELINE_MODEL}.")

    if should_run_bfs_baseline:
        bfs_config = {"bfs_max_visits": get_bfs_max_visits_cap(p95_configs)}

        run_warmup_traversal(
            valid_data,
            causal_graph,
            None,
            ts.bfs_traverse,
            "BFS",
            config=bfs_config,
        )

        bfs_summary = run_evaluation_loop(
            valid_data,
            causal_graph,
            None,
            {"BFS": ts.bfs_traverse},
            f"BFS Capped Baseline | {dataset_name} | {run_suffix}",
            config=bfs_config,
        )

        save_result(
            {
                "model": BFS_CAPPED_BASELINE_MODEL,
                "dimension": None,
                "split": current_split,
                "run_suffix": run_suffix,
                "config_source_dataset": config_source_dataset_name,
                "used_config": bfs_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": bfs_summary,
            },
            output_json_file,
            output_csv_file,
            replace_existing=(
                (
                    lambda entry: entry.get("model")
                    == BFS_CAPPED_BASELINE_MODEL
                )
                if force_bfs_baseline
                else None
            ),
        )
    else:
        if args.skip_bfs_baseline:
            print(
                f"Skipping {BFS_CAPPED_BASELINE_MODEL} because "
                "--skip-bfs-baseline was set."
            )
        elif has_bfs_baseline:
            print(f"Skipping {BFS_CAPPED_BASELINE_MODEL} because it already exists.")
        else:
            print(f"Skipping {BFS_CAPPED_BASELINE_MODEL}.")

    existing_results = load_results_file(output_json_file)

    if should_run_rl_baseline:
        print(f"Loading RL graph from: {graph_path}")
        graph_load_start = time.time()
        rl_graph = load_rl_graph(
            graph_path,
            use_inverse=False,
            progress_every=1_000_000,
            progress_label=f"{graph_name} RL graph",
        )
        print(
            "Loaded RL graph: "
            f"{len(rl_graph.nodes):,} nodes in "
            f"{time.time() - graph_load_start:.1f}s"
        )

        rl_embeder = GloveEmbeder(
            GLOVE_300D_PATH,
            DistanceMetric.COSINE,
            device=args.embedding_device,
        )

        rl_config = {
            "rl_model_path": "data/models/rl/msmarco_no_inverse_state_dict.pt",
            "rl_beam_width": 50,
            "rl_max_path_len": 2,
            "rl_max_actions": 5000,
            "rl_max_visits": -1,
        }

        preload_rl_embeddings(
            rl_embeder,
            rl_graph,
            data=valid_data,
        )

        run_warmup_traversal(
            valid_data,
            rl_graph,
            rl_embeder,
            ts.rl_traverse,
            "RL",
            config=rl_config,
        )

        rl_summary = run_evaluation_loop(
            valid_data,
            rl_graph,
            rl_embeder,
            {"RL": ts.rl_traverse},
            f"RL Baseline | {dataset_name} | {run_suffix}",
            config=rl_config,
        )

        save_result(
            {
                "model": RL_BASELINE_MODEL,
                "dimension": None,
                "split": current_split,
                "run_suffix": run_suffix,
                "config_source_dataset": config_source_dataset_name,
                "embedding_device": args.embedding_device,
                "used_config": rl_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation": rl_summary,
            },
            output_json_file,
            output_csv_file,
            replace_existing=(
                (lambda entry: entry.get("model") == RL_BASELINE_MODEL)
                if force_rl_baseline
                else None
            ),
        )

        del rl_embeder
        del rl_graph
        gc.collect()
    else:
        if args.skip_rl_baseline:
            print(
                f"Skipping {RL_BASELINE_MODEL} because "
                "--skip-rl-baseline was set."
            )
        elif has_rl_baseline:
            print(f"Skipping {RL_BASELINE_MODEL} because it already exists.")
        else:
            print(f"Skipping {RL_BASELINE_MODEL}.")

    if args.baselines_only:
        print(
            "\nBaseline handling complete. "
            "Skipping A*/Dijkstra model evaluation because --baselines-only was set."
        )
        raise SystemExit(0)

    if args.skip_dijkstra:
        print("Skipping Dijkstra model evaluation because --skip-dijkstra was set.")

    if current_split == "test" and not args.ablation:
        semantic_model_queue = [selected_test_model_path]
    else:
        semantic_model_queue = model_queue

    evaluation_errors = []

    for model_path in semantic_model_queue:
        model_name = get_model_name(model_path)

        print(f"\nEVALUATING: {model_path}")

        try:
            distance_metric = get_model_distance_metric(model_path)
            print(f"Distance metric: {distance_metric}")

            main_embeder = STEmbedder(
                model_path=model_path,
                distance_metric=distance_metric,
                device=args.embedding_device,
                cache_suffix=embedding_cache_suffix,
                node_universe=node_universe,
            )

            full_dim = main_embeder.get_model_dim()

            if args.ablation:
                if args.dim > full_dim:
                    raise ValueError(
                        f"Requested dim {args.dim}, but {model_path} only "
                        f"has {full_dim} embedding dimensions."
                    )
                dims = [args.dim]
            elif current_split == "test":
                dims = [selected_test_dimension]
            else:
                dims = get_matryoshka_dims(full_dim)

            existing_results = load_results_file(output_json_file)
            pending_work = []

            for dim in dims:
                if args.force_model_results:
                    completed_algorithms = set()
                else:
                    completed_algorithms = {
                        algorithm
                        for entry in existing_results
                        if entry.get("model") == model_name
                        and entry.get("dimension") == dim
                        for algorithm in entry.get("evaluation", {}).keys()
                    }

                pending_strategies = {}
                used_config = {}

                if "A*" not in completed_algorithms:
                    used_config["astar_max_visits"] = get_p95_cap(
                        p95_configs,
                        model_name,
                        dim,
                        "A*",
                    )
                    pending_strategies["A*"] = ts.astar_traverse

                if (
                    not args.skip_dijkstra
                    and "Dijkstra" not in completed_algorithms
                ):
                    used_config["dijkstra_max_visits"] = get_p95_cap(
                        p95_configs,
                        model_name,
                        dim,
                        "Dijkstra",
                    )
                    pending_strategies["Dijkstra"] = ts.dijkstra_traverse

                if not pending_strategies:
                    print(f"Skipping {model_name} dim {dim}")
                    continue

                used_config["embedding_index_min_successors"] = (
                    EMBEDDING_INDEX_MIN_SUCCESSORS
                )
                pending_work.append((dim, pending_strategies, used_config))

            if not pending_work:
                print(f"No pending dimensions for {model_name}.")
                del main_embeder
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for dim, pending_strategies, used_config in pending_work:

                print(f"--- Dim: {dim} ---")
                main_embeder.set_matryoshka_dim(dim)

                indexed_graph = None
                if not args.skip_embedding_preload:
                    print(
                        "Loading graph embedding index at Matryoshka dim "
                        f"{dim}."
                    )
                    indexed_graph = preload_graph_embeddings(
                        main_embeder,
                        causal_graph,
                        batch_size=args.embedding_batch_size,
                        save_cache=not args.no_save_embedding_cache,
                    )

                if indexed_graph is not None and main_embeder.has_embedding_index():
                    used_config["_indexed_graph"] = indexed_graph
                else:
                    used_config.pop("_indexed_graph", None)

                for strategy_name, strategy in pending_strategies.items():
                    run_warmup_traversal(
                        valid_data,
                        causal_graph,
                        main_embeder,
                        strategy,
                        strategy_name,
                        config=used_config,
                    )

                main_summary = run_evaluation_loop(
                    valid_data,
                    causal_graph,
                    main_embeder,
                    pending_strategies,
                    f"{model_path} | dim {dim} | {run_suffix}",
                    config=used_config,
                )

                save_result(
                    {
                        "model": model_name,
                        "model_path": model_path,
                        "dimension": dim,
                        "split": current_split,
                        "run_suffix": run_suffix,
                        "ablation": args.ablation,
                        "config_source_dataset": config_source_dataset_name,
                        "config_source_graph": config_source_graph_name,
                        "ablation_shared_max_visits": args.ablation,
                        "ablation_cap_reference_model": ablation_reference_model,
                        "embedding_device": args.embedding_device,
                        "used_config": strip_runtime_config(used_config),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "evaluation": main_summary,
                    },
                    output_json_file,
                    output_csv_file,
                    replace_existing=(
                        (
                            lambda entry, model_name=model_name, dim=dim: (
                                entry.get("model") == model_name
                                and entry.get("dimension") == dim
                            )
                        )
                        if args.force_model_results
                        else None
                    ),
                )

            del main_embeder
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error for {model_path}: {e}")
            evaluation_errors.append((model_path, e))

    if args.ablation and evaluation_errors:
        failed_models = [get_model_name(path) for path, _ in evaluation_errors]
        raise RuntimeError(
            "Joint four-model ablation evaluation was incomplete. Failed "
            f"models: {failed_models}"
        )

    print("\nAll evaluations complete.")
