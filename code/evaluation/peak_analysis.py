import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import traverse_strategies as ts
from embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from utils import get_concept, load_graph, traverse_graph

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

# Causal graph used for traversal
GRAPH_PATH = "../data/graphs/causenet-precision.jsonl"

# Dataset used for analysis (train/valid depending on what you want to inspect)
DATASET_PATH = "../data/datasets/msmarco_valid.json"
TRAIN_OR_VALID = "Train" if "train" in DATASET_PATH else "Valid"

# Where plots will be stored
PLOT_OUTPUT_DIR = "../data/plots"


def ensure_directory(path):
    # Utility: create output directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)


def plot_visited_distribution(distribution_data):
    """
    Creates log-scale histograms of visited node counts for each strategy.
    Also highlights the 95th percentile to show worst-case behavior.
    """
    ensure_directory(PLOT_OUTPUT_DIR)

    strategies = list(distribution_data.keys())

    # 2x2 subplot grid (BFS, A*, Dijkstra, RL)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = ['gray', 'blue', 'orange', 'red']

    for i, name in enumerate(strategies):
        data = distribution_data[name]
        ax = axes[i]

        # Remove zeros (log-scale can't handle them)
        clean_data = [d for d in data if d > 0]

        if not clean_data:
            ax.text(0.5, 0.5, f"No data for {name}", ha='center')
            continue

        # Logarithmic bins to handle large variance in visited nodes
        bins = np.logspace(
            np.log10(min(clean_data)),
            np.log10(max(clean_data)),
            30
        )

        ax.hist(
            clean_data,
            bins=bins,
            color=colors[i],
            edgecolor='black',
            alpha=0.7
        )

        ax.set_xscale('log')

        ax.set_title(
            f'Node Visited Distribution: {name} _ {TRAIN_OR_VALID}',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Nodes Visited (Log Scale)')
        ax.set_ylabel('Frequency')
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Highlight 95th percentile (important for thesis discussion)
        p95_val = np.percentile(clean_data, 95)
        ax.axvline(
            p95_val,
            color='purple',
            linestyle='dashed',
            linewidth=2,
            label=f'95th Pctl: {p95_val:.1f}'
        )
        ax.legend()

    plt.tight_layout()

    output_path = os.path.join(
        PLOT_OUTPUT_DIR,
        f"{TRAIN_OR_VALID}_visited_nodes_distribution_log_p95.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(f"\nDistribution plots (Log Scale + P95) saved to {output_path}")


def run_peak_investigation():
    """
    Re-runs all traversal strategies on the dataset and collects
    how many nodes each algorithm visits per query.

    This is used to analyze:
    - worst-case behavior
    - heavy tails
    - search efficiency differences
    """
    print("Loading data...")
    with open(DATASET_PATH) as f:
        json_data = json.load(f)

    print("Loading graph...")
    graph = load_graph(GRAPH_PATH)

    print("Initializing Embedders...")

    # SentenceTransformer for semantic search (A*, Dijkstra)
    st_embeder = STEmbedder(
        '../data/models/lightning/all-mpnet-base-v2_relu_cosine_matryoshka_finetuned',
        DistanceMetric.COSINE
    )

    # GloVe for RL baseline (required input format)
    glove_embeder = GloveEmbeder(
        '../data/embeddings/glove.6B/glove.6B.300d.txt',
        DistanceMetric.COSINE
    )

    # RL-specific config
    rl_config = {
        'rl_model_path': "../data/models/rl/msmarco_evaluation_state_dict.pt",
        'rl_beam_width': 5,
        'rl_max_path_len': -1
    }

    # Store visited node counts per strategy
    all_visited_counts = {
        "BFS": [],
        "A*": [],
        "Dijkstra": [],
        "RL": []
    }

    # Define traversal strategies
    strategies = [
        ("BFS", ts.bfs_traverse, None),
        ("A*", ts.astar_traverse, st_embeder),
        ("Dijkstra", ts.dijkstra_traverse, st_embeder),
        ("RL", ts.rl_traverse, glove_embeder)
    ]

    for name, strategy_fn, embeder in strategies:
        print(f"\nEvaluating visited nodes for: {name}")

        for item in tqdm(json_data):
            cause = get_concept(item, 0)
            effect = get_concept(item, 1)

            # RL needs extra config, others don't
            config = rl_config if name == "RL" else None

            path, visited_count = traverse_graph(
                graph,
                cause,
                effect,
                embeder,
                strategy_fn,
                config
            )

            # Only count successful paths (same logic as evaluation)
            if bool(path):
                all_visited_counts[name].append(visited_count)

    # Print worst-case values
    print("\n" + "=" * 40)
    print("ABSOLUTE MAX VISITED NODES PER QUERY")
    print("=" * 40)

    for name, data in all_visited_counts.items():
        peak = max(data) if data else 0
        print(f"{name:<10}: {peak:,} nodes")

    print("=" * 40)

    # Generate histogram plots
    plot_visited_distribution(all_visited_counts)


if __name__ == "__main__":
    run_peak_investigation()
