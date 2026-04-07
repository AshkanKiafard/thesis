import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Paths / global config
# -------------------------------------------------------------------------

GRAPH_PATH = "data/graphs/causenet-precision.jsonl"
EVAL_RESULTS_PATH = "data/evaluation/evaluation_results_valid.json"
VALID_DATA_PATH = "data/datasets/msmarco_train.json"
PLOT_OUTPUT_DIR = "data/plots"


def ensure_directory(path):
    # Create output directory if it does not exist yet.
    if not os.path.exists(path):
        os.makedirs(path)


def get_bfs_baseline(json_data):
    # Extract the BFS baseline entry from the evaluation JSON.
    for entry in json_data:
        if entry.get("model") == "BFS_Baseline" and "evaluation" in entry:
            bfs_data = entry["evaluation"].get("BFS", {})
            return {
                "avg_nodes_visited": bfs_data.get("avg_nodes_visited"),
                "avg_time_sec": bfs_data.get("avg_time_sec"),
                "avg_path_length": bfs_data.get("avg_path_length"),
                "accuracy": bfs_data.get("accuracy"),
                # BFS baseline usually has no meaningful embedding-based costs.
                "avg_path_cost": bfs_data.get("avg_path_cost"),
                "avg_cost_per_hop": bfs_data.get("avg_cost_per_hop"),
            }
    return None


def get_rl_baseline(json_data):
    # Extract the RL baseline entry from the evaluation JSON.
    for entry in json_data:
        if entry.get("model") == "RL_Baseline" and "evaluation" in entry:
            rl_data = entry["evaluation"].get("RL", {})
            return {
                "avg_nodes_visited": rl_data.get("avg_nodes_visited"),
                "avg_time_sec": rl_data.get("avg_time_sec"),
                "avg_path_length": rl_data.get("avg_path_length"),
                "accuracy": rl_data.get("accuracy"),
                "avg_path_cost": rl_data.get("avg_path_cost"),
                "avg_cost_per_hop": rl_data.get("avg_cost_per_hop"),
            }
    return None


def extract_semantic_data(json_data):
    """
    Extract both A* and Dijkstra metrics for each (model, dimension).

    Output columns include:
      - astar_*
      - dijkstra_*
    """
    rows = []

    for entry in json_data:
        # Skip baselines because they do not have per-dimension semantic results.
        if entry.get("model") in ["BFS_Baseline", "RL_Baseline"]:
            continue

        if "dimension" not in entry or "evaluation" not in entry:
            continue

        eval_data = entry["evaluation"]
        row = {
            "model": entry["model"],
            "dimension": int(entry["dimension"]),
        }

        if "A*" in eval_data:
            row.update({
                "astar_nodes": eval_data["A*"].get("avg_nodes_visited"),
                "astar_path_len": eval_data["A*"].get("avg_path_length"),
                "astar_time": eval_data["A*"].get("avg_time_sec"),
                "astar_accuracy": eval_data["A*"].get("accuracy"),
                "astar_path_cost": eval_data["A*"].get("avg_path_cost"),
                "astar_cost_per_hop": eval_data["A*"].get("avg_cost_per_hop"),
            })

        if "Dijkstra" in eval_data:
            row.update({
                "dijkstra_nodes": eval_data["Dijkstra"].get("avg_nodes_visited"),
                "dijkstra_path_len": eval_data["Dijkstra"].get("avg_path_length"),
                "dijkstra_time": eval_data["Dijkstra"].get("avg_time_sec"),
                "dijkstra_accuracy": eval_data["Dijkstra"].get("accuracy"),
                "dijkstra_path_cost": eval_data["Dijkstra"].get("avg_path_cost"),
                "dijkstra_cost_per_hop": eval_data["Dijkstra"].get("avg_cost_per_hop"),
            })

        # Keep only rows where at least one semantic strategy exists.
        if any(k in row for k in ["astar_nodes", "dijkstra_nodes"]):
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Standard metric plots
# -------------------------------------------------------------------------

def plot_nodes_visited_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_nodes"], marker="o", label=model)

    if bfs_baseline and bfs_baseline.get("avg_nodes_visited") is not None:
        plt.axhline(
            y=bfs_baseline["avg_nodes_visited"],
            color="black",
            linestyle="--",
            label="BFS Baseline"
        )

    if rl_baseline and rl_baseline.get("avg_nodes_visited") is not None:
        plt.axhline(
            y=rl_baseline["avg_nodes_visited"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Average Nodes Visited (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Nodes Visited")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_execution_time_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_time"], marker="o", label=model)

    if bfs_baseline and bfs_baseline.get("avg_time_sec") is not None:
        plt.axhline(
            y=bfs_baseline["avg_time_sec"],
            color="black",
            linestyle="--",
            label="BFS Baseline"
        )

    if rl_baseline and rl_baseline.get("avg_time_sec") is not None:
        plt.axhline(
            y=rl_baseline["avg_time_sec"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Average Execution Time (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_path_length_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_path_len"], marker="o", label=model)

    if bfs_baseline and bfs_baseline.get("avg_path_length") is not None:
        plt.axhline(
            y=bfs_baseline["avg_path_length"],
            color="black",
            linestyle="--",
            label="BFS Baseline"
        )

    if rl_baseline and rl_baseline.get("avg_path_length") is not None:
        plt.axhline(
            y=rl_baseline["avg_path_length"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Average Path Length (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Path Length")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_accuracy_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_accuracy"], marker="o", label=model)

    if bfs_baseline and bfs_baseline.get("accuracy") is not None:
        plt.axhline(
            y=bfs_baseline["accuracy"],
            color="black",
            linestyle="--",
            label="BFS Baseline"
        )

    if rl_baseline and rl_baseline.get("accuracy") is not None:
        plt.axhline(
            y=rl_baseline["accuracy"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Accuracy (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0, top=1.05)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_astar_path_cost_vs_dimension(df, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_path_cost"], marker="o", label=model)

    if rl_baseline and rl_baseline.get("avg_path_cost") is not None:
        plt.axhline(
            y=rl_baseline["avg_path_cost"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Average Path Cost (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Path Cost (sum of embedding distances)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_astar_cost_per_hop_vs_dimension(df, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["astar_cost_per_hop"], marker="o", label=model)

    if rl_baseline and rl_baseline.get("avg_cost_per_hop") is not None:
        plt.axhline(
            y=rl_baseline["avg_cost_per_hop"],
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )

    plt.title("Average Cost per Hop (A*) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Cost per Hop (path_cost / hops)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_dijkstra_path_cost_vs_dimension(df, output_path):
    plt.figure(figsize=(10, 6))

    models = df["model"].unique()
    for model in models:
        subset = df[df["model"] == model].sort_values(by="dimension")
        plt.plot(subset["dimension"], subset["dijkstra_path_cost"], marker="o", label=model)

    plt.title("Average Path Cost (Dijkstra) vs Embedding Size")
    plt.xlabel("Embedding Size")
    plt.ylabel("Avg Path Cost (sum of embedding distances)")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_astar_vs_dijkstra_path_cost(df, model_name, output_path):
    plt.figure(figsize=(10, 6))

    subset = df[df["model"] == model_name].sort_values(by="dimension")

    if subset.empty:
        print(f"No data found for model {model_name}")
        return

    plt.plot(
        subset["dimension"],
        subset["astar_path_cost"],
        marker="o",
        label="A* Path Cost"
    )

    plt.plot(
        subset["dimension"],
        subset["dijkstra_path_cost"],
        marker="s",
        label="Dijkstra Path Cost"
    )

    plt.title(f"A* vs Dijkstra Path Cost ({model_name})")
    plt.xlabel("Embedding Size")
    plt.ylabel("Average Path Cost")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    plt.xticks(sorted(subset["dimension"].unique()))

    plt.savefig(output_path)
    plt.close()


# -------------------------------------------------------------------------
# Peak / distribution investigation
# -------------------------------------------------------------------------

def plot_visited_distribution(distribution_data, output_path):
    """
    Plot log-scale histograms of visited-node counts for each strategy
    and mark the 95th percentile.
    """
    strategies = list(distribution_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = ["gray", "blue", "orange", "red"]

    for i, name in enumerate(strategies):
        ax = axes[i]
        data = distribution_data[name]

        # Ignore zero entries to avoid broken log bins.
        clean_data = [d for d in data if d > 0]

        if not clean_data:
            ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
            ax.set_title(f"Node Visited Distribution: {name}")
            continue

        # Use logarithmic binning because visited counts can vary heavily.
        bins = np.logspace(np.log10(min(clean_data)), np.log10(max(clean_data)), 30)

        ax.hist(
            clean_data,
            bins=bins,
            color=colors[i],
            edgecolor="black",
            alpha=0.7
        )

        ax.set_xscale("log")
        ax.set_title(f"Node Visited Distribution: {name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Nodes Visited (Log Scale)")
        ax.set_ylabel("Frequency")
        ax.grid(True, which="both", ls="-", alpha=0.2)

        p95_val = np.percentile(clean_data, 95)
        ax.axvline(
            p95_val,
            color="purple",
            linestyle="dashed",
            linewidth=2,
            label=f"95th Pctl: {p95_val:.1f}"
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nDistribution plots saved to {output_path}")


if __name__ == "__main__":
    ensure_directory(PLOT_OUTPUT_DIR)

    # ---------------------------------------------------------------------
    # Load evaluation summaries used for the standard plots
    # ---------------------------------------------------------------------
    with open(EVAL_RESULTS_PATH) as f:
        eval_data = json.load(f)

    df = extract_semantic_data(eval_data)
    bfs_data = get_bfs_baseline(eval_data)
    rl_data = get_rl_baseline(eval_data)

    plot_nodes_visited_vs_dimension(
        df,
        bfs_data,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_nodes_visited.png")
    )

    plot_execution_time_vs_dimension(
        df,
        bfs_data,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_time.png")
    )

    plot_path_length_vs_dimension(
        df,
        bfs_data,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_path_length.png")
    )

    plot_accuracy_vs_dimension(
        df,
        bfs_data,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_accuracy.png")
    )

    plot_astar_path_cost_vs_dimension(
        df,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_astar_path_cost.png")
    )

    plot_astar_cost_per_hop_vs_dimension(
        df,
        rl_data,
        os.path.join(PLOT_OUTPUT_DIR, "metric_astar_cost_per_hop.png")
    )

    plot_dijkstra_path_cost_vs_dimension(
        df,
        os.path.join(PLOT_OUTPUT_DIR, "metric_dijkstra_path_cost.png")
    )

    plot_astar_vs_dijkstra_path_cost(
        df,
        "all-mpnet-base-v2_relu_cosine_v2_finetuned",
        os.path.join(PLOT_OUTPUT_DIR, "compare_astar_dijkstra_path_cost.png")
    )
