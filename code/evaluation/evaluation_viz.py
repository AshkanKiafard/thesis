import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Paths / global config
# -------------------------------------------------------------------------

EVAL_RESULTS_PATH = "../data/evaluation/webis_binary_causal_answered/evaluation_results.json"
PLOT_OUTPUT_DIR = "../data/plots"


def build_plot_output_dir(eval_results_path):
    dataset_name = Path(eval_results_path).parent.name
    if dataset_name == "evaluation":
        dataset_name = Path(eval_results_path).stem.replace("evaluation_results_", "")
    return os.path.join(PLOT_OUTPUT_DIR, dataset_name)


# Keep only these models
BASE_MODEL_LABELS = {
    "all-mpnet-base-v2": "Base",
    "Qwen3-Embedding-0.6B": "Qwen 0.6B Base",
    "Qwen3-Embedding-4B": "Qwen 4B Base",
}


def parse_model_label(model_name):
    if model_name in BASE_MODEL_LABELS:
        return BASE_MODEL_LABELS[model_name]

    name = model_name.removesuffix("_finetuned")
    parts = name.split("_")

    stop_tokens = {"relu", "gelu", "cosine", "euclid", "norm", "nonorm", "matryoshka", "single"}

    base_parts = []
    for part in parts:
        if part in stop_tokens:
            break
        base_parts.append(part)

    base_name = "_".join(base_parts)

    activation = None
    distance = None
    normalization = None
    training = None

    for part in parts:
        if part == "relu":
            activation = "ReLU"
        elif part == "gelu":
            activation = "GELU"
        elif part == "cosine":
            distance = "Cosine"
        elif part == "euclid":
            distance = "Euclid"
        elif part == "norm":
            normalization = "Norm"
        elif part == "nonorm":
            normalization = "NoNorm"
        elif part == "matryoshka":
            training = "Matryoshka"
        elif part == "single":
            training = "Single"

    if base_name == "all-mpnet-base-v2":
        prefix = "Base"
    elif base_name == "Qwen3-Embedding-0.6B":
        prefix = "Qwen 0.6B"
    elif base_name == "Qwen3-Embedding-4B":
        prefix = "Qwen 4B"
    else:
        prefix = base_name

    label_parts = [prefix]
    if activation:
        label_parts.append(activation)
    if distance:
        label_parts.append(distance)
    if normalization:
        label_parts.append(normalization)
    if training:
        label_parts.append(training)

    return " + ".join(label_parts)


def model_sort_key(model_name):
    name = model_name.removesuffix("_finetuned")
    parts = name.split("_")

    stop_tokens = {"relu", "gelu", "cosine", "euclid", "norm", "nonorm", "matryoshka", "single"}

    base_parts = []
    for part in parts:
        if part in stop_tokens:
            break
        base_parts.append(part)

    base_name = "_".join(base_parts)

    base_order = {
        "all-mpnet-base-v2": 0,
        "Qwen3-Embedding-0.6B": 1,
        "Qwen3-Embedding-4B": 2,
    }.get(base_name, 999)

    activation_order = 0
    if "relu" in parts:
        activation_order = 1
    elif "gelu" in parts:
        activation_order = 2

    distance_order = 0
    if "cosine" in parts:
        distance_order = 1
    elif "euclid" in parts:
        distance_order = 2

    norm_order = 0
    if "nonorm" in parts:
        norm_order = 1
    elif "norm" in parts:
        norm_order = 2

    training_order = 0
    if "matryoshka" in parts:
        training_order = 1
    elif "single" in parts:
        training_order = 2

    return (base_order, activation_order, distance_order, norm_order, training_order, model_name)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_plot_path(plot_root, plot_type, filename):
    plot_dir = os.path.join(plot_root, plot_type)
    ensure_directory(plot_dir)
    return os.path.join(plot_dir, filename)


def get_bfs_baseline(json_data):
    for entry in json_data:
        if entry.get("model") == "BFS_Baseline" and "evaluation" in entry:
            bfs_data = entry["evaluation"].get("BFS", {})
            return {
                "avg_nodes_visited": bfs_data.get("avg_nodes_visited"),
                "avg_time_sec": bfs_data.get("avg_time_sec"),
                "avg_path_length": bfs_data.get("avg_path_length"),
                "accuracy": bfs_data.get("accuracy"),
                "avg_path_cost": bfs_data.get("avg_path_cost"),
                "avg_cost_per_hop": bfs_data.get("avg_cost_per_hop"),
            }
    return None


def get_rl_baseline(json_data):
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
    rows = []

    all_models = sorted(
        {
            entry.get("model")
            for entry in json_data
            if entry.get("model") not in ["BFS_Baseline", "RL_Baseline"]
        },
        key=model_sort_key,
    )

    for entry in json_data:
        if entry.get("model") in ["BFS_Baseline", "RL_Baseline"]:
            continue

        model_name = entry.get("model")
        if model_name not in all_models:
            continue

        if "dimension" not in entry or "evaluation" not in entry:
            continue

        eval_data = entry["evaluation"]
        row = {
            "model": model_name,
            "model_label": parse_model_label(model_name),
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

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        ordered_models = sorted(df["model"].unique(), key=model_sort_key)
        df["model"] = pd.Categorical(df["model"], categories=ordered_models, ordered=True)
        df = df.sort_values(["model", "dimension"]).reset_index(drop=True)

    return df


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def set_zoomed_ylim_from_models(ax, values, pad_ratio=0.08, min_pad=1e-6):
    clean = pd.Series(values).dropna()
    if clean.empty:
        return

    y_min = clean.min()
    y_max = clean.max()

    if y_min == y_max:
        pad = max(abs(y_min) * pad_ratio, min_pad)
    else:
        pad = max((y_max - y_min) * pad_ratio, min_pad)

    ax.set_ylim(y_min - pad, y_max + pad)


def get_model_subsets(df, value_col):
    subsets = []
    for model in df["model"].cat.categories:
        subset = df[df["model"] == model].sort_values(by="dimension")
        if not subset.empty and value_col in subset.columns:
            label = subset["model_label"].iloc[0]
            subsets.append((model, label, subset))
    return subsets


def plot_standard_lines(ax, df, value_col):
    all_model_values = []
    for _, label, subset in get_model_subsets(df, value_col):
        y = subset[value_col]
        ax.plot(subset["dimension"], y, marker="o", label=label)
        all_model_values.extend(y.dropna().tolist())
    return all_model_values


def apply_common_axis_style(ax, df, ylabel, title):
    ax.set_title(title)
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if not df.empty:
        ax.set_xticks(sorted(df["dimension"].unique()))


def add_baselines(ax, bfs_value=None, rl_value=None):
    if bfs_value is not None:
        ax.axhline(
            y=bfs_value,
            color="black",
            linestyle="--",
            label="BFS Baseline"
        )

    if rl_value is not None:
        ax.axhline(
            y=rl_value,
            color="red",
            linestyle="-.",
            label="RL Baseline"
        )


def add_broken_axis_marks(ax_top, ax_bottom, d=0.008):
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def plot_broken_y(
        df,
        value_col,
        output_path,
        title,
        ylabel,
        lower_ylim,
        upper_ylim,
):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    for ax in [ax_top, ax_bottom]:
        for _, label, subset in get_model_subsets(df, value_col):
            ax.plot(subset["dimension"], subset[value_col], marker="o", label=label)
        ax.grid(True)

    ax_top.set_ylim(*upper_ylim)
    ax_bottom.set_ylim(*lower_ylim)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    add_broken_axis_marks(ax_top, ax_bottom)

    ax_top.set_title(title)
    ax_bottom.set_xlabel("Embedding Size")
    ax_bottom.set_ylabel(ylabel)

    if not df.empty:
        ax_bottom.set_xticks(sorted(df["dimension"].unique()))

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# Standard metric plots
# -------------------------------------------------------------------------

def plot_nodes_visited_vs_dimension(df, bfs_baseline, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_nodes")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Nodes Visited (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            bfs_value=bfs_baseline.get("avg_nodes_visited") if bfs_baseline else None,
            rl_value=rl_baseline.get("avg_nodes_visited") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0)
        title = "Average Nodes Visited (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Nodes Visited", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_execution_time_vs_dimension(df, bfs_baseline, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_time")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Execution Time (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            bfs_value=bfs_baseline.get("avg_time_sec") if bfs_baseline else None,
            rl_value=rl_baseline.get("avg_time_sec") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0)
        title = "Average Execution Time (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Time (seconds)", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_path_length_vs_dimension(df, bfs_baseline, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_path_len")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Path Length (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            bfs_value=bfs_baseline.get("avg_path_length") if bfs_baseline else None,
            rl_value=rl_baseline.get("avg_path_length") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0)
        title = "Average Path Length (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Path Length", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_dimension(df, bfs_baseline, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_accuracy")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.15)
        title = "Accuracy (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            bfs_value=bfs_baseline.get("accuracy") if bfs_baseline else None,
            rl_value=rl_baseline.get("accuracy") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0, top=1.05)
        title = "Accuracy (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Accuracy", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_astar_path_cost_vs_dimension(df, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_path_cost")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Path Cost (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            rl_value=rl_baseline.get("avg_path_cost") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0)
        title = "Average Path Cost (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Path Cost (sum of embedding distances)", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_astar_cost_per_hop_vs_dimension(df, rl_baseline, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "astar_cost_per_hop")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Cost per Hop (A*) vs Embedding Size — Zoomed"
    else:
        add_baselines(
            ax,
            rl_value=rl_baseline.get("avg_cost_per_hop") if rl_baseline else None,
        )
        ax.set_ylim(bottom=0)
        title = "Average Cost per Hop (A*) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Cost per Hop (path_cost / hops)", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_dijkstra_path_cost_vs_dimension(df, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_model_values = plot_standard_lines(ax, df, "dijkstra_path_cost")

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = "Average Path Cost (Dijkstra) vs Embedding Size — Zoomed"
    else:
        ax.set_ylim(bottom=0)
        title = "Average Path Cost (Dijkstra) vs Embedding Size"

    apply_common_axis_style(ax, df, "Avg Path Cost (sum of embedding distances)", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_astar_vs_dijkstra_path_cost(df, model_name, output_path, zoom=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = df[df["model"] == model_name].sort_values(by="dimension")
    if subset.empty:
        print(f"No data found for model {model_name}")
        plt.close()
        return

    label = subset["model_label"].iloc[0]

    ax.plot(
        subset["dimension"],
        subset["astar_path_cost"],
        marker="o",
        label=f"{label} — A*"
    )
    ax.plot(
        subset["dimension"],
        subset["dijkstra_path_cost"],
        marker="s",
        label=f"{label} — Dijkstra"
    )

    if zoom:
        combined_values = (
                subset["astar_path_cost"].dropna().tolist()
                + subset["dijkstra_path_cost"].dropna().tolist()
        )
        set_zoomed_ylim_from_models(ax, combined_values, pad_ratio=0.12)
        title = f"A* vs Dijkstra Path Cost ({label}) — Zoomed"
    else:
        ax.set_ylim(bottom=0)
        title = f"A* vs Dijkstra Path Cost ({label})"

    apply_common_axis_style(ax, subset, "Average Path Cost", title)
    ax.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# Broken-axis versions for highly separated ranges
# -------------------------------------------------------------------------

def plot_astar_path_cost_broken(df, output_path):
    plot_broken_y(
        df=df,
        value_col="astar_path_cost",
        output_path=output_path,
        title="Average Path Cost (A*) vs Embedding Size — Broken Y-Axis",
        ylabel="Avg Path Cost",
        lower_ylim=(0.0, 0.08),
        upper_ylim=(0.35, 1.45),
    )


def plot_astar_cost_per_hop_broken(df, output_path):
    plot_broken_y(
        df=df,
        value_col="astar_cost_per_hop",
        output_path=output_path,
        title="Average Cost per Hop (A*) vs Embedding Size — Broken Y-Axis",
        ylabel="Avg Cost per Hop",
        lower_ylim=(0.0, 0.02),
        upper_ylim=(0.12, 0.60),
    )


def plot_dijkstra_path_cost_broken(df, output_path):
    plot_broken_y(
        df=df,
        value_col="dijkstra_path_cost",
        output_path=output_path,
        title="Average Path Cost (Dijkstra) vs Embedding Size — Broken Y-Axis",
        ylabel="Avg Path Cost",
        lower_ylim=(0.0, 0.02),
        upper_ylim=(0.30, 1.40),
    )


# -------------------------------------------------------------------------
# Peak / distribution investigation
# -------------------------------------------------------------------------

def plot_visited_distribution(distribution_data, output_path):
    strategies = list(distribution_data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = ["gray", "blue", "orange", "red"]

    for i, name in enumerate(strategies):
        ax = axes[i]
        data = distribution_data[name]

        clean_data = [d for d in data if d > 0]

        if not clean_data:
            ax.text(0.5, 0.5, f"No data for {name}", ha="center", va="center")
            ax.set_title(f"Node Visited Distribution: {name}")
            continue

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
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"\nDistribution plots saved to {output_path}")


if __name__ == "__main__":
    plot_root = build_plot_output_dir(EVAL_RESULTS_PATH)
    ensure_directory(plot_root)

    with open(EVAL_RESULTS_PATH) as f:
        eval_data = json.load(f)

    df = extract_semantic_data(eval_data)
    bfs_data = get_bfs_baseline(eval_data)
    rl_data = get_rl_baseline(eval_data)

    # Standard + zoomed plots
    plot_nodes_visited_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "nodes_visited", "metric_nodes_visited.png"),
        zoom=False
    )
    plot_nodes_visited_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "nodes_visited", "metric_nodes_visited_zoom.png"),
        zoom=True
    )

    plot_execution_time_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "execution_time", "metric_time.png"),
        zoom=False
    )
    plot_execution_time_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "execution_time", "metric_time_zoom.png"),
        zoom=True
    )

    plot_path_length_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "path_length", "metric_path_length.png"),
        zoom=False
    )
    plot_path_length_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "path_length", "metric_path_length_zoom.png"),
        zoom=True
    )

    plot_accuracy_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "accuracy", "metric_accuracy.png"),
        zoom=False
    )
    plot_accuracy_vs_dimension(
        df, bfs_data, rl_data,
        get_plot_path(plot_root, "accuracy", "metric_accuracy_zoom.png"),
        zoom=True
    )

    plot_astar_path_cost_vs_dimension(
        df, rl_data,
        get_plot_path(plot_root, "astar_path_cost", "metric_astar_path_cost.png"),
        zoom=False
    )
    plot_astar_path_cost_vs_dimension(
        df, rl_data,
        get_plot_path(plot_root, "astar_path_cost", "metric_astar_path_cost_zoom.png"),
        zoom=True
    )
    plot_astar_path_cost_broken(
        df,
        get_plot_path(plot_root, "astar_path_cost", "metric_astar_path_cost_broken.png")
    )

    plot_astar_cost_per_hop_vs_dimension(
        df, rl_data,
        get_plot_path(plot_root, "astar_cost_per_hop", "metric_astar_cost_per_hop.png"),
        zoom=False
    )
    plot_astar_cost_per_hop_vs_dimension(
        df, rl_data,
        get_plot_path(plot_root, "astar_cost_per_hop", "metric_astar_cost_per_hop_zoom.png"),
        zoom=True
    )
    plot_astar_cost_per_hop_broken(
        df,
        get_plot_path(plot_root, "astar_cost_per_hop", "metric_astar_cost_per_hop_broken.png")
    )

    plot_dijkstra_path_cost_vs_dimension(
        df,
        get_plot_path(plot_root, "dijkstra_path_cost", "metric_dijkstra_path_cost.png"),
        zoom=False
    )
    plot_dijkstra_path_cost_vs_dimension(
        df,
        get_plot_path(plot_root, "dijkstra_path_cost", "metric_dijkstra_path_cost_zoom.png"),
        zoom=True
    )
    plot_dijkstra_path_cost_broken(
        df,
        get_plot_path(plot_root, "dijkstra_path_cost", "metric_dijkstra_path_cost_broken.png")
    )

    candidate_models = list(df["model"].cat.categories) if not df.empty else []

    if candidate_models:
        compare_model = next(
            (m for m in candidate_models if "relu" in m and "cosine" in m),
            candidate_models[0]
        )

        plot_astar_vs_dijkstra_path_cost(
            df,
            compare_model,
            get_plot_path(plot_root, "compare_astar_dijkstra_path_cost", "compare_astar_dijkstra_path_cost.png"),
            zoom=False
        )
        plot_astar_vs_dijkstra_path_cost(
            df,
            compare_model,
            get_plot_path(plot_root, "compare_astar_dijkstra_path_cost", "compare_astar_dijkstra_path_cost_zoom.png"),
            zoom=True
        )
