import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Paths / global config
# -------------------------------------------------------------------------

PLOT_OUTPUT_DIR = "data/plots"

BASE_MODEL_LABELS = {
    "all-mpnet-base-v2": "MPNet Base",
    "bge-large-en-v1.5": "BGE Large Base",
    "mxbai-embed-large-v1": "MxBai Large Base",
    "Qwen3-Embedding-0.6B": "Qwen 0.6B Base",
    "Qwen3-Embedding-4B": "Qwen 4B Base",
    "BFS_Baseline": "BFS Baseline",
    "RL_Baseline": "RL Baseline",
}


# -------------------------------------------------------------------------
# CLI / path helpers
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results and visited-node p95 analysis."
    )
    parser.add_argument(
        "dataset",
        help=(
            "Dataset name or dataset path, e.g. "
            "msmarco_valid or data/datasets/filtered/msmarco_valid_filtered.json"
        ),
    )
    return parser.parse_args()


def dataset_name_from_arg(dataset_arg: str):
    path = Path(dataset_arg)

    if path.suffix == ".json":
        return path.stem.replace("_filtered", "")

    return dataset_arg.replace("_filtered", "")


def detect_split(dataset_name: str):
    name = dataset_name.lower()

    if "train" in name:
        return "train"
    if "valid" in name:
        return "valid"
    if "test" in name:
        return "test"

    return "unknown"


def get_p95_source_dataset_name(dataset_name: str):
    """
    Match evaluation.py behavior:

    valid evaluation/plots -> train p95
    test evaluation/plots  -> valid p95
    train                  -> own p95 if available
    """
    split = detect_split(dataset_name)

    if split == "valid":
        return dataset_name.replace("valid", "train")
    if split == "test":
        return dataset_name.replace("test", "valid")

    return dataset_name


def build_input_paths(dataset_name: str):
    eval_dir = Path("data/evaluation") / dataset_name
    eval_results_path = eval_dir / "evaluation_results.json"

    p95_source_dataset = get_p95_source_dataset_name(dataset_name)
    visited_nodes_path = (
        Path("data/evaluation")
        / p95_source_dataset
        / "visited_nodes_analysis.json"
    )

    return eval_results_path, visited_nodes_path, p95_source_dataset


def build_plot_output_dir(dataset_name: str):
    return Path(PLOT_OUTPUT_DIR) / dataset_name


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_plot_path(plot_root, plot_type, filename):
    plot_dir = Path(plot_root) / plot_type
    ensure_directory(plot_dir)
    return plot_dir / filename


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# -------------------------------------------------------------------------
# Model labels / sorting
# -------------------------------------------------------------------------

def parse_model_label(model_name):
    if model_name in BASE_MODEL_LABELS:
        return BASE_MODEL_LABELS[model_name]

    name = model_name.removesuffix("_finetuned")
    name = name.removesuffix("_best")
    parts = name.split("_")

    stop_tokens = {
        "relu",
        "gelu",
        "cosine",
        "euclid",
        "norm",
        "nonorm",
        "matryoshka",
        "single",
        "best",
    }

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
            distance = "Euclidean"
        elif part == "norm":
            normalization = "Norm"
        elif part == "nonorm":
            normalization = "NoNorm"
        elif part == "matryoshka":
            training = "Matryoshka"
        elif part == "single":
            training = "Single"

    if base_name == "all-mpnet-base-v2":
        prefix = "MPNet"
    elif base_name == "bge-large-en-v1.5":
        prefix = "BGE Large"
    elif base_name == "mxbai-embed-large-v1":
        prefix = "MxBai Large"
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
    name = model_name.removesuffix("_finetuned").removesuffix("_best")
    parts = name.split("_")

    stop_tokens = {
        "relu",
        "gelu",
        "cosine",
        "euclid",
        "norm",
        "nonorm",
        "matryoshka",
        "single",
        "best",
    }

    base_parts = []
    for part in parts:
        if part in stop_tokens:
            break
        base_parts.append(part)

    base_name = "_".join(base_parts)

    base_order = {
        "BFS_Baseline": -2,
        "RL_Baseline": -1,
        "all-mpnet-base-v2": 0,
        "bge-large-en-v1.5": 1,
        "mxbai-embed-large-v1": 2,
        "Qwen3-Embedding-0.6B": 3,
        "Qwen3-Embedding-4B": 4,
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

    return (
        base_order,
        activation_order,
        distance_order,
        norm_order,
        training_order,
        model_name,
    )


# -------------------------------------------------------------------------
# Evaluation extraction
# -------------------------------------------------------------------------

def extract_baselines(eval_data):
    baselines = {}

    for entry in eval_data:
        model = entry.get("model")

        if model not in {"BFS_Baseline", "RL_Baseline"}:
            continue

        evaluation = entry.get("evaluation", {})

        for algorithm, result in evaluation.items():
            metrics = result.get("metrics", {})

            baselines[model] = {
                "algorithm": algorithm,
                "model_label": parse_model_label(model),
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "avg_nodes_visited": metrics.get("avg_nodes_visited"),
                "avg_time_sec": metrics.get("avg_time_sec"),
                "avg_path_length": metrics.get("avg_path_length"),
                "avg_path_cost": metrics.get("avg_path_cost"),
                "avg_cost_per_hop": metrics.get("avg_cost_per_hop"),
                "used_config": entry.get("used_config"),
                "config_source_dataset": entry.get("config_source_dataset"),
            }

    return baselines


def extract_semantic_data(eval_data):
    rows = []

    for entry in eval_data:
        model_name = entry.get("model")

        if model_name in {"BFS_Baseline", "RL_Baseline"}:
            continue

        if "dimension" not in entry or "evaluation" not in entry:
            continue

        dimension = entry.get("dimension")

        for algorithm, result in entry["evaluation"].items():
            if algorithm != "A*":
                continue

            metrics = result.get("metrics", {})

            rows.append(
                {
                    "model": model_name,
                    "model_label": parse_model_label(model_name),
                    "dimension": int(dimension),
                    "algorithm": algorithm,
                    "accuracy": metrics.get("accuracy"),
                    "f1_score": metrics.get("f1_score"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "avg_nodes_visited": metrics.get("avg_nodes_visited"),
                    "avg_time_sec": metrics.get("avg_time_sec"),
                    "avg_path_length": metrics.get("avg_path_length"),
                    "avg_path_cost": metrics.get("avg_path_cost"),
                    "avg_cost_per_hop": metrics.get("avg_cost_per_hop"),
                    "used_config": entry.get("used_config"),
                    "config_source_dataset": entry.get("config_source_dataset"),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        ordered_models = sorted(df["model"].unique(), key=model_sort_key)
        df["model"] = pd.Categorical(
            df["model"],
            categories=ordered_models,
            ordered=True,
        )
        df = df.sort_values(["model", "dimension"]).reset_index(drop=True)

    return df


def extract_per_example_rows(eval_data):
    rows = []

    for entry in eval_data:
        model = entry.get("model")
        dimension = entry.get("dimension")

        for algorithm, result in entry.get("evaluation", {}).items():
            for row in result.get("per_example", []):
                rows.append(
                    {
                        "model": model,
                        "model_label": parse_model_label(model),
                        "dimension": dimension,
                        "algorithm": algorithm,
                        "id": row.get("id"),
                        "true": row.get("true"),
                        "pred": row.get("pred"),
                        "correct": row.get("correct"),
                        "nodes_visited": row.get("nodes_visited"),
                        "path_length": row.get("path_length"),
                        "time_sec": row.get("time_sec"),
                        "path_cost": row.get("path_cost"),
                    }
                )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# General plotting helpers
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

    if df.empty or value_col not in df.columns:
        return subsets

    for model in df["model"].cat.categories:
        subset = df[df["model"] == model].sort_values(by="dimension")

        if not subset.empty and value_col in subset.columns:
            label = subset["model_label"].iloc[0]
            subsets.append((model, label, subset))

    return subsets


def plot_standard_lines(ax, df, value_col):
    all_values = []

    for _, label, subset in get_model_subsets(df, value_col):
        y = subset[value_col]
        ax.plot(subset["dimension"], y, marker="o", label=label)
        all_values.extend(y.dropna().tolist())

    return all_values


def apply_common_axis_style(ax, df, ylabel, title):
    ax.set_title(title)
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if not df.empty:
        ax.set_xticks(sorted(df["dimension"].unique()))


def add_baseline_lines(ax, baselines, metric_key):
    bfs = baselines.get("BFS_Baseline")
    rl = baselines.get("RL_Baseline")

    if bfs and bfs.get(metric_key) is not None:
        ax.axhline(
            y=bfs[metric_key],
            color="black",
            linestyle="--",
            label="BFS Baseline",
        )

    if rl and rl.get(metric_key) is not None:
        ax.axhline(
            y=rl[metric_key],
            color="red",
            linestyle="-.",
            label="RL Baseline",
        )


def plot_metric_vs_dimension(
    df,
    baselines,
    metric_key,
    ylabel,
    title,
    output_path,
    zoom=False,
    y_min=None,
    y_max=None,
):
    if df.empty:
        print(f"No semantic data. Skipping {title}")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    all_model_values = plot_standard_lines(ax, df, metric_key)

    if zoom:
        set_zoomed_ylim_from_models(ax, all_model_values, pad_ratio=0.12)
        title = f"{title} — Zoomed"
    else:
        add_baseline_lines(ax, baselines, metric_key)

        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        else:
            ax.set_ylim(bottom=0)

    apply_common_axis_style(ax, df, ylabel, title)

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


# -------------------------------------------------------------------------
# P95 / visited-node analysis extraction and plots
# -------------------------------------------------------------------------

def extract_p95_data(visited_nodes_data):
    rows = []

    for entry in visited_nodes_data:
        model = entry.get("model")
        dimension = entry.get("dimension")
        analysis = entry.get("analysis", {})

        if model is None:
            continue

        if model in {"BFS_Baseline", "RL_Baseline"}:
            algorithm = analysis.get("strategy", model.replace("_Baseline", ""))
            model_label = parse_model_label(model)
            model_id = model
        else:
            algorithm = analysis.get("strategy", "A*")
            model_label = parse_model_label(model)
            model_id = f"{model}__dim{dimension}"

        rows.append(
            {
                "model": model,
                "model_id": model_id,
                "model_label": model_label,
                "dimension": dimension,
                "algorithm": algorithm,
                "split": entry.get("split"),
                "dataset": entry.get("dataset"),
                "num_examples": analysis.get("num_examples"),
                "num_successful_paths": analysis.get("num_successful_paths"),
                "max_visited_all": analysis.get("max_visited_all"),
                "max_visited_successful_only": analysis.get("max_visited_successful_only"),
                "p95_visited_all": analysis.get("p95_visited_all"),
                "p95_visited_successful_only": analysis.get("p95_visited_successful_only"),
                "visited_counts_all": analysis.get("visited_counts_all", []),
                "visited_counts_successful_only": analysis.get(
                    "visited_counts_successful_only",
                    [],
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["sort_key"] = df["model"].map(model_sort_key)
        df = df.sort_values(["sort_key", "dimension"], na_position="first")
        df = df.drop(columns=["sort_key"]).reset_index(drop=True)

    return df


def plot_p95_bar_chart(p95_df, output_path, successful_only=True):
    if p95_df.empty:
        print("No p95 data found. Skipping p95 bar chart.")
        return

    value_col = (
        "p95_visited_successful_only"
        if successful_only
        else "p95_visited_all"
    )

    title_suffix = "Successful Paths Only" if successful_only else "All Searches"

    df = p95_df.copy()
    df = df[df[value_col].notna()]

    labels = []

    for _, row in df.iterrows():
        label = row["model_label"]

        if pd.notna(row["dimension"]):
            label = f"{label}\nDim {int(row['dimension'])}"

        labels.append(label)

    values = df[value_col].astype(float).tolist()

    fig_width = max(12, len(values) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(range(len(values)), values)

    ax.set_title(f"P95 Visited Nodes — {title_suffix}")
    ax.set_ylabel("P95 Visited Nodes")
    ax.set_xlabel("Model / Baseline")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved p95 bar chart: {output_path}")


def plot_p95_vs_dimension(p95_df, output_path, successful_only=True):
    df = p95_df.copy()
    df = df[~df["model"].isin(["BFS_Baseline", "RL_Baseline"])]
    df = df[df["dimension"].notna()]

    if df.empty:
        print("No semantic p95 dimension data found. Skipping p95-vs-dimension plot.")
        return

    value_col = (
        "p95_visited_successful_only"
        if successful_only
        else "p95_visited_all"
    )

    title_suffix = "Successful Paths Only" if successful_only else "All Searches"

    fig, ax = plt.subplots(figsize=(11, 6))

    for model in sorted(df["model"].unique(), key=model_sort_key):
        subset = df[df["model"] == model].sort_values("dimension")
        label = subset["model_label"].iloc[0]

        ax.plot(
            subset["dimension"].astype(int),
            subset[value_col],
            marker="o",
            label=label,
        )

    ax.set_title(f"P95 Visited Nodes vs Embedding Size — {title_suffix}")
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel("P95 Visited Nodes")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved p95-vs-dimension plot: {output_path}")


def plot_visited_distributions_grid(
    p95_df,
    output_path,
    successful_only=True,
):
    if p95_df.empty:
        print("No p95 distribution data found. Skipping distribution grid.")
        return

    value_col = (
        "visited_counts_successful_only"
        if successful_only
        else "visited_counts_all"
    )

    p95_col = (
        "p95_visited_successful_only"
        if successful_only
        else "p95_visited_all"
    )

    title_suffix = "Successful Paths Only" if successful_only else "All Searches"

    rows = []

    for _, row in p95_df.iterrows():
        counts = row.get(value_col, [])

        if not isinstance(counts, list) or len(counts) == 0:
            continue

        clean_counts = [v for v in counts if v > 0]

        if not clean_counts:
            continue

        rows.append((row, clean_counts))

    if not rows:
        print("No valid visited-count distributions found. Skipping grid.")
        return

    n = len(rows)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
    )

    axes = np.array(axes).flatten()

    for ax, (row, counts) in zip(axes, rows):
        min_count = min(counts)
        max_count = max(counts)

        if min_count == max_count:
            bins = 10
        else:
            bins = np.logspace(
                np.log10(min_count),
                np.log10(max_count),
                30,
            )

        ax.hist(
            counts,
            bins=bins,
            edgecolor="black",
            alpha=0.75,
        )

        if min_count != max_count:
            ax.set_xscale("log")

        label = row["model_label"]

        if pd.notna(row["dimension"]):
            label = f"{label} | dim {int(row['dimension'])}"

        p95_value = row.get(p95_col)

        if p95_value is not None and not pd.isna(p95_value):
            ax.axvline(
                p95_value,
                linestyle="--",
                linewidth=2,
                label=f"P95: {p95_value:.1f}",
            )

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Visited Nodes")
        ax.set_ylabel("Frequency")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle(
        f"Visited Node Distributions — {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved visited distribution grid: {output_path}")


# -------------------------------------------------------------------------
# Per-example histogram plots from evaluation_results.json
# -------------------------------------------------------------------------

def plot_per_example_histograms(per_example_df, value_col, output_path, log_x=False):
    if per_example_df.empty or value_col not in per_example_df.columns:
        print(f"No data for {value_col}. Skipping.")
        return

    models = sorted(per_example_df["model"].unique(), key=model_sort_key)

    n = len(models)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
    )

    axes = np.array(axes).flatten()

    for ax, model in zip(axes, models):
        subset = per_example_df[per_example_df["model"] == model]
        values = subset[value_col].dropna()

        if values.empty:
            ax.axis("off")
            continue

        values = values[values > 0] if log_x else values

        if values.empty:
            ax.axis("off")
            continue

        ax.hist(values, bins=30, edgecolor="black", alpha=0.75)

        if log_x:
            ax.set_xscale("log")

        ax.set_title(parse_model_label(model))
        ax.set_xlabel(value_col)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(models):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram: {output_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    dataset_name = dataset_name_from_arg(args.dataset)

    eval_results_path, visited_nodes_path, p95_source_dataset = build_input_paths(dataset_name)
    plot_root = build_plot_output_dir(dataset_name)

    ensure_directory(plot_root)

    print(f"Dataset: {dataset_name}")
    print(f"Evaluation results: {eval_results_path}")
    print(f"P95 source dataset: {p95_source_dataset}")
    print(f"Visited nodes analysis: {visited_nodes_path}")
    print(f"Plot output dir: {plot_root}")

    eval_data = load_json(eval_results_path)

    df = extract_semantic_data(eval_data)
    baselines = extract_baselines(eval_data)
    per_example_df = extract_per_example_rows(eval_data)

    # -------------------------------------------------------------------------
    # Standard evaluation metric plots
    # -------------------------------------------------------------------------

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_nodes_visited",
        ylabel="Average Visited Nodes",
        title="Average Visited Nodes (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "nodes_visited", "metric_nodes_visited.png"),
        zoom=False,
        y_min=0,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_nodes_visited",
        ylabel="Average Visited Nodes",
        title="Average Visited Nodes (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "nodes_visited", "metric_nodes_visited_zoom.png"),
        zoom=True,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_time_sec",
        ylabel="Average Time (seconds)",
        title="Average Runtime (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "execution_time", "metric_time.png"),
        zoom=False,
        y_min=0,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_time_sec",
        ylabel="Average Time (seconds)",
        title="Average Runtime (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "execution_time", "metric_time_zoom.png"),
        zoom=True,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_path_length",
        ylabel="Average Path Length",
        title="Average Path Length (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "path_length", "metric_path_length.png"),
        zoom=False,
        y_min=0,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_path_length",
        ylabel="Average Path Length",
        title="Average Path Length (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "path_length", "metric_path_length_zoom.png"),
        zoom=True,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="accuracy",
        ylabel="Accuracy",
        title="Accuracy (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "accuracy", "metric_accuracy.png"),
        zoom=False,
        y_min=0,
        y_max=1.05,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="accuracy",
        ylabel="Accuracy",
        title="Accuracy (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "accuracy", "metric_accuracy_zoom.png"),
        zoom=True,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="f1_score",
        ylabel="F1 Score",
        title="F1 Score (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "f1_score", "metric_f1_score.png"),
        zoom=False,
        y_min=0,
        y_max=1.05,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="f1_score",
        ylabel="F1 Score",
        title="F1 Score (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "f1_score", "metric_f1_score_zoom.png"),
        zoom=True,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="precision",
        ylabel="Precision",
        title="Precision (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "precision", "metric_precision.png"),
        zoom=False,
        y_min=0,
        y_max=1.05,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="recall",
        ylabel="Recall",
        title="Recall (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "recall", "metric_recall.png"),
        zoom=False,
        y_min=0,
        y_max=1.05,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_path_cost",
        ylabel="Average Path Cost",
        title="Average Path Cost (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "astar_path_cost", "metric_astar_path_cost.png"),
        zoom=False,
        y_min=0,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key="avg_cost_per_hop",
        ylabel="Average Cost per Hop",
        title="Average Cost per Hop (A*) vs Embedding Size",
        output_path=get_plot_path(plot_root, "astar_cost_per_hop", "metric_astar_cost_per_hop.png"),
        zoom=False,
        y_min=0,
    )

    # -------------------------------------------------------------------------
    # P95 / visited-node analysis plots from previous split
    # -------------------------------------------------------------------------

    if visited_nodes_path.exists():
        visited_nodes_data = load_json(visited_nodes_path)
        p95_df = extract_p95_data(visited_nodes_data)

        plot_p95_bar_chart(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                f"p95_from_{p95_source_dataset}_successful_only_bar.png",
            ),
            successful_only=True,
        )

        plot_p95_bar_chart(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                f"p95_from_{p95_source_dataset}_all_bar.png",
            ),
            successful_only=False,
        )

        plot_p95_vs_dimension(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                f"p95_from_{p95_source_dataset}_successful_only_vs_dimension.png",
            ),
            successful_only=True,
        )

        plot_visited_distributions_grid(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "visited_node_distributions",
                f"distribution_from_{p95_source_dataset}_successful_only_grid.png",
            ),
            successful_only=True,
        )

        plot_visited_distributions_grid(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "visited_node_distributions",
                f"distribution_from_{p95_source_dataset}_all_grid.png",
            ),
            successful_only=False,
        )
    else:
        print(f"No visited nodes analysis found at {visited_nodes_path}. Skipping p95 plots.")

    # -------------------------------------------------------------------------
    # Per-example histograms from current evaluation_results.json
    # -------------------------------------------------------------------------

    plot_per_example_histograms(
        per_example_df,
        "nodes_visited",
        get_plot_path(plot_root, "histograms", "hist_nodes_visited.png"),
        log_x=True,
    )

    plot_per_example_histograms(
        per_example_df,
        "path_length",
        get_plot_path(plot_root, "histograms", "hist_path_length.png"),
        log_x=False,
    )

    plot_per_example_histograms(
        per_example_df,
        "time_sec",
        get_plot_path(plot_root, "histograms", "hist_time_sec.png"),
        log_x=True,
    )

    print("\nAll plots created.")