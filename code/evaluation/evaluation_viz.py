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

# Save vector plots for thesis.
# "pdf" is best for LaTeX. Add "png" too if you want preview images.
PLOT_FORMATS = ["pdf"]
PNG_DPI = 300

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


def build_input_paths(dataset_name: str):
    eval_dir = Path("data/evaluation") / dataset_name
    eval_results_path = eval_dir / "evaluation_results.json"

    visited_nodes_path = eval_dir / "visited_nodes_analysis.json"

    return eval_results_path, visited_nodes_path


def build_plot_output_dir(dataset_name: str):
    return Path(PLOT_OUTPUT_DIR) / dataset_name


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_plot_path(plot_root, plot_type, filename):
    plot_dir = Path(plot_root) / plot_type
    ensure_directory(plot_dir)

    # Keep caller filenames unchanged, but remove raster suffix internally.
    # save_plot() will add the final suffix from PLOT_FORMATS.
    return plot_dir / Path(filename).stem


def save_plot(fig, output_path):
    output_path = Path(output_path)

    saved_paths = []

    for fmt in PLOT_FORMATS:
        final_path = output_path.with_suffix(f".{fmt}")

        if fmt == "png":
            fig.savefig(final_path, bbox_inches="tight", dpi=PNG_DPI)
        else:
            fig.savefig(final_path, bbox_inches="tight")

        saved_paths.append(str(final_path))

    print(f"Saved plot: {', '.join(saved_paths)}")


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

    prefix_map = {
        "all-mpnet-base-v2": "MPNet",
        "bge-large-en-v1.5": "BGE Large",
        "mxbai-embed-large-v1": "MxBai Large",
        "Qwen3-Embedding-0.6B": "Qwen 0.6B",
        "Qwen3-Embedding-4B": "Qwen 4B",
    }

    prefix = prefix_map.get(base_name, base_name)

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


def dimension_sort_value(dimension):
    if pd.isna(dimension):
        return -1

    return int(dimension)


def build_run_id(model, dimension, algorithm):
    if pd.isna(dimension):
        return f"{model}__{algorithm}"

    return f"{model}__dim{int(dimension)}__{algorithm}"


def build_run_label(model_label, dimension=None, algorithm=None):
    label = model_label

    if dimension is not None and pd.notna(dimension):
        label = f"{label}\nDim {int(dimension)}"

    if algorithm is not None and algorithm != "A*":
        label = f"{label}\n{algorithm}"

    return label


def run_sort_key(model, dimension, algorithm):
    return (
        model_sort_key(model),
        dimension_sort_value(dimension),
        algorithm or "",
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
                        "run_id": build_run_id(model, dimension, algorithm),
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

    df = pd.DataFrame(rows)

    if not df.empty:
        df["sort_key"] = df.apply(
            lambda row: run_sort_key(
                row["model"],
                row["dimension"],
                row["algorithm"],
            ),
            axis=1,
        )
        df = df.sort_values(["sort_key", "id"]).drop(columns=["sort_key"])
        df = df.reset_index(drop=True)

    return df


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
    save_plot(fig, output_path)
    plt.close(fig)


def plot_metric_pair(
    df,
    baselines,
    metric_key,
    ylabel,
    title,
    plot_root,
    plot_type,
    filename,
    y_min=None,
    y_max=None,
):
    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        ylabel=ylabel,
        title=title,
        output_path=get_plot_path(plot_root, plot_type, filename),
        zoom=False,
        y_min=y_min,
        y_max=y_max,
    )

    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        ylabel=ylabel,
        title=title,
        output_path=get_plot_path(
            plot_root,
            plot_type,
            f"{Path(filename).stem}_zoom.png",
        ),
        zoom=True,
    )


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
                "max_visited_successful_only": analysis.get(
                    "max_visited_successful_only"
                ),
                "p95_visited_all": analysis.get("p95_visited_all"),
                "p95_visited_successful_only": analysis.get(
                    "p95_visited_successful_only"
                ),
                "visited_counts_all": analysis.get("visited_counts_all", []),
                "visited_counts_successful_only": analysis.get(
                    "visited_counts_successful_only",
                    [],
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["sort_key"] = df.apply(
            lambda row: run_sort_key(
                row["model"],
                row["dimension"],
                row["algorithm"],
            ),
            axis=1,
        )
        df = df.sort_values(["sort_key"]).drop(columns=["sort_key"])
        df = df.reset_index(drop=True)

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
    save_plot(fig, output_path)
    plt.close(fig)


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
    save_plot(fig, output_path)
    plt.close(fig)


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
    save_plot(fig, output_path)
    plt.close(fig)


# -------------------------------------------------------------------------
# Per-example histogram plots from evaluation_results.json
# -------------------------------------------------------------------------

def get_per_example_run_groups(per_example_df):
    if per_example_df.empty:
        return []

    groups = []

    group_cols = ["model", "dimension", "algorithm"]

    for (model, dimension, algorithm), subset in per_example_df.groupby(
        group_cols,
        dropna=False,
    ):
        model_label = subset["model_label"].iloc[0]
        title = build_run_label(model_label, dimension, algorithm)

        groups.append(
            {
                "model": model,
                "dimension": dimension,
                "algorithm": algorithm,
                "model_label": model_label,
                "title": title,
                "subset": subset,
                "sort_key": run_sort_key(model, dimension, algorithm),
            }
        )

    groups = sorted(groups, key=lambda item: item["sort_key"])

    return groups


def plot_per_example_histograms(per_example_df, value_col, output_path, log_x=False):
    if per_example_df.empty or value_col not in per_example_df.columns:
        print(f"No data for {value_col}. Skipping.")
        return

    groups = get_per_example_run_groups(per_example_df)

    if not groups:
        print(f"No grouped per-example data for {value_col}. Skipping.")
        return

    n = len(groups)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
    )

    axes = np.array(axes).flatten()

    for ax, group in zip(axes, groups):
        values = group["subset"][value_col].dropna()

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

        ax.set_title(group["title"], fontsize=9)
        ax.set_xlabel(value_col)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(groups):]:
        ax.axis("off")

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close(fig)


# -------------------------------------------------------------------------
# Confusion matrix plots from evaluation_results.json
# -------------------------------------------------------------------------

def format_confusion_value(value, total=None, normalize=False):
    if not normalize or total is None or total == 0:
        return str(int(value))

    percent = 100.0 * value / total
    return f"{int(value)}\n({percent:.1f}%)"


def compute_confusion_counts(subset):
    true_values = subset["true"].astype(bool)
    pred_values = subset["pred"].astype(bool)

    tp = int(((true_values == True) & (pred_values == True)).sum())
    fn = int(((true_values == True) & (pred_values == False)).sum())
    fp = int(((true_values == False) & (pred_values == True)).sum())
    tn = int(((true_values == False) & (pred_values == False)).sum())

    return tp, fn, fp, tn


def plot_single_confusion_matrix(ax, matrix, title, normalize=False):
    im = ax.imshow(matrix)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred False", "Pred True"])
    ax.set_yticklabels(["True False", "True True"])

    total = matrix.sum()

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format_confusion_value(matrix[i, j], total, normalize),
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    return im


def extract_confusion_matrix_rows(per_example_df):
    rows = []

    if per_example_df.empty:
        return pd.DataFrame(rows)

    required_cols = {"model", "dimension", "algorithm", "true", "pred"}
    missing_cols = required_cols - set(per_example_df.columns)

    if missing_cols:
        print(f"Missing columns for confusion matrices: {missing_cols}.")
        return pd.DataFrame(rows)

    df = per_example_df.copy()
    df = df[df["true"].notna() & df["pred"].notna()]

    if df.empty:
        return pd.DataFrame(rows)

    group_cols = ["model", "dimension", "algorithm"]

    for (model, dimension, algorithm), subset in df.groupby(group_cols, dropna=False):
        tp, fn, fp, tn = compute_confusion_counts(subset)

        accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1_value = 0.0
        else:
            f1_value = 2 * precision * recall / (precision + recall)

        model_label = parse_model_label(model)

        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "dimension": dimension,
                "algorithm": algorithm,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "accuracy": accuracy,
                "f1_score": f1_value,
                "precision": precision,
                "recall": recall,
                "sort_key": run_sort_key(model, dimension, algorithm),
            }
        )

    matrix_df = pd.DataFrame(rows)

    if not matrix_df.empty:
        matrix_df = matrix_df.sort_values(["sort_key"]).drop(columns=["sort_key"])
        matrix_df = matrix_df.reset_index(drop=True)

    return matrix_df


def select_best_dimension_per_model(matrix_df):
    selected_rows = []

    for model in sorted(matrix_df["model"].unique(), key=model_sort_key):
        subset = matrix_df[matrix_df["model"] == model].copy()
        subset = subset.sort_values(
            ["f1_score", "accuracy", "dimension"],
            ascending=[False, False, True],
            na_position="first",
        )
        selected_rows.append(subset.iloc[0])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_confusion_matrices_grid(
    per_example_df,
    output_path,
    normalize=False,
    best_dimension_only=False,
):
    matrix_df = extract_confusion_matrix_rows(per_example_df)

    if matrix_df.empty:
        print("No confusion matrix rows created. Skipping.")
        return

    if best_dimension_only:
        matrix_df = select_best_dimension_per_model(matrix_df)

    n = len(matrix_df)
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.0 * n_rows),
        constrained_layout=True,
    )

    axes = np.array(axes).flatten()

    last_im = None

    for ax, (_, row) in zip(axes, matrix_df.iterrows()):
        # Matrix layout:
        #
        #              Pred False    Pred True
        # True False       TN            FP
        # True True        FN            TP
        #
        matrix = np.array(
            [
                [row["tn"], row["fp"]],
                [row["fn"], row["tp"]],
            ]
        )

        title = build_run_label(
            row["model_label"],
            row["dimension"],
            row["algorithm"],
        )
        title = f"{title}\nF1={row['f1_score']:.3f}, Acc={row['accuracy']:.3f}"

        last_im = plot_single_confusion_matrix(
            ax=ax,
            matrix=matrix,
            title=title,
            normalize=normalize,
        )

    for ax in axes[len(matrix_df):]:
        ax.axis("off")

    title_suffix = "Best Dimension per Model" if best_dimension_only else "All Runs"
    norm_suffix = "Normalized" if normalize else "Counts"

    fig.suptitle(
        f"Confusion Matrices — {title_suffix} — {norm_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes[:len(matrix_df)].tolist(),
            shrink=0.75,
        )

    save_plot(fig, output_path)
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    dataset_name = dataset_name_from_arg(args.dataset)
    current_split = detect_split(dataset_name)

    eval_results_path, visited_nodes_path = build_input_paths(dataset_name)
    plot_root = build_plot_output_dir(dataset_name)

    ensure_directory(plot_root)

    print(f"Dataset: {dataset_name}")
    print(f"Split: {current_split}")
    print(f"Evaluation results: {eval_results_path}")
    print(f"Visited nodes analysis: {visited_nodes_path}")
    print(f"Plot output dir: {plot_root}")
    print(f"Plot formats: {PLOT_FORMATS}")

    # -------------------------------------------------------------------------
    # Standard evaluation plots from evaluation_results.json
    #
    # These plots are only possible when evaluation_results.json exists.
    # This allows train-only visited-node analysis plots to be created even
    # when train has no evaluation_results.json.
    # -------------------------------------------------------------------------

    if eval_results_path.exists():
        eval_data = load_json(eval_results_path)

        df = extract_semantic_data(eval_data)
        baselines = extract_baselines(eval_data)
        per_example_df = extract_per_example_rows(eval_data)

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="avg_nodes_visited",
            ylabel="Average Visited Nodes",
            title="Average Visited Nodes (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="nodes_visited",
            filename="metric_nodes_visited.png",
            y_min=0,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="avg_time_sec",
            ylabel="Average Time (seconds)",
            title="Average Runtime (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="execution_time",
            filename="metric_time.png",
            y_min=0,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="avg_path_length",
            ylabel="Average Path Length",
            title="Average Path Length (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="path_length",
            filename="metric_path_length.png",
            y_min=0,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="accuracy",
            ylabel="Accuracy",
            title="Accuracy (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="accuracy",
            filename="metric_accuracy.png",
            y_min=0,
            y_max=1.05,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="f1_score",
            ylabel="F1 Score",
            title="F1 Score (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="f1_score",
            filename="metric_f1_score.png",
            y_min=0,
            y_max=1.05,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="precision",
            ylabel="Precision",
            title="Precision (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="precision",
            filename="metric_precision.png",
            y_min=0,
            y_max=1.05,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="recall",
            ylabel="Recall",
            title="Recall (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="recall",
            filename="metric_recall.png",
            y_min=0,
            y_max=1.05,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="avg_path_cost",
            ylabel="Average Path Cost",
            title="Average Path Cost (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="astar_path_cost",
            filename="metric_astar_path_cost.png",
            y_min=0,
        )

        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key="avg_cost_per_hop",
            ylabel="Average Cost per Hop",
            title="Average Cost per Hop (A*) vs Embedding Size",
            plot_root=plot_root,
            plot_type="astar_cost_per_hop",
            filename="metric_astar_cost_per_hop.png",
            y_min=0,
        )

        # ---------------------------------------------------------------------
        # Per-example histograms from current evaluation_results.json
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Confusion matrices from current evaluation_results.json
        # ---------------------------------------------------------------------

        plot_confusion_matrices_grid(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrices",
                "confusion_matrices_all_runs.png",
            ),
            normalize=False,
            best_dimension_only=False,
        )

        plot_confusion_matrices_grid(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrices",
                "confusion_matrices_all_runs_normalized.png",
            ),
            normalize=True,
            best_dimension_only=False,
        )

        plot_confusion_matrices_grid(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrices",
                "confusion_matrices_best_dimension_per_model.png",
            ),
            normalize=False,
            best_dimension_only=True,
        )

        plot_confusion_matrices_grid(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrices",
                "confusion_matrices_best_dimension_per_model_normalized.png",
            ),
            normalize=True,
            best_dimension_only=True,
        )

    else:
        print(
            f"No evaluation results found at {eval_results_path}. "
            "Skipping metric/histogram/confusion-matrix plots."
        )

    # -------------------------------------------------------------------------
    # P95 / visited-node analysis plots from the current split only
    #
    # These plots are independent of evaluation_results.json.
    # -------------------------------------------------------------------------

    if visited_nodes_path.exists():
        visited_nodes_data = load_json(visited_nodes_path)
        p95_df = extract_p95_data(visited_nodes_data)

        plot_p95_bar_chart(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                "p95_successful_only_bar.png",
            ),
            successful_only=True,
        )

        plot_p95_bar_chart(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                "p95_all_bar.png",
            ),
            successful_only=False,
        )

        plot_p95_vs_dimension(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "p95_visited_nodes",
                "p95_successful_only_vs_dimension.png",
            ),
            successful_only=True,
        )

        plot_visited_distributions_grid(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "visited_node_distributions",
                "distribution_successful_only_grid.png",
            ),
            successful_only=True,
        )

        plot_visited_distributions_grid(
            p95_df=p95_df,
            output_path=get_plot_path(
                plot_root,
                "visited_node_distributions",
                "distribution_all_grid.png",
            ),
            successful_only=False,
        )

    else:
        print(
            f"No visited nodes analysis found at {visited_nodes_path}. "
            "Skipping p95/distribution plots for this split."
        )

    print("\nAll available plots created.")