import argparse
import json
import math
from pathlib import Path

from matplotlib import colors as mcolors
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
    "granite-embedding-small-english-r2": "Granite Small Base",
    "granite-embedding-english-r2": "Granite Base",
    "BFS_Baseline": "BFS Baseline",
    "RL_Baseline": "RL Baseline",
}

MODEL_BASE_COLORS = {
    "all-mpnet-base-v2": "#0072B2",
    "bge-large-en-v1.5": "#D55E00",
    "mxbai-embed-large-v1": "#009E73",
    "Qwen3-Embedding-0.6B": "#E69F00",
    "Qwen3-Embedding-4B": "#CC79A7",
    "granite-embedding-small-english-r2": "#56B4E9",
    "granite-embedding-english-r2": "#56B4E9",
}

MODEL_BASE_MARKERS = {
    "all-mpnet-base-v2": "o",
    "bge-large-en-v1.5": "s",
    "mxbai-embed-large-v1": "^",
    "Qwen3-Embedding-0.6B": "D",
    "Qwen3-Embedding-4B": "P",
    "granite-embedding-small-english-r2": "v",
    "granite-embedding-english-r2": "v",
}


def apply_thesis_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": PNG_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
                "Computer Modern Roman",
            ],
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            "legend.fontsize": 8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#333333",
            "legend.fancybox": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
        }
    )


apply_thesis_plot_style()


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
    parser.add_argument(
        "--run-suffix",
        default="best_v2",
        help="Run suffix used in evaluation and plot paths, e.g. best_v2.",
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


def build_input_paths(dataset_name: str, run_suffix: str):
    eval_dir = Path("data/evaluation") / dataset_name / run_suffix
    eval_results_path = eval_dir / "evaluation_results.json"
    visited_nodes_path = eval_dir / "visited_nodes_analysis.json"

    return eval_results_path, visited_nodes_path


def build_plot_output_dir(dataset_name: str, run_suffix: str):
    return Path(PLOT_OUTPUT_DIR) / dataset_name / run_suffix


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


def sanitize_path_component(value):
    value = str(value)
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace(":", "_")
    value = value.replace(" ", "_")
    return value


# -------------------------------------------------------------------------
# Model labels / sorting
# -------------------------------------------------------------------------

MODEL_NAME_STOP_TOKENS = {
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


def adjust_color(color, amount):
    rgb = np.array(mcolors.to_rgb(color))

    if amount >= 0:
        adjusted = rgb + (1.0 - rgb) * amount
    else:
        adjusted = rgb * (1.0 + amount)

    return mcolors.to_hex(np.clip(adjusted, 0.0, 1.0))


def get_model_base_name(model_name):
    name = str(model_name).removesuffix("_finetuned")
    name = name.removesuffix("_best")
    parts = name.split("_")

    base_parts = []
    for part in parts:
        if part in MODEL_NAME_STOP_TOKENS:
            break
        base_parts.append(part)

    return "_".join(base_parts)


def is_finetuned_model(model_name):
    name = str(model_name)

    if name.endswith("_finetuned"):
        return True

    parts = name.removesuffix("_finetuned").split("_")
    return any(part in MODEL_NAME_STOP_TOKENS for part in parts)


def parse_model_label(model_name):
    if model_name in BASE_MODEL_LABELS:
        return BASE_MODEL_LABELS[model_name]

    name = model_name.removesuffix("_finetuned")
    name = name.removesuffix("_best")
    parts = name.split("_")
    base_name = get_model_base_name(model_name)

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
        "granite-embedding-small-english-r2": "Granite Small",
        "granite-embedding-english-r2": "Granite",
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
    base_name = get_model_base_name(model_name)

    base_order = {
        "BFS_Baseline": -2,
        "RL_Baseline": -1,
        "all-mpnet-base-v2": 0,
        "bge-large-en-v1.5": 1,
        "mxbai-embed-large-v1": 2,
        "Qwen3-Embedding-0.6B": 3,
        "Qwen3-Embedding-4B": 4,
        "granite-embedding-small-english-r2": 5,
        "granite-embedding-english-r2": 6,
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


def run_sort_key(model, dimension, algorithm):
    return (
        model_sort_key(model),
        dimension_sort_value(dimension),
        algorithm or "",
    )


def build_run_label(model_label, dimension=None, algorithm=None):
    label = model_label

    if dimension is not None and pd.notna(dimension):
        label = f"{label}\nDim {int(dimension)}"

    if algorithm is not None and algorithm != "A*":
        label = f"{label}\n{algorithm}"

    return label


def normalize_metric_name(metric_key):
    if metric_key == "f1_score":
        return "F1 Score"
    if metric_key == "accuracy":
        return "Accuracy"
    if metric_key == "precision":
        return "Precision"
    if metric_key == "recall":
        return "Recall"
    if metric_key == "avg_time_ms":
        return "Average Runtime (ms)"
    if metric_key == "avg_time_sec":
        return "Average Runtime (seconds)"
    if metric_key == "avg_nodes_visited":
        return "Average Visited Nodes"

    return metric_key


def get_metric_value(metrics, metric_key):
    if metric_key == "avg_time_ms":
        if metrics.get("avg_time_ms") is not None:
            return metrics.get("avg_time_ms")
        if metrics.get("avg_time_sec") is not None:
            return metrics.get("avg_time_sec") * 1000.0
        return None

    if metric_key == "avg_time_sec":
        if metrics.get("avg_time_sec") is not None:
            return metrics.get("avg_time_sec")
        if metrics.get("avg_time_ms") is not None:
            return metrics.get("avg_time_ms") / 1000.0
        return None

    return metrics.get(metric_key)


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
                "accuracy": get_metric_value(metrics, "accuracy"),
                "f1_score": get_metric_value(metrics, "f1_score"),
                "precision": get_metric_value(metrics, "precision"),
                "recall": get_metric_value(metrics, "recall"),
                "avg_time_ms": get_metric_value(metrics, "avg_time_ms"),
                "avg_time_sec": get_metric_value(metrics, "avg_time_sec"),
                "avg_nodes_visited": get_metric_value(metrics, "avg_nodes_visited"),
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
                    "accuracy": get_metric_value(metrics, "accuracy"),
                    "f1_score": get_metric_value(metrics, "f1_score"),
                    "precision": get_metric_value(metrics, "precision"),
                    "recall": get_metric_value(metrics, "recall"),
                    "avg_time_ms": get_metric_value(metrics, "avg_time_ms"),
                    "avg_time_sec": get_metric_value(metrics, "avg_time_sec"),
                    "avg_nodes_visited": get_metric_value(metrics, "avg_nodes_visited"),
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
# Metric plots from evaluation_results.json
# -------------------------------------------------------------------------

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


def collect_metric_values(df, baselines, metric_key):
    values = []

    if not df.empty and metric_key in df.columns:
        values.extend(df[metric_key].dropna().astype(float).tolist())

    for baseline in baselines.values():
        value = baseline.get(metric_key)
        if value is not None and not pd.isna(value):
            values.append(float(value))

    return values


def compute_zoom_ylim(df, baselines, metric_key, pad_ratio=0.12, min_pad=1e-6):
    values = collect_metric_values(df, baselines, metric_key)

    if not values:
        return None, None

    y_min = min(values)
    y_max = max(values)

    if y_min == y_max:
        pad = max(abs(y_min) * pad_ratio, min_pad)
    else:
        pad = max((y_max - y_min) * pad_ratio, min_pad)

    return y_min - pad, y_max + pad


def get_model_line_style(model_name):
    base_name = get_model_base_name(model_name)
    family_color = MODEL_BASE_COLORS.get(base_name, "#4C78A8")
    marker = MODEL_BASE_MARKERS.get(base_name, "o")

    if is_finetuned_model(model_name):
        color = adjust_color(family_color, -0.08)

        return {
            "color": color,
            "marker": marker,
            "linestyle": "--",
            "linewidth": 2.0,
            "markersize": 5.6,
            "markerfacecolor": "white",
            "markeredgecolor": color,
            "markeredgewidth": 1.1,
            "zorder": 4,
        }

    color = adjust_color(family_color, 0.18)

    return {
        "color": color,
        "marker": marker,
        "linestyle": "-",
        "linewidth": 1.7,
        "markersize": 5.0,
        "markerfacecolor": color,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
        "zorder": 4,
    }


def plot_model_lines_on_axis(ax, df, metric_key, y_limits=None):
    for model, label, subset in get_model_subsets(df, metric_key):
        y_values = subset[metric_key].astype(float)

        if y_limits is not None:
            y_min, y_max = y_limits
            y_values = y_values.where(
                (y_values >= y_min) & (y_values <= y_max),
                np.nan,
            )

            if y_values.notna().sum() == 0:
                continue

        ax.plot(
            subset["dimension"],
            y_values,
            label=label,
            **get_model_line_style(model),
        )


BASELINE_LINE_STYLES = [
    {"color": "#111111", "linestyle": "--", "linewidth": 1.2, "zorder": 3},
    {"color": "#777777", "linestyle": "-.", "linewidth": 1.2, "zorder": 3},
    {"color": "#444444", "linestyle": ":", "linewidth": 1.2, "zorder": 3},
    {"color": "#999999", "linestyle": "--", "linewidth": 1.2, "zorder": 3},
]


def get_baseline_line_style(baselines, baseline_name):
    baseline_names = sorted(baselines.keys(), key=model_sort_key)
    baseline_index = baseline_names.index(baseline_name)

    return BASELINE_LINE_STYLES[
        baseline_index % len(BASELINE_LINE_STYLES)
    ]


def get_baseline_label(baseline_name, baseline):
    return baseline.get("model_label") or parse_model_label(baseline_name)


def get_valid_baseline_names(baselines, metric_key):
    names = []

    for baseline_name in sorted(baselines.keys(), key=model_sort_key):
        baseline = baselines.get(baseline_name)

        if not baseline:
            continue

        value = baseline.get(metric_key)

        if value is None or pd.isna(value):
            continue

        names.append(baseline_name)

    return names


def plot_selected_baseline_lines_on_axis(ax, baselines, metric_key, selected_names):
    for baseline_name in selected_names:
        baseline = baselines.get(baseline_name)

        if not baseline:
            continue

        value = baseline.get(metric_key)

        if value is None or pd.isna(value):
            continue

        ax.axhline(
            y=value,
            label=get_baseline_label(baseline_name, baseline),
            **get_baseline_line_style(baselines, baseline_name),
        )


def get_dimension_ticks(df):
    if df.empty or "dimension" not in df.columns:
        return []

    return sorted(
        int(value)
        for value in df["dimension"].dropna().unique()
        if int(value) > 0
    )


def style_dimension_axis(ax, dimensions):
    if not dimensions:
        return

    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions)
    if len(dimensions) > 8:
        ax.set_xticklabels(
            [str(value) for value in dimensions],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels([str(value) for value in dimensions])
    ax.set_xlim(dimensions[0] / 1.12, dimensions[-1] * 1.12)
    ax.margins(x=0.03)


def style_metric_axis(ax, df):
    style_dimension_axis(ax, get_dimension_ticks(df))
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)


def plot_metric_lines_on_axis(ax, df, baselines, metric_key):
    plot_model_lines_on_axis(ax, df, metric_key)
    plot_selected_baseline_lines_on_axis(
        ax=ax,
        baselines=baselines,
        metric_key=metric_key,
        selected_names=get_valid_baseline_names(baselines, metric_key),
    )


def compute_values_ylim(values, pad_ratio=0.18, min_pad=1e-6):
    if not values:
        return None, None

    y_min = min(values)
    y_max = max(values)

    if y_min == y_max:
        pad = max(abs(y_min) * pad_ratio, min_pad)
    else:
        pad = max((y_max - y_min) * pad_ratio, min_pad)

    lower = y_min - pad
    upper = y_max + pad

    if y_min >= 0:
        lower = max(0, lower)

    return lower, upper


def get_model_metric_values(df, metric_key):
    if df.empty or metric_key not in df.columns:
        return []

    return df[metric_key].dropna().astype(float).tolist()


def compute_model_focus_ylim(df, metric_key):
    values = get_model_metric_values(df, metric_key)
    return compute_values_ylim(values, pad_ratio=0.18)


def split_baselines_by_ylim(baselines, metric_key, y_min, y_max):
    below = []
    inside = []
    above = []

    for baseline_name in get_valid_baseline_names(baselines, metric_key):
        baseline = baselines[baseline_name]
        value = float(baseline[metric_key])

        if value < y_min:
            below.append(baseline_name)
        elif value > y_max:
            above.append(baseline_name)
        else:
            inside.append(baseline_name)

    return below, inside, above


def compute_baseline_ylim(baselines, metric_key, selected_names):
    values = [
        float(baselines[baseline_name][metric_key])
        for baseline_name in selected_names
    ]

    return compute_values_ylim(values, pad_ratio=0.12)


def should_split_focus_values(values, max_bands):
    if max_bands <= 1 or len(values) < 12:
        return False

    sorted_values = np.array(sorted(values), dtype=float)

    if sorted_values[0] <= 0:
        return False

    q25, q75 = np.quantile(sorted_values, [0.25, 0.75])

    if q25 <= 0 or q75 <= 0:
        return False

    return (
        sorted_values[-1] / sorted_values[0] >= 4.0
        and q75 / q25 >= 2.5
        and sorted_values[-1] / q75 >= 1.4
    )


def build_focus_value_bands(values, max_bands=3):
    values = sorted(float(value) for value in values if value is not None and not pd.isna(value))

    if not values:
        return []

    if not should_split_focus_values(values, max_bands):
        return [
            {
                "ylim": compute_values_ylim(values, pad_ratio=0.18),
                "height": 3.0,
            }
        ]

    split_fractions = [0.25, 0.75] if max_bands >= 3 else [0.5]
    split_indices = []
    min_segment_size = max(4, int(len(values) * 0.10))

    for fraction in split_fractions:
        index = int(round(len(values) * fraction))
        index = min(max(index, min_segment_size), len(values) - min_segment_size)

        if index not in split_indices:
            split_indices.append(index)

    split_indices = sorted(split_indices)
    boundaries = [0] + split_indices + [len(values)]
    bands = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_values = values[start:end]

        if not segment_values:
            continue

        bands.append(
            {
                "raw_min": min(segment_values),
                "raw_max": max(segment_values),
                "ylim": compute_values_ylim(segment_values, pad_ratio=0.08),
                "height": 1.45,
            }
        )

    if not bands:
        return [
            {
                "ylim": compute_values_ylim(values, pad_ratio=0.18),
                "height": 3.0,
            }
        ]

    for index in range(len(bands) - 1):
        lower_band = bands[index]
        upper_band = bands[index + 1]
        boundary = (lower_band["raw_max"] + upper_band["raw_min"]) / 2.0

        lower_min, lower_max = lower_band["ylim"]
        upper_min, upper_max = upper_band["ylim"]

        lower_band["ylim"] = (lower_min, min(lower_max, boundary))
        upper_band["ylim"] = (max(upper_min, boundary), upper_max)

    return bands


def assign_baselines_to_focus_bands(baselines, metric_key, baseline_names, focus_bands):
    assignments = {index: [] for index in range(len(focus_bands))}

    for baseline_name in baseline_names:
        value = float(baselines[baseline_name][metric_key])
        matching_index = None

        for index, band in enumerate(focus_bands):
            y_min, y_max = band["ylim"]

            if y_min <= value <= y_max:
                matching_index = index
                break

        if matching_index is None:
            distances = []

            for index, band in enumerate(focus_bands):
                y_min, y_max = band["ylim"]
                distances.append(
                    (
                        min(abs(value - y_min), abs(value - y_max)),
                        index,
                    )
                )

            matching_index = min(distances)[1]

        assignments[matching_index].append(baseline_name)

    return assignments


def build_piecewise_axis(panels, gap_height=0.07):
    bands = []

    for panel in panels:
        y_min, y_max = panel["ylim"]

        if y_min is None or y_max is None or y_min == y_max:
            continue

        bands.append(
            {
                "source_min": float(y_min),
                "source_max": float(y_max),
                "height": float(panel["height"]),
            }
        )

    bands = sorted(bands, key=lambda band: band["source_min"])

    offset = 0.0
    for index, band in enumerate(bands):
        band["display_min"] = offset
        band["display_max"] = offset + band["height"]
        offset = band["display_max"]

        if index < len(bands) - 1:
            offset += gap_height

    return bands, max(offset, 1.0), gap_height


def transform_piecewise_y(values, bands, gap_height):
    array = np.asarray(values, dtype=float)
    transformed = np.full(array.shape, np.nan, dtype=float)

    if not bands:
        return transformed

    for band in bands:
        source_min = band["source_min"]
        source_max = band["source_max"]
        display_min = band["display_min"]
        display_max = band["display_max"]

        mask = (array >= source_min) & (array <= source_max)

        if source_max == source_min:
            transformed[mask] = display_min
        else:
            transformed[mask] = display_min + (
                (array[mask] - source_min)
                / (source_max - source_min)
                * (display_max - display_min)
            )

    for lower_band, upper_band in zip(bands[:-1], bands[1:]):
        gap_min = lower_band["source_max"]
        gap_max = upper_band["source_min"]

        if gap_max <= gap_min:
            continue

        mask = (array > gap_min) & (array < gap_max)

        transformed[mask] = lower_band["display_max"] + (
            (array[mask] - gap_min)
            / (gap_max - gap_min)
            * gap_height
        )

    below_mask = array < bands[0]["source_min"]
    above_mask = array > bands[-1]["source_max"]

    transformed[below_mask] = bands[0]["display_min"]
    transformed[above_mask] = bands[-1]["display_max"]

    return transformed


def transform_piecewise_scalar(value, bands, gap_height):
    return float(transform_piecewise_y([value], bands, gap_height)[0])


def get_piecewise_ticks(bands, gap_height, max_ticks_per_band=4):
    tick_values = []

    for band in bands:
        source_min = band["source_min"]
        source_max = band["source_max"]

        values = np.linspace(source_min, source_max, max_ticks_per_band)

        for value in values:
            tick_values.append(float(value))

    unique_values = []
    seen_values = set()

    for value in tick_values:
        key = round(value, 6)

        if key in seen_values:
            continue

        seen_values.add(key)
        unique_values.append(value)

    transformed_ticks = transform_piecewise_y(unique_values, bands, gap_height)

    labels = []
    for value in unique_values:
        if abs(value) >= 100:
            labels.append(f"{value:.0f}")
        elif abs(value) >= 10:
            labels.append(f"{value:.1f}".rstrip("0").rstrip("."))
        else:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))

    return transformed_ticks, labels


def draw_piecewise_break_marks(ax, bands):
    if len(bands) < 2:
        return

    for lower_band, upper_band in zip(bands[:-1], bands[1:]):
        y_min = lower_band["display_max"]
        y_max = upper_band["display_min"]

        center = (y_min + y_max) / 2.0
        amplitude = max((y_max - y_min) * 0.24, 0.010)
        zigzag_x = np.linspace(0.02, 0.98, 75)
        zigzag_y = center + np.where(
            np.arange(len(zigzag_x)) % 2 == 0,
            amplitude,
            -amplitude,
        )

        ax.plot(
            zigzag_x,
            zigzag_y,
            transform=ax.get_yaxis_transform(),
            color="#777777",
            linewidth=0.7,
            clip_on=False,
            zorder=6,
        )

        slash_kwargs = dict(
            transform=ax.get_yaxis_transform(),
            color="black",
            linewidth=1.4,
            clip_on=False,
            zorder=7,
        )

        ax.plot([-0.012, 0.012], [center - amplitude, center + amplitude], **slash_kwargs)
        ax.plot([0.988, 1.012], [center - amplitude, center + amplitude], **slash_kwargs)


def plot_model_lines_on_piecewise_axis(ax, df, metric_key, bands, gap_height):
    for model, label, subset in get_model_subsets(df, metric_key):
        y_values = transform_piecewise_y(
            subset[metric_key].astype(float).to_numpy(),
            bands,
            gap_height,
        )

        ax.plot(
            subset["dimension"],
            y_values,
            label=label,
            **get_model_line_style(model),
        )


def plot_baseline_lines_on_piecewise_axis(
    ax,
    baselines,
    metric_key,
    selected_names,
    bands,
    gap_height,
):
    for baseline_name in selected_names:
        baseline = baselines.get(baseline_name)

        if not baseline:
            continue

        value = baseline.get(metric_key)

        if value is None or pd.isna(value):
            continue

        transformed_value = transform_piecewise_scalar(value, bands, gap_height)

        ax.axhline(
            y=transformed_value,
            label=get_baseline_label(baseline_name, baseline),
            **get_baseline_line_style(baselines, baseline_name),
        )


def add_broken_axis_marks(ax_top, ax_bottom):
    # Draw small diagonal break marks between the upper and lower y-axis panels.
    d = 0.012

    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_bottom.transAxes, color="black", clip_on=False)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    zigzag_x = np.linspace(0.02, 0.98, 65)
    zigzag_y = np.where(np.arange(len(zigzag_x)) % 2 == 0, -0.040, -0.080)
    ax_top.plot(
        zigzag_x,
        zigzag_y,
        transform=ax_top.transAxes,
        color="#8a8a8a",
        linewidth=0.65,
        clip_on=False,
        zorder=10,
    )


def plot_metric_vs_dimension(
    df,
    baselines,
    metric_key,
    output_path,
    y_min=None,
    y_max=None,
):
    if df.empty:
        print(f"No semantic data. Skipping {metric_key}")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    plot_metric_lines_on_axis(ax, df, baselines, metric_key)

    ylabel = normalize_metric_name(metric_key)

    ax.set_title(f"{ylabel} vs Embedding Size")
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel(ylabel)
    style_metric_axis(ax, df)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.legend(loc="best")
    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close(fig)


def build_metric_panels(df, baselines, metric_key, max_focus_bands=3):
    focus_min, focus_max = compute_model_focus_ylim(
        df=df,
        metric_key=metric_key,
    )

    if focus_min is None or focus_max is None:
        return [], False

    below_baselines, inside_baselines, above_baselines = split_baselines_by_ylim(
        baselines=baselines,
        metric_key=metric_key,
        y_min=focus_min,
        y_max=focus_max,
    )

    is_broken = bool(below_baselines or above_baselines)
    model_values = get_model_metric_values(df, metric_key)
    inside_baseline_values = [
        float(baselines[baseline_name][metric_key])
        for baseline_name in inside_baselines
    ]
    focus_bands = build_focus_value_bands(
        model_values + inside_baseline_values,
        max_bands=max_focus_bands,
    )

    if not focus_bands:
        return [], False

    baseline_assignments = assign_baselines_to_focus_bands(
        baselines=baselines,
        metric_key=metric_key,
        baseline_names=inside_baselines,
        focus_bands=focus_bands,
    )
    is_broken = is_broken or len(focus_bands) > 1

    if not is_broken:
        return [
            {
                "kind": "focus",
                "baselines": baseline_assignments.get(0, []),
                "ylim": focus_bands[0]["ylim"],
                "height": focus_bands[0]["height"],
            }
        ], False

    panels = []

    if above_baselines:
        panels.append(
            {
                "kind": "baseline",
                "baselines": above_baselines,
                "ylim": compute_baseline_ylim(
                    baselines,
                    metric_key,
                    above_baselines,
                ),
                "height": 0.42,
            }
        )

    for index, focus_band in reversed(list(enumerate(focus_bands))):
        panels.append(
            {
                "kind": "focus",
                "baselines": baseline_assignments.get(index, []),
                "ylim": focus_band["ylim"],
                "height": focus_band["height"],
            }
        )

    if below_baselines:
        panels.append(
            {
                "kind": "baseline",
                "baselines": below_baselines,
                "ylim": compute_baseline_ylim(
                    baselines,
                    metric_key,
                    below_baselines,
                ),
                "height": 0.42,
            }
        )

    return panels, True


def plot_metric_vs_dimension_broken_axis(
    df,
    baselines,
    metric_key,
    output_path,
):
    if df.empty:
        print(f"No semantic data. Skipping broken-axis {metric_key}")
        return

    panels, is_broken = build_metric_panels(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        max_focus_bands=3,
    )

    if not panels:
        print(f"No valid model data. Skipping broken-axis {metric_key}")
        return

    if not is_broken:
        fig, ax = plt.subplots(figsize=(11, 6))
        panel = panels[0]

        plot_model_lines_on_axis(ax, df, metric_key)
        plot_selected_baseline_lines_on_axis(
            ax=ax,
            baselines=baselines,
            metric_key=metric_key,
            selected_names=panel["baselines"],
        )

        ylabel = normalize_metric_name(metric_key)

        ax.set_title(f"{ylabel} vs Embedding Size - Zoomed")
        ax.set_xlabel("Embedding Size")
        ax.set_ylabel(ylabel)
        style_metric_axis(ax, df)
        panel_min, panel_max = panel["ylim"]
        ax.set_ylim(bottom=panel_min, top=panel_max)
        ax.legend(loc="best")

        plt.tight_layout()
        save_plot(fig, output_path)
        plt.close(fig)
        return

    bands, display_max, gap_height = build_piecewise_axis(panels)

    ylabel = normalize_metric_name(metric_key)

    fig, ax = plt.subplots(figsize=(11, 6.8))

    plot_model_lines_on_piecewise_axis(
        ax=ax,
        df=df,
        metric_key=metric_key,
        bands=bands,
        gap_height=gap_height,
    )

    baseline_names = get_valid_baseline_names(baselines, metric_key)
    plot_baseline_lines_on_piecewise_axis(
        ax=ax,
        baselines=baselines,
        metric_key=metric_key,
        selected_names=baseline_names,
        bands=bands,
        gap_height=gap_height,
    )

    ax.set_title(f"{ylabel} vs Embedding Size - Broken Axis")
    ax.set_xlabel("Embedding Size")
    ax.set_ylabel(ylabel)
    style_metric_axis(ax, df)
    ax.set_ylim(0, display_max)

    tick_positions, tick_labels = get_piecewise_ticks(bands, gap_height)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    draw_piecewise_break_marks(ax, bands)

    handles, labels = collect_unique_legend_entries([ax])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=8,
        ncol=min(3, max(1, len(labels))),
        bbox_to_anchor=(0.5, 0.03),
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.25,
        top=0.94,
    )
    save_plot(fig, output_path)
    plt.close(fig)


def collect_unique_legend_entries(axes):
    handles = []
    labels = []
    seen_labels = set()

    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()

        for handle, label in zip(ax_handles, ax_labels):
            if label in seen_labels:
                continue

            handles.append(handle)
            labels.append(label)
            seen_labels.add(label)

    return handles, labels


def plot_metric_summary_panel(
    fig,
    grid_spec,
    df,
    baselines,
    metric_key,
    title,
    show_xlabel,
):
    panels, _ = build_metric_panels(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        max_focus_bands=2,
    )

    if not panels:
        return []

    bands, display_max, gap_height = build_piecewise_axis(panels)

    if not bands:
        return []

    ax = fig.add_subplot(grid_spec)
    ylabel = normalize_metric_name(metric_key)

    plot_model_lines_on_piecewise_axis(
        ax=ax,
        df=df,
        metric_key=metric_key,
        bands=bands,
        gap_height=gap_height,
    )

    baseline_names = get_valid_baseline_names(baselines, metric_key)
    plot_baseline_lines_on_piecewise_axis(
        ax=ax,
        baselines=baselines,
        metric_key=metric_key,
        selected_names=baseline_names,
        bands=bands,
        gap_height=gap_height,
    )

    ax.set_title(title, pad=5)
    ax.set_ylabel(ylabel)
    style_metric_axis(ax, df)
    ax.set_ylim(0, display_max)

    tick_positions, tick_labels = get_piecewise_ticks(
        bands,
        gap_height,
        max_ticks_per_band=3,
    )
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    draw_piecewise_break_marks(ax, bands)

    if show_xlabel:
        ax.set_xlabel("Embedding Size")
    else:
        ax.tick_params(labelbottom=False)

    return [ax]


def plot_metrics_summary(df, baselines, output_path):
    if df.empty:
        print("No semantic data. Skipping metrics summary.")
        return

    summary_metrics = [
        ("f1_score", "F1 Score"),
        ("accuracy", "Accuracy"),
        ("avg_nodes_visited", "Average Visited Nodes"),
        ("avg_time_ms", "Average Runtime"),
    ]

    fig = plt.figure(figsize=(12.5, 8.8))
    outer_grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.98,
        bottom=0.24,
        top=0.91,
        wspace=0.24,
        hspace=0.34,
    )

    all_axes = []

    for index, (metric_key, title) in enumerate(summary_metrics):
        row = index // 2
        col = index % 2

        axes = plot_metric_summary_panel(
            fig=fig,
            grid_spec=outer_grid[row, col],
            df=df,
            baselines=baselines,
            metric_key=metric_key,
            title=title,
            show_xlabel=row == 1,
        )

        all_axes.extend(axes)

    if not all_axes:
        plt.close(fig)
        print("No summary panels created. Skipping metrics summary.")
        return

    handles, labels = collect_unique_legend_entries(all_axes)

    fig.suptitle(
        "Effectiveness and Efficiency Across Embedding Sizes",
        fontsize=14,
        fontweight="bold",
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        bbox_to_anchor=(0.5, 0.035),
    )

    save_plot(fig, output_path)
    plt.close(fig)


def plot_metric_pair(
    df,
    baselines,
    metric_key,
    plot_root,
    y_min=None,
    y_max=None,
):
    plot_metric_vs_dimension(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        output_path=get_plot_path(
            plot_root,
            "metrics",
            f"{metric_key}.png",
        ),
        y_min=y_min,
        y_max=y_max,
    )

    plot_metric_vs_dimension_broken_axis(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        output_path=get_plot_path(
            plot_root,
            "metrics_zoom",
            f"{metric_key}_zoomed.png",
        ),
    )


def plot_all_standard_metrics(df, baselines, plot_root):
    metrics = [
        ("f1_score", 0, 1.05),
        ("accuracy", 0, 1.05),
        ("precision", 0, 1.05),
        ("recall", 0, 1.05),
        ("avg_time_ms", 0, None),
        ("avg_nodes_visited", 0, None),
    ]

    for metric_key, y_min, y_max in metrics:
        plot_metric_pair(
            df=df,
            baselines=baselines,
            metric_key=metric_key,
            plot_root=plot_root,
            y_min=y_min,
            y_max=y_max,
        )

    plot_metrics_summary(
        df=df,
        baselines=baselines,
        output_path=get_plot_path(
            plot_root,
            "metrics_summary",
            "metrics_summary.png",
        ),
    )


# -------------------------------------------------------------------------
# One normalized confusion matrix grid from evaluation_results.json
# -------------------------------------------------------------------------

def compute_confusion_counts(subset):
    true_values = subset["true"].astype(bool)
    pred_values = subset["pred"].astype(bool)

    tp = int(((true_values == True) & (pred_values == True)).sum())
    fn = int(((true_values == True) & (pred_values == False)).sum())
    fp = int(((true_values == False) & (pred_values == True)).sum())
    tn = int(((true_values == False) & (pred_values == False)).sum())

    return tp, fn, fp, tn


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

        total = tp + fn + fp + tn
        accuracy = (tp + tn) / max(total, 1)

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

        rows.append(
            {
                "model": model,
                "model_label": parse_model_label(model),
                "dimension": dimension,
                "algorithm": algorithm,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "total": total,
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


def format_confusion_percentage(value, total):
    if total == 0:
        return "0\n(0.0%)"

    percent = 100.0 * value / total
    return f"{int(value)}\n({percent:.1f}%)"


def plot_single_normalized_confusion_matrix(ax, matrix, title):
    im = ax.imshow(matrix)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred False", "Pred True"])
    ax.set_yticklabels(["Actual False", "Actual True"])

    total = matrix.sum()

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format_confusion_percentage(matrix[i, j], total),
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    return im


def plot_normalized_confusion_matrices_all_models(per_example_df, output_path):
    matrix_df = extract_confusion_matrix_rows(per_example_df)

    if matrix_df.empty:
        print("No confusion matrix rows created. Skipping.")
        return

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
        #                    Predicted False    Predicted True
        # Actual False              TN                FP
        # Actual True               FN                TP
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

        last_im = plot_single_normalized_confusion_matrix(
            ax=ax,
            matrix=matrix,
            title=title,
        )

    for ax in axes[len(matrix_df):]:
        ax.axis("off")

    fig.suptitle(
        "Normalized Confusion Matrices — All Models",
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
# P95 / visited-node analysis extraction
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


# -------------------------------------------------------------------------
# P95 / visited-node analysis plots
# -------------------------------------------------------------------------

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


def get_histogram_bins(counts, log_x=True):
    min_count = min(counts)
    max_count = max(counts)

    if min_count == max_count:
        return 10

    if log_x:
        return np.logspace(
            np.log10(min_count),
            np.log10(max_count),
            30,
        )

    return 30


def get_clean_counts(counts):
    if not isinstance(counts, list):
        return []

    return [int(v) for v in counts if v is not None and v > 0]


def plot_visited_distributions_grid(
    p95_df,
    output_path,
    successful_only=False,
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
        clean_counts = get_clean_counts(row.get(value_col, []))

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
        bins = get_histogram_bins(counts, log_x=True)

        ax.hist(
            counts,
            bins=bins,
            edgecolor="black",
            alpha=0.75,
        )

        if min(counts) != max(counts):
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


def build_individual_histogram_path(plot_root, model, dimension):
    model_dir = sanitize_path_component(model)

    if pd.isna(dimension):
        dim_dir = "baseline"
    else:
        dim_dir = f"dim_{int(dimension)}"

    output_dir = (
        Path(plot_root)
        / "p95_analysis"
        / "individual_histograms"
        / model_dir
        / dim_dir
    )

    ensure_directory(output_dir)

    return output_dir / "histogram"


def plot_single_visited_counts_histogram(
    counts,
    p95_value,
    title,
    output_path,
):
    clean_counts = get_clean_counts(counts)

    if not clean_counts:
        print(f"No valid visited counts for {title}. Skipping.")
        return

    bins = get_histogram_bins(clean_counts, log_x=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        clean_counts,
        bins=bins,
        edgecolor="black",
        alpha=0.75,
    )

    if min(clean_counts) != max(clean_counts):
        ax.set_xscale("log")

    if p95_value is not None and not pd.isna(p95_value):
        ax.axvline(
            p95_value,
            linestyle="--",
            linewidth=2,
            label=f"P95: {p95_value:.1f}",
        )
        ax.legend(fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Visited Nodes")
    ax.set_ylabel("Frequency")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close(fig)


def plot_individual_visited_node_histograms(p95_df, plot_root):
    if p95_df.empty:
        print("No visited-node analysis data found. Skipping individual histograms.")
        return

    for _, row in p95_df.iterrows():
        counts = row.get("visited_counts_all", [])

        if not isinstance(counts, list) or len(counts) == 0:
            continue

        model = row["model"]
        dimension = row["dimension"]

        title = row["model_label"]

        if pd.notna(dimension):
            title = f"{title} | Dim {int(dimension)}"

        title = f"{title} — Visited Nodes Histogram"

        output_path = build_individual_histogram_path(
            plot_root=plot_root,
            model=model,
            dimension=dimension,
        )

        plot_single_visited_counts_histogram(
            counts=counts,
            p95_value=row.get("p95_visited_all"),
            title=title,
            output_path=output_path,
        )


def plot_all_p95_analysis(p95_df, plot_root):
    # Bar chart with p95 values for every baseline/model/dimension.
    plot_p95_bar_chart(
        p95_df=p95_df,
        output_path=get_plot_path(
            plot_root,
            "p95_analysis",
            "p95_successful_only_bar.png",
        ),
        successful_only=True,
    )

    # Line plot: A* p95 over Matryoshka dimensions.
    plot_p95_vs_dimension(
        p95_df=p95_df,
        output_path=get_plot_path(
            plot_root,
            "p95_analysis",
            "p95_successful_only_vs_dimension.png",
        ),
        successful_only=True,
    )

    # One grid plot with all visited-node histograms together.
    plot_visited_distributions_grid(
        p95_df=p95_df,
        output_path=get_plot_path(
            plot_root,
            "p95_analysis",
            "visited_counts_all_grid.png",
        ),
        successful_only=False,
    )

    # Individual histogram:
    # - one for BFS
    # - one for RL
    # - one per A* model and Matryoshka dimension
    plot_individual_visited_node_histograms(
        p95_df=p95_df,
        plot_root=plot_root,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    dataset_name = dataset_name_from_arg(args.dataset)
    current_split = detect_split(dataset_name)

    eval_results_path, visited_nodes_path = build_input_paths(
        dataset_name,
        args.run_suffix,
    )
    plot_root = build_plot_output_dir(
        dataset_name,
        args.run_suffix,
    )

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
    # Created plots:
    # - metrics/f1_score.pdf
    # - metrics/accuracy.pdf
    # - metrics/precision.pdf
    # - metrics/recall.pdf
    # - metrics/avg_time_ms.pdf
    # - metrics/avg_nodes_visited.pdf
    # - metrics_zoom/f1_score_zoomed.pdf
    # - metrics_zoom/accuracy_zoomed.pdf
    # - metrics_zoom/precision_zoomed.pdf
    # - metrics_zoom/recall_zoomed.pdf
    # - metrics_zoom/avg_time_ms_zoomed.pdf
    # - metrics_zoom/avg_nodes_visited_zoomed.pdf
    # - metrics_summary/metrics_summary.pdf
    # - confusion_matrix/confusion_matrices_all_models_normalized.pdf
    # -------------------------------------------------------------------------

    if eval_results_path.exists():
        eval_data = load_json(eval_results_path)

        df = extract_semantic_data(eval_data)
        baselines = extract_baselines(eval_data)
        per_example_df = extract_per_example_rows(eval_data)

        plot_all_standard_metrics(
            df=df,
            baselines=baselines,
            plot_root=plot_root,
        )

        plot_normalized_confusion_matrices_all_models(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrix",
                "confusion_matrices_all_models_normalized.png",
            ),
        )

    else:
        print(
            f"No evaluation results found at {eval_results_path}. "
            "Skipping metric/confusion-matrix plots."
        )

    # -------------------------------------------------------------------------
    # P95 / visited-node analysis plots from visited_nodes_analysis.json
    #
    # Created plots:
    # - p95_analysis/p95_successful_only_bar.pdf
    # - p95_analysis/p95_successful_only_vs_dimension.pdf
    # - p95_analysis/visited_counts_all_grid.pdf
    # - p95_analysis/individual_histograms/BFS_Baseline/baseline/histogram.pdf
    # - p95_analysis/individual_histograms/RL_Baseline/baseline/histogram.pdf
    # - p95_analysis/individual_histograms/<model>/dim_<dim>/histogram.pdf
    # -------------------------------------------------------------------------

    if visited_nodes_path.exists():
        visited_nodes_data = load_json(visited_nodes_path)
        p95_df = extract_p95_data(visited_nodes_data)

        plot_all_p95_analysis(
            p95_df=p95_df,
            plot_root=plot_root,
        )

    else:
        print(
            f"No visited nodes analysis found at {visited_nodes_path}. "
            "Skipping p95/distribution plots for this split."
        )

    print("\nAll available plots created.")
