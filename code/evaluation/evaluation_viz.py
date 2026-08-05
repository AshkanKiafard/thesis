import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd

from core.constants import (
    BASELINE_MODEL_NAMES,
    BFS_CAPPED_BASELINE_MODEL,
    BFS_UNCAPPED_BASELINE_MODEL,
    EVALUATION_DIR,
    PLOTS_DIR,
    RL_BASELINE_MODEL,
    THESIS_PLOTS_DIR,
)
from core.graph_config import (
    DEFAULT_GRAPH_NAME,
    canonical_graph_name,
    graph_aliases_for,
    graph_arg,
    graph_choices,
)
from core.utils import (
    format_model_display_name,
    get_ablation_model_names,
    get_model_base_name,
    is_finetuned_model_name,
)
from evaluation.run_budget_tradeoff import (
    BUDGETS as TRADEOFF_BUDGETS,
    MODEL_NAMES as TRADEOFF_MODEL_NAMES,
    PLOT_PDF_PATH as TRADEOFF_PDF_PATH,
    PLOT_PNG_PATH as TRADEOFF_PNG_PATH,
    RESULTS_CSV_PATH as TRADEOFF_RESULTS_PATH,
    validate_aggregated_results as validate_tradeoff_results,
)

# -------------------------------------------------------------------------
# Paths / global config
# -------------------------------------------------------------------------

PLOT_OUTPUT_DIR = PLOTS_DIR
EVALUATION_INPUT_ROOT = EVALUATION_DIR

# Save vector plots for thesis.
# "pdf" is best for LaTeX. Add "png" too if you want preview images.
PLOT_FORMATS = ["pdf", "png"]
PNG_DPI = 300

BASE_MODEL_LABELS = {
    "all-mpnet-base-v2": format_model_display_name("all-mpnet-base-v2"),
    "bge-large-en-v1.5": format_model_display_name("bge-large-en-v1.5"),
    "mxbai-embed-large-v1": format_model_display_name("mxbai-embed-large-v1"),
    "Qwen3-Embedding-0.6B": format_model_display_name("Qwen3-Embedding-0.6B"),
    "Qwen3-Embedding-4B": format_model_display_name("Qwen3-Embedding-4B"),
    "granite-embedding-small-english-r2": format_model_display_name(
        "granite-embedding-small-english-r2"
    ),
    "granite-embedding-english-r2": format_model_display_name(
        "granite-embedding-english-r2"
    ),
    BFS_UNCAPPED_BASELINE_MODEL: format_model_display_name(
        BFS_UNCAPPED_BASELINE_MODEL
    ),
    BFS_CAPPED_BASELINE_MODEL: format_model_display_name(
        BFS_CAPPED_BASELINE_MODEL
    ),
    RL_BASELINE_MODEL: format_model_display_name(RL_BASELINE_MODEL),
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
        nargs="?",
        help=(
            "Dataset name or dataset path, e.g. "
            "msmarco_valid or data/datasets/filtered/msmarco_valid_filtered.json"
        ),
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help=(
            "Run suffix used in evaluation and plot paths, e.g. v3. "
            "Defaults to v3 for single-dataset mode. In --all mode, "
            "omitting this includes every run suffix."
        ),
    )
    parser.add_argument(
        "--graph",
        type=graph_arg,
        choices=graph_choices(),
        default=None,
        help=(
            "Graph results to visualize. Defaults to CauseNet for "
            "single-dataset mode. In --all mode, omitting this includes "
            "every graph."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Generate plots for every result directory under data/evaluation. "
            "Can be filtered with --graph and/or --run-suffix."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Visualize ablation results from data/evaluation/ablation and "
            "compare them with the original ReLU+Euclid result from the "
            "standard evaluation directory. Baselines are omitted."
        ),
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help=(
            "Create the fixed thesis figures for CauseNet/msmarco_valid/v3 "
            "using only BFS/RL baselines and fine-tuned A* models. Outputs "
            "are saved under data/plots/thesis."
        ),
    )
    parser.add_argument(
        "--tradeoff",
        action="store_true",
        help=(
            "Create the budget trade-off figure from "
            "data/evaluation/budget_tradeoff/budget_tradeoff_results.csv."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Matryoshka dimension to compare in --ablation mode (default: 32).",
    )
    args = parser.parse_args()

    if not args.all and not args.dataset and not args.thesis and not args.tradeoff:
        parser.error("dataset is required unless --all, --thesis, or --tradeoff is set")

    if args.tradeoff:
        incompatible = []
        if args.dataset:
            incompatible.append("dataset")
        if args.all:
            incompatible.append("--all")
        if args.ablation:
            incompatible.append("--ablation")
        if args.thesis:
            incompatible.append("--thesis")
        if args.graph:
            incompatible.append("--graph")
        if args.run_suffix:
            incompatible.append("--run-suffix")
        if incompatible:
            parser.error(
                "--tradeoff is an independent mode and cannot be combined with: "
                + ", ".join(incompatible)
            )

    if args.thesis:
        if args.all or args.ablation:
            parser.error("--thesis cannot be combined with --all or --ablation")
        if args.dataset and dataset_name_from_arg(args.dataset) != "msmarco_valid":
            parser.error("--thesis is fixed to the msmarco_valid dataset")
        if args.graph and args.graph != "causenet":
            parser.error("--thesis is fixed to the causenet graph")
        if args.run_suffix and args.run_suffix != "v3":
            parser.error("--thesis is fixed to run suffix v3")

    return args


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


def build_input_paths(
    dataset_name: str,
    run_suffix: str,
    graph_name: str,
    ablation: bool = False,
):
    graph_name = canonical_graph_name(graph_name)
    input_root = EVALUATION_INPUT_ROOT / "ablation" if ablation else EVALUATION_INPUT_ROOT
    candidate_dirs = [
        input_root / graph_dir / dataset_name / run_suffix
        for graph_dir in (graph_name, *graph_aliases_for(graph_name))
    ]
    eval_dir = next(
        (
            path
            for path in candidate_dirs
            if (path / "evaluation_results.json").exists()
            or (path / "visited_nodes_analysis.json").exists()
        ),
        candidate_dirs[0],
    )
    eval_results_path = eval_dir / "evaluation_results.json"
    visited_nodes_path = eval_dir / "visited_nodes_analysis.json"

    return eval_results_path, visited_nodes_path


def build_plot_output_dir(
    dataset_name: str,
    run_suffix: str,
    graph_name: str,
    ablation: bool = False,
):
    graph_name = canonical_graph_name(graph_name)
    plot_root = PLOT_OUTPUT_DIR / "ablation" if ablation else PLOT_OUTPUT_DIR
    return plot_root / graph_name / dataset_name / run_suffix


def discover_result_sets(graph_name=None, run_suffix=None, ablation=False):
    if graph_name is not None:
        graph_name = canonical_graph_name(graph_name)

    result_sets = []

    input_root = EVALUATION_INPUT_ROOT / "ablation" if ablation else EVALUATION_INPUT_ROOT

    if not input_root.exists():
        return result_sets

    graph_dirs = []

    if graph_name is None:
        graph_dirs = [
            path
            for path in input_root.iterdir()
            if path.is_dir()
        ]
    else:
        graph_dirs = [
            input_root / graph_dir
            for graph_dir in (graph_name, *graph_aliases_for(graph_name))
        ]

    for graph_dir in sorted(graph_dirs):
        if not graph_dir.is_dir():
            continue

        for dataset_dir in sorted(path for path in graph_dir.iterdir() if path.is_dir()):
            run_dirs = []

            if run_suffix is None:
                run_dirs = [
                    path
                    for path in dataset_dir.iterdir()
                    if path.is_dir()
                ]
            else:
                run_dirs = [dataset_dir / run_suffix]

            for run_dir in sorted(run_dirs):
                if not run_dir.is_dir():
                    continue

                eval_results_path = run_dir / "evaluation_results.json"
                visited_nodes_path = run_dir / "visited_nodes_analysis.json"

                if not eval_results_path.exists() and not visited_nodes_path.exists():
                    continue

                result_sets.append(
                    {
                        "graph": canonical_graph_name(graph_dir.name),
                        "dataset": dataset_dir.name,
                        "run_suffix": run_dir.name,
                        "eval_results_path": eval_results_path,
                        "visited_nodes_path": visited_nodes_path,
                    }
                )

    return result_sets


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


def dimension_matches(value, expected_dim):
    if value is None or pd.isna(value):
        return expected_dim is None

    try:
        return int(value) == int(expected_dim)
    except (TypeError, ValueError):
        return False


def filter_entries_by_model_and_dim(entries, model_names, dim):
    model_names = set(model_names)

    return [
        entry for entry in entries
        if entry.get("model") in model_names
        and dimension_matches(entry.get("dimension"), dim)
    ]


def load_optional_json(path):
    if not path.exists():
        print(f"Optional file missing: {path}")
        return []

    return load_json(path)


def get_ablation_variant_label(model_name):
    name = Path(str(model_name)).name
    parts = name.removesuffix("_finetuned").split("_")

    activation = "ReLU" if "relu" in parts else "GELU" if "gelu" in parts else "?"
    distance = "Euclid" if "euclid" in parts else "Cosine" if "cosine" in parts else "?"
    suffix = "main" if "ablation" not in parts else "ablation"

    return f"{activation} + {distance}\n{suffix}"


def load_ablation_comparison_entries(
    dataset_name,
    run_suffix,
    graph_name,
    dim,
    filename,
):
    ablation_path = (
        EVALUATION_INPUT_ROOT
        / "ablation"
        / graph_name
        / dataset_name
        / run_suffix
        / filename
    )
    comparison_model_names = get_ablation_model_names(run_suffix)
    comparison_entries = filter_entries_by_model_and_dim(
        load_optional_json(ablation_path),
        comparison_model_names,
        dim,
    )

    found_models = {
        entry.get("model")
        for entry in comparison_entries
    }
    expected_models = set(comparison_model_names)
    missing_models = sorted(expected_models - found_models, key=model_sort_key)

    if missing_models:
        print(
            "Warning: missing ablation comparison rows for "
            f"dim {dim}: {missing_models}"
        )

    return comparison_entries, ablation_path


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

def adjust_color(color, amount):
    rgb = np.array(mcolors.to_rgb(color))

    if amount >= 0:
        adjusted = rgb + (1.0 - rgb) * amount
    else:
        adjusted = rgb * (1.0 + amount)

    return mcolors.to_hex(np.clip(adjusted, 0.0, 1.0))


def parse_model_label(model_name):
    return format_model_display_name(model_name)


def model_sort_key(model_name):
    name = model_name.removesuffix("_finetuned").removesuffix("_best")
    parts = name.split("_")
    base_name = get_model_base_name(model_name)

    base_order = {
        BFS_UNCAPPED_BASELINE_MODEL: -3,
        BFS_CAPPED_BASELINE_MODEL: -2,
        RL_BASELINE_MODEL: -1,
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

        if model not in BASELINE_MODEL_NAMES:
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

        if model_name in BASELINE_MODEL_NAMES:
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
    plot_selected_baseline_lines_on_axis(
        ax=ax,
        baselines=baselines,
        metric_key=metric_key,
        selected_names=get_valid_baseline_names(baselines, metric_key),
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

    if is_finetuned_model_name(model_name):
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

BASELINE_LINE_STYLES_BY_MODEL = {
    BFS_UNCAPPED_BASELINE_MODEL: {
        "color": "#111111",
        "linestyle": "--",
        "linewidth": 1.2,
        "zorder": 3,
    },
    BFS_CAPPED_BASELINE_MODEL: {
        "color": "#777777",
        "linestyle": "-.",
        "linewidth": 1.2,
        "zorder": 3,
    },
    RL_BASELINE_MODEL: {
        "color": "#B00020",
        "linestyle": ":",
        "linewidth": 1.4,
        "zorder": 3,
    },
}


def get_baseline_line_style(baselines, baseline_name):
    if baseline_name in BASELINE_LINE_STYLES_BY_MODEL:
        return BASELINE_LINE_STYLES_BY_MODEL[baseline_name]

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


def get_baseline_value_extent(baselines, metric_key, selected_names):
    values = [
        float(baselines[baseline_name][metric_key])
        for baseline_name in selected_names
    ]

    return min(values), max(values)


def constrain_panel_y_ranges(panels):
    """Prevent padded broken-axis panels from overlapping in source space."""
    if len(panels) < 2:
        return panels

    constrained = [dict(panel) for panel in panels]
    ordered_indices = sorted(
        range(len(constrained)),
        key=lambda index: constrained[index]["ylim"][0],
    )

    for lower_index, upper_index in zip(ordered_indices[:-1], ordered_indices[1:]):
        lower_panel = constrained[lower_index]
        upper_panel = constrained[upper_index]

        lower_raw_max = lower_panel.get("raw_max")
        upper_raw_min = upper_panel.get("raw_min")

        if lower_raw_max is None or upper_raw_min is None:
            continue

        lower_raw_max = float(lower_raw_max)
        upper_raw_min = float(upper_raw_min)

        if not lower_raw_max < upper_raw_min:
            continue

        boundary = (lower_raw_max + upper_raw_min) / 2.0
        lower_min, lower_max = lower_panel["ylim"]
        upper_min, upper_max = upper_panel["ylim"]

        lower_panel["ylim"] = (lower_min, min(lower_max, boundary))
        upper_panel["ylim"] = (max(upper_min, boundary), upper_max)

    return constrained


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
                "raw_min": min(values),
                "raw_max": max(values),
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
                "raw_min": min(values),
                "raw_max": max(values),
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


def is_meaningful_axis_gap(lower_value, upper_value, gap, local_range, global_range):
    if gap <= 0 or local_range <= 0 or global_range <= 0:
        return False

    # Bounded ratio metrics such as F1/accuracy should usually stay on one axis.
    if 0 <= lower_value and upper_value <= 1:
        return False

    if gap / global_range < 0.04:
        return False

    if gap / local_range < 0.12:
        return False

    return True


def find_best_axis_split(values, global_range):
    if len(values) < 2:
        return None

    local_range = values[-1] - values[0]
    best_split = None

    for index in range(len(values) - 1):
        lower_value = values[index]
        upper_value = values[index + 1]
        gap = upper_value - lower_value

        if not is_meaningful_axis_gap(
            lower_value=lower_value,
            upper_value=upper_value,
            gap=gap,
            local_range=local_range,
            global_range=global_range,
        ):
            continue

        score = gap / global_range

        if best_split is None or score > best_split["score"]:
            best_split = {
                "index": index,
                "score": score,
            }

    return best_split


def split_values_into_axis_clusters(values, max_bands):
    values = sorted(
        float(value)
        for value in values
        if value is not None and not pd.isna(value)
    )

    if not values:
        return []

    if max_bands <= 1 or len(values) < 2:
        return [values]

    global_range = values[-1] - values[0]

    if global_range <= 0:
        return [values]

    clusters = [values]

    while len(clusters) < max_bands:
        best_cluster = None

        for cluster_index, cluster_values in enumerate(clusters):
            split = find_best_axis_split(cluster_values, global_range)

            if split is None:
                continue

            if best_cluster is None or split["score"] > best_cluster["score"]:
                best_cluster = {
                    "cluster_index": cluster_index,
                    "split_index": split["index"],
                    "score": split["score"],
                }

        if best_cluster is None:
            break

        cluster_index = best_cluster["cluster_index"]
        split_index = best_cluster["split_index"]
        cluster_values = clusters[cluster_index]
        lower_cluster = cluster_values[: split_index + 1]
        upper_cluster = cluster_values[split_index + 1 :]

        if not lower_cluster or not upper_cluster:
            break

        clusters = (
            clusters[:cluster_index]
            + [lower_cluster, upper_cluster]
            + clusters[cluster_index + 1 :]
        )

    return clusters


def values_in_cluster(values, raw_min, raw_max):
    epsilon = max(abs(raw_min), abs(raw_max), 1.0) * 1e-9

    return [
        float(value)
        for value in values
        if raw_min - epsilon <= float(value) <= raw_max + epsilon
    ]


def build_clustered_metric_panels(df, baselines, metric_key, max_bands=3):
    model_values = get_model_metric_values(df, metric_key)
    baseline_names = get_valid_baseline_names(baselines, metric_key)
    baseline_values = {
        baseline_name: float(baselines[baseline_name][metric_key])
        for baseline_name in baseline_names
    }
    all_values = model_values + list(baseline_values.values())

    clusters = split_values_into_axis_clusters(all_values, max_bands=max_bands)

    if not clusters:
        return [], False

    panels = []

    for cluster_values in clusters:
        raw_min = min(cluster_values)
        raw_max = max(cluster_values)
        cluster_model_values = values_in_cluster(model_values, raw_min, raw_max)
        cluster_baselines = [
            baseline_name
            for baseline_name, value in baseline_values.items()
            if values_in_cluster([value], raw_min, raw_max)
        ]
        contains_model = bool(cluster_model_values)

        panels.append(
            {
                "kind": "focus" if contains_model else "baseline",
                "baselines": cluster_baselines,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "ylim": compute_values_ylim(
                    cluster_values,
                    pad_ratio=0.18 if contains_model else 0.12,
                ),
                "height": 3.0 if contains_model else 0.42,
            }
        )

    return constrain_panel_y_ranges(panels), len(panels) > 1


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


def get_piecewise_tick_count(band, max_ticks_per_band):
    if max_ticks_per_band <= 1:
        return 1

    display_height = band["display_max"] - band["display_min"]

    if display_height <= 0.6:
        return 1
    if display_height <= 1.6:
        return min(2, max_ticks_per_band)

    return max_ticks_per_band


def get_piecewise_tick_fractions(band_index, band_count, tick_count):
    has_lower_gap = band_index > 0
    has_upper_gap = band_index < band_count - 1

    lower_fraction = 0.0 if not has_lower_gap else 0.15
    upper_fraction = 1.0 if not has_upper_gap else 0.85

    if tick_count == 1:
        return [(lower_fraction + upper_fraction) / 2.0]

    return np.linspace(lower_fraction, upper_fraction, tick_count).tolist()


def get_piecewise_ticks(bands, gap_height, max_ticks_per_band=4):
    tick_values = []

    for band_index, band in enumerate(bands):
        source_min = band["source_min"]
        source_max = band["source_max"]
        tick_count = get_piecewise_tick_count(band, max_ticks_per_band)
        fractions = get_piecewise_tick_fractions(
            band_index,
            len(bands),
            tick_count,
        )

        if source_max == source_min:
            tick_values.append(float(source_min))
            continue

        for fraction in fractions:
            tick_values.append(
                float(source_min + (source_max - source_min) * fraction)
            )

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
    return build_clustered_metric_panels(
        df=df,
        baselines=baselines,
        metric_key=metric_key,
        max_bands=max_focus_bands,
    )


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


def plot_ablation_metric_summary_bars(df, output_path):
    if df.empty:
        print("No ablation semantic data. Skipping ablation bar summary.")
        return

    metrics = [
        ("f1_score", "F1 Score"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("avg_nodes_visited", "Avg Visited Nodes"),
        ("avg_time_ms", "Avg Runtime (ms)"),
    ]

    plot_df = df.copy()
    plot_df["sort_key"] = plot_df["model"].astype(str).map(model_sort_key)
    plot_df = plot_df.sort_values("sort_key").drop(columns=["sort_key"])

    labels = [
        get_ablation_variant_label(model)
        for model in plot_df["model"].astype(str).tolist()
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    axes = axes.flatten()

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        if metric_key not in plot_df.columns:
            ax.axis("off")
            continue

        values = plot_df[metric_key].astype(float).tolist()
        ax.bar(range(len(values)), values, color=colors[:len(values)])
        ax.set_title(metric_label)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.grid(True, axis="y", alpha=0.3)

        if metric_key in {"f1_score", "accuracy", "precision", "recall"}:
            ax.set_ylim(0, 1.05)

        for index, value in enumerate(values):
            ax.text(
                index,
                value,
                f"{value:.3f}" if metric_key in {"f1_score", "accuracy", "precision", "recall"} else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Granite Activation/Distance Ablation", fontsize=14, fontweight="bold")
    save_plot(fig, output_path)
    plt.close(fig)


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

def get_p95_model_label(model):
    if model == BFS_CAPPED_BASELINE_MODEL:
        return BASE_MODEL_LABELS[BFS_UNCAPPED_BASELINE_MODEL]

    return parse_model_label(model)


def get_baseline_default_algorithm(model):
    if model in {BFS_UNCAPPED_BASELINE_MODEL, BFS_CAPPED_BASELINE_MODEL}:
        return "BFS"

    if model == RL_BASELINE_MODEL:
        return "RL"

    return model.replace("_Baseline", "")


def extract_p95_data(visited_nodes_data):
    rows = []

    for entry in visited_nodes_data:
        model = entry.get("model")
        dimension = entry.get("dimension")
        analysis = entry.get("analysis", {})

        if model is None:
            continue

        if model in BASELINE_MODEL_NAMES:
            algorithm = analysis.get(
                "strategy",
                get_baseline_default_algorithm(model),
            )
            model_label = get_p95_model_label(model)
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
    df = df[~df["model"].isin(BASELINE_MODEL_NAMES)]
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
    # - one for each baseline with visited-node data
    # - one per A* model and Matryoshka dimension
    plot_individual_visited_node_histograms(
        p95_df=p95_df,
        plot_root=plot_root,
    )


def plot_result_set(dataset_name, run_suffix, graph_name, eval_results_path, visited_nodes_path):
    current_split = detect_split(dataset_name)
    plot_root = build_plot_output_dir(
        dataset_name,
        run_suffix,
        graph_name,
    )

    ensure_directory(plot_root)

    print(f"Dataset: {dataset_name}")
    print(f"Split: {current_split}")
    print(f"Graph: {graph_name}")
    print(f"Run suffix: {run_suffix}")
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
    # - p95_analysis/individual_histograms/BFS_Uncapped_Baseline/baseline/histogram.pdf
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


def plot_ablation_result_set(dataset_name, run_suffix, graph_name, dim):
    current_split = detect_split(dataset_name)
    plot_root = build_plot_output_dir(
        dataset_name,
        run_suffix,
        graph_name,
        ablation=True,
    )
    ensure_directory(plot_root)

    print(f"Dataset: {dataset_name}")
    print(f"Split: {current_split}")
    print(f"Graph: {graph_name}")
    print(f"Run suffix: {run_suffix}")
    print(f"Ablation dim: {dim}")
    print(f"Plot output dir: {plot_root}")
    print(f"Plot formats: {PLOT_FORMATS}")

    eval_data, ablation_eval_path = load_ablation_comparison_entries(
        dataset_name=dataset_name,
        run_suffix=run_suffix,
        graph_name=graph_name,
        dim=dim,
        filename="evaluation_results.json",
    )

    print(f"Joint four-model ablation results: {ablation_eval_path}")

    if eval_data:
        df = extract_semantic_data(eval_data)
        baselines = {}
        per_example_df = extract_per_example_rows(eval_data)
        if not per_example_df.empty:
            per_example_df = per_example_df[per_example_df["algorithm"] == "A*"]

        plot_all_standard_metrics(
            df=df,
            baselines=baselines,
            plot_root=plot_root,
        )

        plot_ablation_metric_summary_bars(
            df=df,
            output_path=get_plot_path(
                plot_root,
                "ablation_summary",
                "activation_distance_ablation.png",
            ),
        )

        plot_normalized_confusion_matrices_all_models(
            per_example_df=per_example_df,
            output_path=get_plot_path(
                plot_root,
                "confusion_matrix",
                "confusion_matrices_ablation_models_normalized.png",
            ),
        )
    else:
        print("No ablation comparison evaluation rows found. Skipping metric plots.")

    print("\nAll available ablation plots created.")


# -------------------------------------------------------------------------
# Fixed thesis figure: CauseNet / MS MARCO validation / v3
# -------------------------------------------------------------------------

# This mode deliberately reads one standard evaluation result file only. It
# never reads data/evaluation/ablation or visited_nodes_analysis.json.
THESIS_RESULTS_PATH = (
    EVALUATION_DIR
    / "causenet"
    / "msmarco_valid"
    / "v3"
    / "evaluation_results.json"
)
THESIS_OUTPUT_DIR = THESIS_PLOTS_DIR
THESIS_OUTPUT_STEM = "causenet_msmarco_valid_thesis"
THESIS_PNG_DPI = 400
THESIS_RESULT_METRIC_KEYS = (
    "f1_score",
    "accuracy",
    "avg_nodes_visited",
    "avg_time_ms",
)
THESIS_BUDGET_KEY = "search_budget"

# Both BFS configurations are retained because both are present as distinct
# baselines. RL is included when its row is available.
THESIS_BASELINE_MODELS = (
    BFS_UNCAPPED_BASELINE_MODEL,
    BFS_CAPPED_BASELINE_MODEL,
    RL_BASELINE_MODEL,
)
THESIS_BASELINE_ALGORITHMS = {
    BFS_UNCAPPED_BASELINE_MODEL: "BFS",
    BFS_CAPPED_BASELINE_MODEL: "BFS",
    RL_BASELINE_MODEL: "RL",
}

# Fixed whitelist: unfine-tuned and unrelated fine-tuned/ablation models can
# never enter the thesis figure merely because new rows appear in the JSON.
THESIS_FINETUNED_ASTAR_MODELS = (
    "all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v3_finetuned",
    "bge-large-en-v1.5_relu_euclid_nonorm_matryoshka_v3_finetuned",
    "mxbai-embed-large-v1_relu_euclid_nonorm_matryoshka_v3_finetuned",
    "Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_v3_finetuned",
    "granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v3_finetuned",
)
THESIS_SYSTEM_ORDER = THESIS_BASELINE_MODELS + THESIS_FINETUNED_ASTAR_MODELS

THESIS_DISPLAY_LABELS = {
    BFS_UNCAPPED_BASELINE_MODEL: "BFS (uncapped)",
    BFS_CAPPED_BASELINE_MODEL: "BFS (capped)",
    RL_BASELINE_MODEL: "RL baseline",
    THESIS_FINETUNED_ASTAR_MODELS[0]: "FT A*: MPNet",
    THESIS_FINETUNED_ASTAR_MODELS[1]: "FT A*: BGE",
    THESIS_FINETUNED_ASTAR_MODELS[2]: "FT A*: mxbai",
    THESIS_FINETUNED_ASTAR_MODELS[3]: "FT A*: Qwen",
    THESIS_FINETUNED_ASTAR_MODELS[4]: "FT A*: Granite",
}

THESIS_SYSTEM_STYLES = {
    BFS_UNCAPPED_BASELINE_MODEL: {
        "color": "#111111",
        "linestyle": "--",
        "linewidth": 1.6,
    },
    BFS_CAPPED_BASELINE_MODEL: {
        "color": "#666666",
        "linestyle": "-.",
        "linewidth": 1.6,
    },
    RL_BASELINE_MODEL: {
        "color": "#9A9A9A",
        "linestyle": ":",
        "linewidth": 1.9,
    },
    THESIS_FINETUNED_ASTAR_MODELS[0]: {
        "color": "#0072B2",
        "marker": "o",
    },
    THESIS_FINETUNED_ASTAR_MODELS[1]: {
        "color": "#D55E00",
        "marker": "s",
    },
    THESIS_FINETUNED_ASTAR_MODELS[2]: {
        "color": "#009E73",
        "marker": "^",
    },
    THESIS_FINETUNED_ASTAR_MODELS[3]: {
        "color": "#CC79A7",
        "marker": "D",
    },
    THESIS_FINETUNED_ASTAR_MODELS[4]: {
        "color": "#E69F00",
        "marker": "P",
    },
}


def make_thesis_metric_row(model, dimension, algorithm, result, used_config):
    metrics = result.get("metrics", {})
    if algorithm == "A*":
        search_budget = used_config.get("astar_max_visits")
    elif algorithm == "BFS":
        search_budget = used_config.get("bfs_max_visits")
    else:
        search_budget = None

    return {
        "model": model,
        "dimension": dimension,
        "algorithm": algorithm,
        THESIS_BUDGET_KEY: search_budget,
        **{key: metrics.get(key) for key in THESIS_RESULT_METRIC_KEYS},
    }


def thesis_row_sort_key(row):
    model_position = THESIS_SYSTEM_ORDER.index(row["model"])
    dimension = row["dimension"] if row["dimension"] is not None else -1
    return model_position, dimension


def load_thesis_selected_rows():
    """Read the fixed result file and return only approved thesis systems."""

    eval_data = load_json(THESIS_RESULTS_PATH)
    if not isinstance(eval_data, list):
        raise ValueError(f"Expected a JSON list in {THESIS_RESULTS_PATH}")

    rows = []
    excluded_systems = {}

    for entry in eval_data:
        model = entry.get("model")
        evaluations = entry.get("evaluation", {})
        used_config = entry.get("used_config", {})

        if model in THESIS_BASELINE_MODELS:
            algorithm = THESIS_BASELINE_ALGORITHMS[model]
            result = evaluations.get(algorithm)
            if result is None:
                raise ValueError(
                    f"{model} has no {algorithm!r} evaluation in "
                    f"{THESIS_RESULTS_PATH}"
                )
            rows.append(
                make_thesis_metric_row(
                    model, None, algorithm, result, used_config
                )
            )
            continue

        if model in THESIS_FINETUNED_ASTAR_MODELS:
            result = evaluations.get("A*")
            if result is None:
                raise ValueError(f"Whitelisted model {model} has no A* evaluation")
            dimension = entry.get("dimension")
            if dimension is None:
                raise ValueError(f"Whitelisted model {model} has no dimension")
            rows.append(
                make_thesis_metric_row(
                    model, int(dimension), "A*", result, used_config
                )
            )
            continue

        if model:
            if is_finetuned_model_name(model):
                reason = "fine-tuned run outside the thesis whitelist"
            else:
                reason = "unfine-tuned or non-target system"
            excluded_systems[str(model)] = reason

    validate_thesis_rows(rows)
    rows.sort(key=thesis_row_sort_key)
    return rows, excluded_systems


def validate_thesis_rows(rows):
    present_models = {row["model"] for row in rows}

    # RL is optional. Both BFS variants and all five fine-tuned A* families
    # are required to avoid silently producing a partial thesis figure.
    required_models = (
        BFS_UNCAPPED_BASELINE_MODEL,
        BFS_CAPPED_BASELINE_MODEL,
        *THESIS_FINETUNED_ASTAR_MODELS,
    )
    missing_models = [
        model for model in required_models if model not in present_models
    ]
    if missing_models:
        raise ValueError(f"Missing expected thesis systems: {missing_models}")

    seen_runs = set()
    for row in rows:
        run_key = (row["model"], row["dimension"], row["algorithm"])
        if run_key in seen_runs:
            raise ValueError(f"Duplicate selected evaluation row: {run_key}")
        seen_runs.add(run_key)

        missing_metrics = [
            key for key in THESIS_RESULT_METRIC_KEYS if row[key] is None
        ]
        if missing_metrics:
            raise ValueError(f"Missing metrics {missing_metrics} for {run_key}")
        if row["algorithm"] == "A*" and row[THESIS_BUDGET_KEY] is None:
            raise ValueError(f"Missing p95 search budget for {run_key}")


def group_thesis_rows_by_model(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    return grouped


def get_thesis_dimensions(rows):
    return sorted(
        {
            int(row["dimension"])
            for row in rows
            if row["dimension"] is not None
        }
    )


def set_zoomed_f1_axis(ax, rows):
    """Zoom F1 transparently; line charts do not require a zero baseline."""

    values = [float(row["f1_score"]) for row in rows]
    tick_step = 0.05
    lower = max(
        0.0,
        math.floor((min(values) - 0.01) / tick_step) * tick_step,
    )
    upper = min(
        1.0,
        math.ceil((max(values) + 0.01) / tick_step) * tick_step,
    )
    if lower >= upper:
        lower = max(0.0, min(values) - tick_step)
        upper = min(1.0, max(values) + tick_step)

    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_thesis_metric_panel(
    ax,
    rows,
    metric_key,
    title,
    panel_label,
    log_scale=False,
):
    grouped = group_thesis_rows_by_model(rows)
    dimensions = get_thesis_dimensions(rows)
    x_positions = {dimension: index for index, dimension in enumerate(dimensions)}

    for draw_index, model in enumerate(THESIS_FINETUNED_ASTAR_MODELS):
        model_rows = grouped[model]
        style = THESIS_SYSTEM_STYLES[model]
        ax.plot(
            [x_positions[row["dimension"]] for row in model_rows],
            [float(row[metric_key]) for row in model_rows],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.75,
            markersize=6.0,
            linewidth=1.8,
            zorder=3 + draw_index * 0.05,
            **style,
        )

    for model in THESIS_BASELINE_MODELS:
        if model not in grouped:
            continue
        raw_baseline_value = grouped[model][0][metric_key]
        if raw_baseline_value is None:
            continue
        baseline_value = float(raw_baseline_value)
        if log_scale and baseline_value <= 0:
            continue
        ax.axhline(
            baseline_value,
            zorder=2,
            **THESIS_SYSTEM_STYLES[model],
        )

    ax.set_title(
        f"{panel_label}  {title}",
        loc="left",
        fontsize=14,
        fontweight="semibold",
    )
    ax.set_xlabel("Embedding dimension", fontsize=12)
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([str(dimension) for dimension in dimensions], rotation=35)
    ax.tick_params(axis="both", which="major", labelsize=11.5)
    ax.grid(axis="y", which="major", color="#D2D2D2", linewidth=0.65)
    ax.grid(axis="y", which="minor", color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)

    if metric_key == "f1_score":
        set_zoomed_f1_axis(ax, rows)
    elif metric_key == "accuracy":
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    elif log_scale:
        ax.set_yscale("log")
        positive_values = [
            float(row[metric_key])
            for row in rows
            if row[metric_key] is not None and float(row[metric_key]) > 0
        ]
        ax.set_ylim(min(positive_values) / 1.3, max(positive_values) * 1.5)

def build_thesis_legend_handles(rows):
    present_models = {row["model"] for row in rows}
    handles = []

    for model in THESIS_SYSTEM_ORDER:
        if model not in present_models:
            continue
        style = dict(THESIS_SYSTEM_STYLES[model])
        if model in THESIS_FINETUNED_ASTAR_MODELS:
            style.update(
                {
                    "linestyle": "-",
                    "linewidth": 1.6,
                    "markersize": 5.4,
                    "markeredgecolor": style["color"],
                    "markeredgewidth": 0.75,
                }
            )
        handles.append(
            Line2D([0], [0], label=THESIS_DISPLAY_LABELS[model], **style)
        )

    return handles


def save_thesis_figure(fig, stem):
    THESIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for suffix in (".png", ".pdf"):
        path = THESIS_OUTPUT_DIR / f"{stem}{suffix}"
        save_options = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == ".png":
            save_options["dpi"] = THESIS_PNG_DPI
        fig.savefig(path, **save_options)
        saved_paths.append(path)

    return saved_paths


def create_thesis_main_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))

    plot_thesis_metric_panel(
        axes[0, 0],
        rows,
        "f1_score",
        r"F$_1$ score",
        "(a)",
    )

    plot_thesis_metric_panel(
        axes[0, 1],
        rows,
        THESIS_BUDGET_KEY,
        "p95 budget τ",
        "(b)",
        log_scale=True,
    )

    plot_thesis_metric_panel(
        axes[1, 0],
        rows,
        "avg_nodes_visited",
        "Search effort (visited nodes)",
        "(c)",
        log_scale=True,
    )

    plot_thesis_metric_panel(
        axes[1, 1],
        rows,
        "avg_time_ms",
        "Runtime (ms)",
        "(d)",
        log_scale=True,
    )

    fig.legend(
        handles=build_thesis_legend_handles(rows),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.3,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.99), h_pad=1.5, w_pad=1.0)
    paths = save_thesis_figure(fig, f"{THESIS_OUTPUT_STEM}_main")
    plt.close(fig)
    return paths


def format_thesis_number(value, decimals):
    if not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def print_thesis_selection_report(rows, excluded_systems):
    print(f"Input evaluation file: {THESIS_RESULTS_PATH}")
    print("\nIncluded systems (in legend order):")
    present_models = {row["model"] for row in rows}

    for model in THESIS_SYSTEM_ORDER:
        if model not in present_models:
            continue
        run_type = (
            "baseline" if model in THESIS_BASELINE_MODELS else "fine-tuned A*"
        )
        print(f"  - {THESIS_DISPLAY_LABELS[model]} [{run_type}]\n    {model}")

    print("\nIncluded metric values:")
    header = (
        f"{'System':<24} {'Dim':>5} {'Alg.':>5} {'F1':>7} {'Acc.':>7} "
        f"{'Visited':>10} {'Runtime ms':>11}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        dimension = "-" if row["dimension"] is None else str(row["dimension"])
        print(
            f"{THESIS_DISPLAY_LABELS[row['model']]:<24} "
            f"{dimension:>5} "
            f"{row['algorithm']:>5} "
            f"{format_thesis_number(row['f1_score'], 4):>7} "
            f"{format_thesis_number(row['accuracy'], 4):>7} "
            f"{format_thesis_number(row['avg_nodes_visited'], 2):>10} "
            f"{format_thesis_number(row['avg_time_ms'], 3):>11}"
        )

    print("\nExcluded systems present in the input file:")
    for model, reason in sorted(excluded_systems.items()):
        print(f"  - {model}: {reason}")


def plot_thesis_causenet_msmarco_valid():
    """Create the fixed 2-by-2 thesis figure."""

    apply_thesis_plot_style()
    rows, excluded_systems = load_thesis_selected_rows()
    print_thesis_selection_report(rows, excluded_systems)

    output_paths = create_thesis_main_figure(rows)

    print("\nCreated thesis figures:")
    for path in output_paths:
        print(f"  - {path}")


# -------------------------------------------------------------------------
# Fixed budget trade-off figure: CauseNet / MS MARCO validation / v3
# -------------------------------------------------------------------------

TRADEOFF_PNG_DPI = 400
TRADEOFF_ANNOTATION_OFFSETS = (
    (0, 9),
    (0, -12),
    (8, 7),
    (-8, -11),
    (9, -8),
    (-9, 8),
)


def load_budget_tradeoff_rows():
    """Load and strictly validate the expected five-by-ten result matrix."""

    if not TRADEOFF_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "Budget trade-off results do not exist. Run "
            "`python -m evaluation.run_budget_tradeoff` first. "
            f"Expected: {TRADEOFF_RESULTS_PATH}"
        )

    tradeoff_df = pd.read_csv(TRADEOFF_RESULTS_PATH, sep=";")
    rows = validate_tradeoff_results(tradeoff_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def get_tradeoff_style(model_name):
    checkpoint_name = next(
        checkpoint
        for checkpoint, display_name in THESIS_DISPLAY_LABELS.items()
        if display_name == model_name
    )
    return THESIS_SYSTEM_STYLES[checkpoint_name]


def annotate_tradeoff_budgets(ax, line_index, subset, color):
    """Use staggered offsets so nearby budget labels remain distinguishable."""

    for point_index, row in enumerate(subset.itertuples(index=False)):
        offset_index = (point_index + line_index * 2) % len(
            TRADEOFF_ANNOTATION_OFFSETS
        )
        x_offset, y_offset = TRADEOFF_ANNOTATION_OFFSETS[offset_index]
        ax.annotate(
            rf"$\tau={int(row.budget)}$",
            xy=(row.average_visited_nodes, row.f1),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.5,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            annotation_clip=True,
            zorder=8,
        )


def create_budget_tradeoff_figure(tradeoff_df):
    """Plot measured search effort against F1 in ascending budget order."""

    apply_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(10.2, 6.6))

    for line_index, model_name in enumerate(TRADEOFF_MODEL_NAMES):
        subset = tradeoff_df[tradeoff_df["model"] == model_name].sort_values(
            "budget"
        )
        style = get_tradeoff_style(model_name)
        ax.plot(
            subset["average_visited_nodes"],
            subset["f1"],
            label=model_name,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.2,
            linewidth=1.8,
            zorder=3 + line_index * 0.05,
        )
        annotate_tradeoff_budgets(ax, line_index, subset, style["color"])

    f1_values = tradeoff_df["f1"].astype(float)
    f1_range = max(float(f1_values.max() - f1_values.min()), 0.02)
    lower = max(0.0, float(f1_values.min()) - 0.14 * f1_range)
    upper = min(1.0, float(f1_values.max()) + 0.14 * f1_range)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Average Visited Nodes", fontsize=12)
    ax.set_ylabel(r"F$_1$ Score", fontsize=12)
    ax.grid(axis="both", which="major", color="#D7D7D7", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.legend(
        loc="best",
        frameon=True,
        fontsize=9,
        handlelength=2.5,
        borderpad=0.7,
    )
    fig.tight_layout(pad=0.7)
    return fig


def plot_budget_tradeoff():
    tradeoff_df = load_budget_tradeoff_rows()
    fig = create_budget_tradeoff_figure(tradeoff_df)
    TRADEOFF_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        TRADEOFF_PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        TRADEOFF_PNG_PATH,
        dpi=TRADEOFF_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    print(
        "Created budget trade-off figures for "
        f"{len(TRADEOFF_MODEL_NAMES)} models and {len(TRADEOFF_BUDGETS)} budgets:"
    )
    print(f"  - {TRADEOFF_PDF_PATH}")
    print(f"  - {TRADEOFF_PNG_PATH}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.tradeoff:
        plot_budget_tradeoff()
        raise SystemExit(0)

    if args.thesis:
        plot_thesis_causenet_msmarco_valid()
        raise SystemExit(0)

    if args.dim <= 0:
        raise ValueError("--dim must be greater than 0")

    if args.all:
        result_sets = discover_result_sets(
            graph_name=args.graph,
            run_suffix=args.run_suffix,
            ablation=args.ablation,
        )

        if not result_sets:
            print("No evaluation result directories found.")
            raise SystemExit(0)

        print(f"Found {len(result_sets)} result directories.")

        for index, result_set in enumerate(result_sets, start=1):
            print(
                "\n"
                f"[{index}/{len(result_sets)}] "
                f"{result_set['graph']}/{result_set['dataset']}/"
                f"{result_set['run_suffix']}"
            )
            if args.ablation:
                plot_ablation_result_set(
                    dataset_name=result_set["dataset"],
                    run_suffix=result_set["run_suffix"],
                    graph_name=result_set["graph"],
                    dim=args.dim,
                )
            else:
                plot_result_set(
                    dataset_name=result_set["dataset"],
                    run_suffix=result_set["run_suffix"],
                    graph_name=result_set["graph"],
                    eval_results_path=result_set["eval_results_path"],
                    visited_nodes_path=result_set["visited_nodes_path"],
                )

        print("\nAll result directories processed.")
    else:
        dataset_name = dataset_name_from_arg(args.dataset)
        graph_name = args.graph or DEFAULT_GRAPH_NAME
        run_suffix = args.run_suffix or "v3"

        if args.ablation:
            plot_ablation_result_set(
                dataset_name=dataset_name,
                run_suffix=run_suffix,
                graph_name=graph_name,
                dim=args.dim,
            )
            raise SystemExit(0)

        eval_results_path, visited_nodes_path = build_input_paths(
            dataset_name,
            run_suffix,
            graph_name,
        )

        plot_result_set(
            dataset_name=dataset_name,
            run_suffix=run_suffix,
            graph_name=graph_name,
            eval_results_path=eval_results_path,
            visited_nodes_path=visited_nodes_path,
        )
