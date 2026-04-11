import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_DIR = "../data/evaluation/histogram_exports"
OUTPUT_DIR = "../data/plots/histograms_astar"

MODEL_DISPLAY_NAMES = {
    "relu_cosine": "ReLU + Cosine",
    "relu_euclid": "ReLU + Euclid",
    "gelu_cosine": "GELU + Cosine",
    "gelu_euclid": "GELU + Euclid",
}

TARGET_DIMS = [64, 768]
TARGET_MODELS = list(MODEL_DISPLAY_NAMES.keys())


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=";")


def get_input_file(model_key: str, dim: int) -> str:
    return os.path.join(INPUT_DIR, f"{model_key}_dim{dim}_astar_details.csv")


def plot_single_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_path: str,
    bins=None,
):
    plt.figure(figsize=(9, 6))
    plt.hist(df[column].dropna(), bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_single_model_histograms(model_key: str, dim: int, df: pd.DataFrame) -> None:
    display_name = MODEL_DISPLAY_NAMES[model_key]

    hop_max = int(df["hop_count"].max()) if not df.empty else 0
    hop_bins = range(0, hop_max + 2)

    nodes_max = int(df["nodes_visited"].max()) if not df.empty else 0
    nodes_bins = range(0, nodes_max + 2)

    plot_single_histogram(
        df=df,
        column="hop_count",
        title=f"Hop Count Histogram — {display_name} (dim={dim})",
        xlabel="Hop Count",
        output_path=os.path.join(
            OUTPUT_DIR, f"{model_key}_dim{dim}_hop_count_hist.png"
        ),
        bins=hop_bins,
    )

    plot_single_histogram(
        df=df,
        column="nodes_visited",
        title=f"Nodes Visited Histogram — {display_name} (dim={dim})",
        xlabel="Nodes Visited",
        output_path=os.path.join(
            OUTPUT_DIR, f"{model_key}_dim{dim}_nodes_visited_hist.png"
        ),
        bins=nodes_bins,
    )


def plot_combined_histogram(
    dataframes: list[tuple[str, pd.DataFrame]],
    column: str,
    title: str,
    xlabel: str,
    output_path: str,
    bins=None,
):
    plt.figure(figsize=(10, 6))

    for label, df in dataframes:
        plt.hist(
            df[column].dropna(),
            bins=bins,
            histtype="step",
            linewidth=2,
            label=label,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_combined_by_dimension(dim: int, loaded: dict[tuple[str, int], pd.DataFrame]) -> None:
    model_dfs = []
    hop_max = 0
    nodes_max = 0

    for model_key in TARGET_MODELS:
        df = loaded.get((model_key, dim))
        if df is None or df.empty:
            continue

        model_dfs.append((MODEL_DISPLAY_NAMES[model_key], df))
        hop_max = max(hop_max, int(df["hop_count"].max()))
        nodes_max = max(nodes_max, int(df["nodes_visited"].max()))

    if not model_dfs:
        return

    hop_bins = range(0, hop_max + 2)
    nodes_bins = range(0, nodes_max + 2)

    plot_combined_histogram(
        dataframes=model_dfs,
        column="hop_count",
        title=f"Hop Count Histogram Comparison (dim={dim})",
        xlabel="Hop Count",
        output_path=os.path.join(OUTPUT_DIR, f"combined_dim{dim}_hop_count_hist.png"),
        bins=hop_bins,
    )

    plot_combined_histogram(
        dataframes=model_dfs,
        column="nodes_visited",
        title=f"Nodes Visited Histogram Comparison (dim={dim})",
        xlabel="Nodes Visited",
        output_path=os.path.join(OUTPUT_DIR, f"combined_dim{dim}_nodes_visited_hist.png"),
        bins=nodes_bins,
    )


if __name__ == "__main__":
    ensure_directory(OUTPUT_DIR)

    loaded_data = {}

    for model_key in TARGET_MODELS:
        for dim in TARGET_DIMS:
            file_path = get_input_file(model_key, dim)

            if not Path(file_path).exists():
                print(f"Missing file, skipping: {file_path}")
                continue

            print(f"Loading: {file_path}")
            df = load_csv(file_path)
            loaded_data[(model_key, dim)] = df

            plot_single_model_histograms(model_key, dim, df)

    for dim in TARGET_DIMS:
        plot_combined_by_dimension(dim, loaded_data)

    print("Histogram plots created.")