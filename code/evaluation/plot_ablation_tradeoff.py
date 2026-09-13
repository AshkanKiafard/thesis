"""Plot the Granite-64 ablation effectiveness--efficiency trade-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from core.constants import EVALUATION_DIR
from core.utils import get_ablation_model_names
from evaluation.evaluation_viz import apply_thesis_plot_style


RUN_SUFFIX = "v4"
DIMENSION = 64
EXPECTED_BUDGET = 23
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT / "thesis" / "figures" / "ablation_tradeoff.pdf"
)

RESULT_SETS = (
    ("causenet", "msmarco_test", "CauseNet Prec. / MS MARCO"),
    ("causenet_full", "msmarco_test", "CauseNet Full / MS MARCO"),
    ("ceg", "msmarco_test", "CEG Filtered / MS MARCO"),
    ("causenet", "sem_test", "CauseNet Prec. / SemEval"),
    ("causenet_full", "sem_test", "CauseNet Full / SemEval"),
    ("ceg", "sem_test", "CEG Filtered / SemEval"),
)

# Okabe--Ito-inspired colors, reinforced by distinct markers for grayscale output.
VARIANT_STYLES = {
    "relu_euclid": {
        "label": "ReLU--Euclidean",
        "color": "#0072B2",
        "marker": "o",
    },
    "relu_cosine": {
        "label": "ReLU--Cosine",
        "color": "#56B4E9",
        "marker": "s",
    },
    "gelu_euclid": {
        "label": "GELU--Euclidean",
        "color": "#D55E00",
        "marker": "^",
    },
    "gelu_cosine": {
        "label": "GELU--Cosine",
        "color": "#E69F00",
        "marker": "D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Vector PDF output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def variant_key(model_name: str) -> str:
    for key in VARIANT_STYLES:
        if f"_{key}_" in model_name:
            return key
    raise ValueError(f"Cannot identify ablation configuration: {model_name}")


def load_result_set(graph: str, dataset: str) -> dict[str, dict[str, float]]:
    path = (
        EVALUATION_DIR
        / "ablation"
        / graph
        / dataset
        / RUN_SUFFIX
        / "evaluation_results.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing ablation result file: {path}")

    with path.open("r", encoding="utf-8") as stream:
        entries = json.load(stream)

    expected_models = set(get_ablation_model_names(RUN_SUFFIX))
    rows: dict[str, dict[str, float]] = {}
    for entry in entries:
        if entry.get("model") not in expected_models:
            continue
        if int(entry.get("dimension", -1)) != DIMENSION:
            continue

        budget = entry.get("used_config", {}).get("astar_max_visits")
        if budget != EXPECTED_BUDGET:
            raise ValueError(
                f"Unexpected A* budget in {path}: expected {EXPECTED_BUDGET}, "
                f"found {budget} for {entry.get('model')}"
            )

        key = variant_key(entry["model"])
        if key in rows:
            raise ValueError(f"Duplicate {key} entry in {path}")

        metrics = entry.get("evaluation", {}).get("A*", {}).get("metrics")
        if not metrics:
            raise ValueError(f"Missing A* metrics for {entry['model']} in {path}")
        rows[key] = {
            "f1_percent": 100.0 * float(metrics["f1_score"]),
            "avg_nodes_visited": float(metrics["avg_nodes_visited"]),
        }

    missing = set(VARIANT_STYLES) - set(rows)
    extra = set(rows) - set(VARIANT_STYLES)
    if missing or extra:
        raise ValueError(
            f"Expected exactly four Granite-64 ablation rows in {path}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return rows


def load_all_results() -> list[tuple[str, dict[str, dict[str, float]]]]:
    return [
        (panel_title, load_result_set(graph, dataset))
        for graph, dataset, panel_title in RESULT_SETS
    ]


def print_extracted_values(
    result_sets: list[tuple[str, dict[str, dict[str, float]]]],
) -> None:
    print("Values extracted directly from the latest v4 evaluation JSON files:")
    print(f"{'Setting':<32} {'Configuration':<18} {'F1 [%]':>8} {'N':>8}")
    for setting, rows in result_sets:
        for key, style in VARIANT_STYLES.items():
            row = rows[key]
            print(
                f"{setting:<32} {style['label']:<18} "
                f"{row['f1_percent']:>8.1f} {row['avg_nodes_visited']:>8.1f}"
            )


def create_plot(
    result_sets: list[tuple[str, dict[str, dict[str, float]]]],
    output_path: Path,
) -> None:
    apply_thesis_plot_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.25, 5.05),
    )

    panel_labels = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for ax, (setting, rows), panel_label in zip(
        axes.flat, result_sets, panel_labels, strict=True
    ):
        x_values = []
        y_values = []
        for key, style in VARIANT_STYLES.items():
            row = rows[key]
            x_values.append(row["avg_nodes_visited"])
            y_values.append(row["f1_percent"])
            ax.scatter(
                row["avg_nodes_visited"],
                row["f1_percent"],
                s=48,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="#202020",
                linewidth=0.55,
                zorder=3,
            )

        ax.set_title(f"{panel_label} {setting}", loc="left", fontsize=8.4, pad=4)
        ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)

        # Each setting occupies a different numerical range. Tight panel-local
        # limits make the four-way trade-off legible without changing values.
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)
        x_padding = max(0.12, 0.18 * x_span)
        y_padding = max(0.35, 0.12 * y_span)
        ax.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
        ax.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Average visited nodes $N$", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"F$_1$ score (\%)", fontsize=8.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor="#202020",
            markeredgewidth=0.55,
            markersize=6.0,
            label=style["label"],
        )
        for style in VARIANT_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=4,
        fontsize=7.4,
        columnspacing=1.2,
        handletextpad=0.45,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.98, bottom=0.14, wspace=0.14, hspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Title": "Granite-64 activation and distance ablation trade-off",
            "Subject": "F1 score versus average visited nodes across six test settings",
        },
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_sets = load_all_results()
    print_extracted_values(result_sets)
    create_plot(result_sets, args.output.resolve())
    print(f"Saved vector figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
