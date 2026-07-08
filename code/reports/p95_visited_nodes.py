"""Generate a thesis table for p95 visited-node traversal caps.

The source artifact is produced by ``evaluation.visited_nodes_analysis``.  The
evaluation code uses ``ceil(p95_visited_successful_only)`` as the A* max-visit
cap, so this report stores both the raw percentile and the integer cap while
rendering the cap in CSV/LaTeX.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from core.constants import EVALUATION_DIR, REPORTS_DIR
from core.utils import get_model_base_name, is_finetuned_model_name
from reports.common import (
    display_path,
    latex_escape,
    latex_number,
    report_paths,
    resolve_repo_path,
    write_json,
    write_latex,
)

REPORT_NAME = "p95_visited_nodes"
DEFAULT_GRAPH = "causenet"
DEFAULT_DATASET = "msmarco_train"
DEFAULT_RUN_SUFFIX = "v3"
DEFAULT_STRATEGY = "A*"
DEFAULT_METRIC = "p95_visited_successful_only"
DEFAULT_MODEL_FAMILY = "finetuned"
DEFAULT_DIMENSIONS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024)

MODEL_ORDER = {
    "Qwen3-Embedding-0.6B": 0,
    "all-mpnet-base-v2": 1,
    "bge-large-en-v1.5": 2,
    "granite-embedding-english-r2": 3,
    "mxbai-embed-large-v1": 4,
}

COMPACT_MODEL_LABELS = {
    "Qwen3-Embedding-0.6B": "Qwen 0.6B",
    "all-mpnet-base-v2": "MPNet",
    "bge-large-en-v1.5": "BGE",
    "granite-embedding-english-r2": "Granite Emb.",
    "mxbai-embed-large-v1": "MXBAI Emb.",
}

METRIC_LABELS = {
    "p95_visited_successful_only": "p95 visited nodes, successful paths only",
    "p95_visited_all": "p95 visited nodes, all covered examples",
    "max_visited_successful_only": "maximum visited nodes, successful paths only",
    "max_visited_all": "maximum visited nodes, all covered examples",
}


def default_input_path(graph, dataset, run_suffix, ablation=False):
    output_root = EVALUATION_DIR / "ablation" if ablation else EVALUATION_DIR
    return output_root / graph / dataset / run_suffix / "visited_nodes_analysis.json"


def load_analysis(path):
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of analysis entries in {path}")

    return payload


def matches_model_family(model_name, model_family):
    is_finetuned = is_finetuned_model_name(model_name)

    if model_family == "finetuned":
        return is_finetuned
    if model_family == "base":
        return not is_finetuned
    if model_family == "all":
        return True

    raise ValueError(f"Unsupported model family: {model_family}")


def compact_model_label(model_name):
    base_name = get_model_base_name(model_name)
    return COMPACT_MODEL_LABELS.get(base_name, base_name)


def metric_cap(value):
    if value is None:
        return None
    return int(math.ceil(float(value)))


def extract_p95_rows(
    entries,
    *,
    strategy=DEFAULT_STRATEGY,
    metric=DEFAULT_METRIC,
    dimensions=DEFAULT_DIMENSIONS,
    model_family=DEFAULT_MODEL_FAMILY,
):
    """Return one compact row per model with raw values and integer caps."""
    dimensions = tuple(dimensions)
    rows_by_model = {}

    for entry in entries:
        model_name = entry.get("model")
        dimension = entry.get("dimension")
        analysis = entry.get("analysis", {})

        if model_name is None or dimension not in dimensions:
            continue
        if analysis.get("strategy") != strategy:
            continue
        if not matches_model_family(model_name, model_family):
            continue
        if metric not in analysis:
            continue

        base_name = get_model_base_name(model_name)
        row = rows_by_model.setdefault(
            model_name,
            {
                "model": compact_model_label(model_name),
                "source_model": model_name,
                "base_model": base_name,
                "values": {},
            },
        )

        raw_value = analysis.get(metric)
        row["values"][str(dimension)] = {
            "raw": None if raw_value is None else float(raw_value),
            "cap": metric_cap(raw_value),
            "num_examples": analysis.get("num_examples"),
            "num_successful_paths": analysis.get("num_successful_paths"),
        }

    return sorted(
        rows_by_model.values(),
        key=lambda row: (
            MODEL_ORDER.get(row["base_model"], 999),
            row["model"],
            row["source_model"],
        ),
    )


def write_csv(path, rows, dimensions):
    fieldnames = ["model", *[str(dimension) for dimension in dimensions]]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            flat_row = {"model": row["model"]}

            for dimension in dimensions:
                value = row["values"].get(str(dimension), {}).get("cap")
                flat_row[str(dimension)] = "" if value is None else value

            writer.writerow(flat_row)


def render_latex(rows, dimensions, *, caption, label):
    column_spec = "@{}l" + "r" * len(dimensions) + "@{}"
    dimension_count = len(dimensions)
    header_dimensions = " & ".join(
        rf"\textbf{{{dimension}}}" for dimension in dimensions
    )
    body_lines = []

    for row in rows:
        values = []
        for dimension in dimensions:
            value = row["values"].get(str(dimension), {}).get("cap")
            values.append(latex_number(value))

        body_lines.append(
            f"{latex_escape(row['model'])} & " + " & ".join(values) + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            rf"\textbf{{Model}} & \multicolumn{{{dimension_count}}}{{c}}{{\textbf{{Dimension}}}} \\",
            rf"\cmidrule(l){{2-{dimension_count + 1}}}",
            f" & {header_dimensions} " + r"\\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def write_reports(
    output_dir,
    *,
    source_path,
    rows,
    dimensions,
    graph,
    dataset,
    run_suffix,
    strategy,
    metric,
    model_family,
):
    json_path, csv_path, tex_path = report_paths(REPORT_NAME, output_dir)
    report = {
        "source": {
            "path": display_path(source_path),
            "graph": graph,
            "dataset": dataset,
            "run_suffix": run_suffix,
            "strategy": strategy,
            "metric": metric,
            "metric_description": METRIC_LABELS.get(metric, metric),
            "model_family": model_family,
            "rendered_value": "ceil(metric)",
        },
        "dimensions": list(dimensions),
        "rows": rows,
    }

    write_json(json_path, report)
    write_csv(csv_path, rows, dimensions)

    caption = (
        "A* expansion thresholds derived from the p95 visited-node counts on "
        "MS MARCO train with CauseNet."
    )
    label = "tab:p95-visited-nodes-msmarco-train-causenet"
    write_latex(
        tex_path,
        render_latex(rows, dimensions, caption=caption, label=label),
    )

    return json_path, csv_path, tex_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate JSON/CSV/LaTeX reports for p95 visited-node caps from "
            "visited_nodes_analysis.json."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Optional path to visited_nodes_analysis.json. Defaults to "
            "data/evaluation/<graph>/<dataset>/<run_suffix>/visited_nodes_analysis.json."
        ),
    )
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--run-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Read from data/evaluation/ablation instead of data/evaluation.",
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=tuple(METRIC_LABELS),
        help="Analysis metric to render. Defaults to the cap used by evaluation.",
    )
    parser.add_argument(
        "--model-family",
        choices=("finetuned", "base", "all"),
        default=DEFAULT_MODEL_FAMILY,
        help="Which model entries to include. Defaults to fine-tuned models.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMENSIONS),
        help="Dimension columns to include, in display order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help=(
            "Report root. The files are written below the p95_visited_nodes/ "
            "subdirectory."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = (
        resolve_repo_path(args.input_json)
        if args.input_json is not None
        else default_input_path(
            args.graph,
            args.dataset,
            args.run_suffix,
            ablation=args.ablation,
        )
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing visited-node analysis file: {source_path}. "
            "Generate it first with evaluation.visited_nodes_analysis."
        )

    entries = load_analysis(source_path)
    rows = extract_p95_rows(
        entries,
        strategy=args.strategy,
        metric=args.metric,
        dimensions=args.dimensions,
        model_family=args.model_family,
    )

    if not rows:
        raise ValueError(
            "No matching rows found. Check --strategy, --metric, "
            "--model-family, and --dimensions."
        )

    paths = write_reports(
        args.output_dir,
        source_path=source_path,
        rows=rows,
        dimensions=tuple(args.dimensions),
        graph=args.graph,
        dataset=args.dataset,
        run_suffix=args.run_suffix,
        strategy=args.strategy,
        metric=args.metric,
        model_family=args.model_family,
    )

    print("Wrote p95 visited-node report:")
    for path in paths:
        print(f"  {display_path(path)}")


if __name__ == "__main__":
    main()
