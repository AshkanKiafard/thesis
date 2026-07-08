"""Generate dataset and graph-coverage statistics for the thesis."""

import argparse
import csv
import json
from pathlib import Path

from core.constants import FILTERED_DATASETS_DIR, REPORTS_DIR
from core.graph_config import get_graph_config, graph_choices
from core.utils import load_graph_nodes as load_causal_graph_nodes
from reports.common import (
    latex_escape,
    latex_number,
    report_paths,
    resolve_repo_path,
    write_json,
    write_latex,
)

DATASET_DIR = FILTERED_DATASETS_DIR
DEFAULT_OUTPUT_DIR = REPORTS_DIR
SELECTED_GRAPHS = ("causenet", "causenet_full", "causalbank")


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_split_name(file_path):
    name = file_path.stem.lower()

    if "train_valid" in name:
        return "Train+Valid"
    if "validation" in name or "valid" in name:
        return "Validation"
    if "train" in name:
        return "Train"
    if "test" in name:
        return "Test"

    return file_path.stem


def pretty_dataset_name(file_path):
    name = file_path.stem.lower().replace("_filtered", "")

    for suffix in ("_train_valid", "_valid", "_validation", "_train", "_test"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break

    return name


def count_split(data, graph_nodes_by_name):
    total_pos = 0
    total_neg = 0
    graph_counts = {
        graph_name: {"pos": 0, "neg": 0}
        for graph_name in graph_nodes_by_name
    }

    for item in data:
        label = item.get("answer")

        if label is True:
            total_pos += 1
        elif label is False:
            total_neg += 1
        else:
            print(f"Warning: invalid label in {item.get('id', 'unknown')}")
            continue

        cause = item["cause"]
        effect = item["effect"]

        # Same logic as evaluation: only count examples where both nodes exist.
        for graph_name, graph_nodes in graph_nodes_by_name.items():
            if cause in graph_nodes and effect in graph_nodes:
                if label is True:
                    graph_counts[graph_name]["pos"] += 1
                else:
                    graph_counts[graph_name]["neg"] += 1

    return total_pos, total_neg, graph_counts


def write_reports(output_dir, rows, selected_graphs, graph_metadata):
    json_path, csv_path, tex_path = report_paths("dataset_statistics", output_dir)

    report = {
        "graphs": graph_metadata,
        "datasets": rows,
    }

    write_json(json_path, report)

    fieldnames = ["dataset", "split", "total_pos", "total_neg"]

    for graph_name in selected_graphs:
        fieldnames.extend([f"{graph_name}_pos", f"{graph_name}_neg"])

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            flat_row = {
                "dataset": row["dataset"],
                "split": row["split"],
                "total_pos": row["total_pos"],
                "total_neg": row["total_neg"],
            }

            for graph_name in selected_graphs:
                counts = row["graphs"][graph_name]
                flat_row[f"{graph_name}_pos"] = counts["pos"]
                flat_row[f"{graph_name}_neg"] = counts["neg"]

            writer.writerow(flat_row)

    graph_headers = [
        latex_escape(graph_metadata[graph_name]["label"])
        for graph_name in selected_graphs
    ]
    first_header = [r"    \textbf{Split}", r"\multicolumn{2}{c}{\textbf{Total}}"]
    first_header.extend(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{label}}}}}"
        for label in graph_headers
    )
    second_header = [""] + [r"\textbf{Pos.}", r"\textbf{Neg.}"] * (
        len(selected_graphs) + 1
    )
    split_order = {"Train": 0, "Train+Valid": 1, "Validation": 2, "Test": 3}
    ordered_rows = sorted(
        rows,
        key=lambda row: (row["dataset"], split_order.get(row["split"], 99)),
    )
    table_rows = []
    current_dataset = None
    dataset_labels = {"msmarco": "MS MARCO", "sem": "SemEval"}

    for row in ordered_rows:
        if row["dataset"] != current_dataset:
            if current_dataset is not None:
                table_rows.append(r"    \addlinespace[2pt]")
            current_dataset = row["dataset"]
            dataset_label = dataset_labels.get(current_dataset, current_dataset)
            table_rows.append(
                rf"    \multicolumn{{{2 * len(selected_graphs) + 3}}}"
                rf"{{@{{}}l}}{{\emph{{{latex_escape(dataset_label)}}}}} \\"
            )

        cells = [
            latex_escape(row["split"]),
            latex_number(row["total_pos"]),
            latex_number(row["total_neg"]),
        ]
        for graph_name in selected_graphs:
            cells.extend(
                (
                    latex_number(row["graphs"][graph_name]["pos"]),
                    latex_number(row["graphs"][graph_name]["neg"]),
                )
            )
        table_rows.append("    " + " & ".join(cells) + " \\\\")

    column_spec = "@{}l" + "rr" * (len(selected_graphs) + 1) + "@{}"
    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \resizebox{\textwidth}{!}{%",
            rf"  \begin{{tabular}}{{{column_spec}}}",
            r"    \toprule",
            " & ".join(first_header) + " \\\\",
            " & ".join(second_header) + " \\\\",
            r"    \midrule",
            *table_rows,
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            (
                r"  \caption{Number of positive and negative causal questions "
                r"before and after graph-node coverage filtering.}"
            ),
            r"  \label{tab:dataset-statistics}",
            r"\end{table}",
        ]
    )
    write_latex(tex_path, latex)

    return json_path, csv_path, tex_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check positive/negative dataset balance before and after applying "
            "evaluation-style graph-node coverage filtering."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing normalized *_filtered.json files.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=graph_choices(),
        default=list(SELECTED_GRAPHS),
        help="Graphs to check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Report root; files are written below dataset_statistics/.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print the table; do not write report files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = resolve_repo_path(args.dataset_dir)
    output_dir = resolve_repo_path(args.output_dir)
    selected_graphs = tuple(args.graphs)
    dataset_files = sorted(dataset_dir.glob("*.json"))

    graph_nodes_by_name = {}
    graph_metadata = {}

    for graph_name in selected_graphs:
        graph_config = get_graph_config(graph_name)
        graph_path = resolve_repo_path(graph_config["path"])

        if not graph_path.exists():
            message = f"Missing {graph_config['label']} graph: {graph_path}"
            if graph_name == "causalbank":
                message += (
                    "\nRun: python -m preprocessing.filter_causalbank_graph"
                )
            raise FileNotFoundError(message)

        print(f"Loading {graph_config['label']} nodes from {graph_path}...")
        graph_nodes = load_causal_graph_nodes(graph_path)
        graph_nodes_by_name[graph_name] = graph_nodes
        graph_metadata[graph_name] = {
            "label": graph_config["label"],
            "path": str(graph_path),
            "nodes": len(graph_nodes),
        }
        print(f"Loaded {len(graph_nodes):,} {graph_config['label']} nodes.")

    rows = []

    for dataset_file in dataset_files:
        data = load_dataset(dataset_file)
        total_pos, total_neg, graph_counts = count_split(data, graph_nodes_by_name)

        rows.append(
            {
                "dataset": pretty_dataset_name(dataset_file),
                "split": pretty_split_name(dataset_file),
                "total_pos": total_pos,
                "total_neg": total_neg,
                "graphs": graph_counts,
            }
        )

    print("\nDataset Statistics")
    header = (
        f"{'Dataset':<15}"
        f"{'Split':<15}"
        f"{'Total Pos':>12}"
        f"{'Total Neg':>12}"
    )

    for graph_name in selected_graphs:
        graph_label = get_graph_config(graph_name)["label"]
        header += f"{graph_label + ' Pos':>16}{graph_label + ' Neg':>16}"

    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for row_data in rows:
        row = (
            f"{row_data['dataset']:<15}"
            f"{row_data['split']:<15}"
            f"{row_data['total_pos']:>12}"
            f"{row_data['total_neg']:>12}"
        )

        for graph_name in selected_graphs:
            counts = row_data["graphs"][graph_name]
            row += f"{counts['pos']:>16}{counts['neg']:>16}"

        print(row)

    print("=" * len(header))

    if not args.no_write:
        json_path, csv_path, tex_path = write_reports(
            output_dir=output_dir,
            rows=rows,
            selected_graphs=selected_graphs,
            graph_metadata=graph_metadata,
        )
        print(f"\nWrote JSON report: {json_path}")
        print(f"Wrote CSV report:  {csv_path}")
        print(f"Wrote LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
