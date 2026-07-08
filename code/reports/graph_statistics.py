"""Generate causal-graph size statistics for the thesis."""

import argparse
import csv
import time
from pathlib import Path

from core.constants import REPORTS_DIR
from core.graph_config import get_graph_label, get_graph_path, graph_choices
from core.utils import _iter_graph_edges
from reports.common import (
    display_path,
    latex_escape,
    latex_number,
    report_paths,
    resolve_repo_path,
    write_json,
    write_latex,
)

DEFAULT_GRAPHS = ("causenet", "causenet_full", "causalbank", "causalbank_full")
DEFAULT_OUTPUT_DIR = REPORTS_DIR


def count_graph(graph_name, progress_every=None, deduplicate_edges=False):
    label = get_graph_label(graph_name)
    graph_path = resolve_repo_path(get_graph_path(graph_name))

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    print(f"Counting {label} from {display_path(graph_path)}...", flush=True)
    start_time = time.perf_counter()
    nodes = set()
    edge_count = 0
    seen_edges = set() if deduplicate_edges else None

    for input_edge_count, (cause, effect, _edge_attrs) in enumerate(
        _iter_graph_edges(graph_path),
        start=1,
    ):
        nodes.add(cause)
        nodes.add(effect)

        if seen_edges is None:
            edge_count += 1
        else:
            edge = (cause, effect)
            if edge not in seen_edges:
                seen_edges.add(edge)
                edge_count += 1

        if progress_every and input_edge_count % progress_every == 0:
            print(
                f"Counted {input_edge_count:,} input edges from {label}...",
                flush=True,
            )

    count_seconds = time.perf_counter() - start_time

    return {
        "graph": graph_name,
        "label": label,
        "path": display_path(graph_path),
        "nodes": len(nodes),
        "edges": edge_count,
        "count_mode": (
            "streamed_unique_edges"
            if deduplicate_edges
            else "streamed_input_edges"
        ),
        "count_seconds": count_seconds,
    }


def write_reports(rows, output_dir):
    json_path, csv_path, tex_path = report_paths("graph_statistics", output_dir)
    write_json(json_path, {"graphs": rows})

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "graph",
                "label",
                "path",
                "nodes",
                "edges",
                "count_mode",
                "count_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    table_rows = [
        " & ".join(
            (
                latex_escape(row["label"]),
                latex_number(row["nodes"]),
                latex_number(row["edges"]),
            )
        )
        + r" \\"
        for row in rows
    ]
    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \begin{tabular}{@{}lrr@{}}",
            r"    \toprule",
            r"    \textbf{Graph} & \textbf{Nodes} & \textbf{Edges} \\",
            r"    \midrule",
            *(f"    {row}" for row in table_rows),
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Number of nodes and directed edges in each causal graph.}",
            r"  \label{tab:graph-statistics}",
            r"\end{table}",
        ]
    )
    write_latex(tex_path, latex)

    return json_path, csv_path, tex_path


def print_table(rows):
    print()
    print(
        f"{'Graph':<18} {'Nodes':>14} {'Edges':>14} "
        f"{'Mode':<22} {'Count seconds':>14}"
    )
    print("-" * 90)

    for row in rows:
        print(
            f"{row['label']:<18} "
            f"{row['nodes']:>14,} "
            f"{row['edges']:>14,} "
            f"{row['count_mode']:<22} "
            f"{row['count_seconds']:>14.1f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count nodes and directed edges for the configured causal graphs. "
            "By default this streams through the graph file and does not build "
            "the NetworkX graph, so it can handle full graph files."
        )
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=graph_choices(),
        default=DEFAULT_GRAPHS,
        help="Graphs to count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Report root; files are written below graph_statistics/.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print graph counting progress every N parsed input edges. Use 0 to disable.",
    )
    parser.add_argument(
        "--deduplicate-edges",
        action="store_true",
        help=(
            "Count unique (cause, effect) edges like NetworkX DiGraph. This can "
            "still use a lot of memory on full graphs, so the default only "
            "counts streamed valid input edges."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    progress_every = args.progress_every if args.progress_every > 0 else None

    rows = [
        count_graph(
            graph_name,
            progress_every=progress_every,
            deduplicate_edges=args.deduplicate_edges,
        )
        for graph_name in args.graphs
    ]

    print_table(rows)
    json_path, csv_path, tex_path = write_reports(rows, output_dir)

    print()
    print(f"Wrote JSON report: {display_path(json_path)}")
    print(f"Wrote CSV report:  {display_path(csv_path)}")
    print(f"Wrote LaTeX table: {display_path(tex_path)}")


if __name__ == "__main__":
    main()
