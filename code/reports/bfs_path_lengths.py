"""Generate BFS path-length statistics in JSON, CSV, and LaTeX formats."""

import argparse
import csv
import json
import time
from pathlib import Path

import traverse_strategies as ts
from core.constants import FILTERED_DATASETS_DIR, REPORTS_DIR
from core.graph_config import graph_arg, get_graph_label, get_graph_path, graph_choices
from core.utils import load_causal_graph
from reports.common import (
    latex_escape,
    latex_number,
    report_paths,
    resolve_repo_path,
    write_json,
    write_latex,
)

DEFAULT_DATASET_DIR = FILTERED_DATASETS_DIR
DEFAULT_OUTPUT_DIR = REPORTS_DIR
DEFAULT_DATASETS = ("msmarco_test", "sem_test")
DEFAULT_GRAPHS = ("causenet", "causenet_full", "ceg")


def load_dataset(dataset_dir, dataset_name, stop_after_pairs=None):
    dataset_path = dataset_dir / f"{dataset_name}_filtered.json"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with open(dataset_path, encoding="utf-8") as file:
        examples = json.load(file)

    if stop_after_pairs is not None:
        examples = examples[:stop_after_pairs]

    return dataset_path, examples


def safe_average(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def run_evaluation_bfs(graph, cause, effect, bfs_max_visits):
    config = {"bfs_max_visits": bfs_max_visits}
    start_time = time.perf_counter()
    path, visited_nodes = ts.bfs_traverse(
        graph,
        cause,
        effect,
        None,
        config=config,
    )
    seconds = time.perf_counter() - start_time

    path_found = bool(path)
    path_nodes = len(path) if path_found else None
    path_hops = path_nodes - 1 if path_found else None
    cutoff = bfs_max_visits != -1 and not path_found and visited_nodes > bfs_max_visits

    return {
        "path_found": path_found,
        "path_hops": path_hops,
        "path_nodes": path_nodes,
        "visited_nodes": visited_nodes,
        "cutoff": cutoff,
        "time_ms": seconds * 1000,
    }


def summarize_lengths(rows):
    evaluated_rows = [row for row in rows if row["evaluated"]]
    reachable_rows = [row for row in evaluated_rows if row["path_found"]]
    unreachable_rows = [row for row in evaluated_rows if not row["path_found"]]
    positive_reachable_rows = [
        row for row in reachable_rows
        if row["answer"] is True
    ]
    negative_reachable_rows = [
        row for row in reachable_rows
        if row["answer"] is False
    ]

    return {
        "reachable_pairs": len(reachable_rows),
        "reachable_positive_pairs": len(positive_reachable_rows),
        "reachable_negative_pairs": len(negative_reachable_rows),
        "avg_bfs_path_hops_reachable": safe_average(
            [row["path_hops"] for row in reachable_rows]
        ),
        "avg_bfs_path_nodes_reachable": safe_average(
            [row["path_nodes"] for row in reachable_rows]
        ),
        "avg_bfs_path_hops_positive_reachable": safe_average(
            [row["path_hops"] for row in positive_reachable_rows]
        ),
        "avg_bfs_path_nodes_positive_reachable": safe_average(
            [row["path_nodes"] for row in positive_reachable_rows]
        ),
        "avg_bfs_path_hops_negative_reachable": safe_average(
            [row["path_hops"] for row in negative_reachable_rows]
        ),
        "avg_bfs_path_nodes_negative_reachable": safe_average(
            [row["path_nodes"] for row in negative_reachable_rows]
        ),
        "avg_nodes_visited_evaluated": safe_average(
            [row["visited_nodes"] for row in evaluated_rows]
        ),
        "avg_nodes_visited_reachable": safe_average(
            [row["visited_nodes"] for row in reachable_rows]
        ),
        "avg_nodes_visited_unreachable": safe_average(
            [row["visited_nodes"] for row in unreachable_rows]
        ),
        "avg_time_ms_evaluated": safe_average(
            [row["time_ms"] for row in evaluated_rows]
        ),
        "avg_time_ms_reachable": safe_average(
            [row["time_ms"] for row in reachable_rows]
        ),
        "avg_time_ms_unreachable": safe_average(
            [row["time_ms"] for row in unreachable_rows]
        ),
        "max_time_ms_evaluated": max(
            [row["time_ms"] for row in evaluated_rows],
            default=None,
        ),
    }


def evaluate_dataset_on_graph(
    dataset_name,
    examples,
    graph_name,
    graph,
    bfs_max_visits=-1,
    progress_every=25,
):
    graph_nodes = set(graph.nodes)
    rows = []
    start_time = time.perf_counter()

    for index, item in enumerate(examples, start=1):
        if progress_every and index % progress_every == 0:
            print(
                f"  {graph_name}/{dataset_name}: processed "
                f"{index:,}/{len(examples):,} pairs..."
            )

        cause = item.get("cause", "")
        effect = item.get("effect", "")
        answer = item.get("answer")
        cause_in_graph = cause in graph_nodes
        effect_in_graph = effect in graph_nodes
        evaluated = cause_in_graph and effect_in_graph

        bfs_result = {
            "path_found": False,
            "path_hops": None,
            "path_nodes": None,
            "visited_nodes": None,
            "cutoff": False,
            "time_ms": None,
        }

        if evaluated:
            bfs_result = run_evaluation_bfs(
                graph=graph,
                cause=cause,
                effect=effect,
                bfs_max_visits=bfs_max_visits,
            )

        rows.append(
            {
                "graph": graph_name,
                "dataset": dataset_name,
                "id": item.get("id", ""),
                "answer": answer,
                "cause": cause,
                "effect": effect,
                "cause_in_graph": cause_in_graph,
                "effect_in_graph": effect_in_graph,
                "evaluated": evaluated,
                **bfs_result,
            }
        )

    elapsed = time.perf_counter() - start_time
    evaluated_rows = [row for row in rows if row["evaluated"]]
    skipped_rows = [row for row in rows if not row["evaluated"]]
    found_rows = [row for row in evaluated_rows if row["path_found"]]
    cutoff_rows = [row for row in evaluated_rows if row["cutoff"]]
    summary = {
        "total_pairs": len(rows),
        "positive_pairs": sum(1 for row in rows if row["answer"] is True),
        "negative_pairs": sum(1 for row in rows if row["answer"] is False),
        "evaluated_pairs_both_nodes_in_graph": len(evaluated_rows),
        "skipped_pairs_missing_graph_node": len(skipped_rows),
        "pairs_missing_cause": sum(1 for row in rows if not row["cause_in_graph"]),
        "pairs_missing_effect": sum(1 for row in rows if not row["effect_in_graph"]),
        "pairs_with_bfs_path": len(found_rows),
        "pairs_without_bfs_path": len(evaluated_rows) - len(found_rows),
        "pairs_cutoff": len(cutoff_rows),
        "bfs_max_visits": bfs_max_visits,
        "seconds": elapsed,
        **summarize_lengths(rows),
    }

    return summary, rows


def print_summary(graph_name, dataset_name, summary):
    print(f"\n{graph_name} / {dataset_name}")
    print("=" * (len(graph_name) + len(dataset_name) + 3))
    print(f"Total pairs:                         {summary['total_pairs']:,}")
    print(
        "Evaluated pairs, both nodes present: "
        f"{summary['evaluated_pairs_both_nodes_in_graph']:,}"
    )
    print(f"Skipped, missing graph node:         {summary['skipped_pairs_missing_graph_node']:,}")
    print(f"Pairs with BFS path:                 {summary['pairs_with_bfs_path']:,}")
    print(f"Pairs without BFS path:              {summary['pairs_without_bfs_path']:,}")
    print(f"Pairs cut off:                       {summary['pairs_cutoff']:,}")
    print(
        "Avg BFS path length, nodes:          "
        f"{summary['avg_bfs_path_nodes_reachable']}"
    )
    print(
        "Avg BFS path length, hops:           "
        f"{summary['avg_bfs_path_hops_reachable']}"
    )
    print(
        "Avg nodes visited, evaluated:        "
        f"{summary['avg_nodes_visited_evaluated']}"
    )
    print(
        "Avg time ms, evaluated:              "
        f"{summary['avg_time_ms_evaluated']}"
    )


def write_summary_csv(path, summaries):
    fieldnames = [
        "graph",
        "dataset",
        "total_pairs",
        "positive_pairs",
        "negative_pairs",
        "evaluated_pairs_both_nodes_in_graph",
        "skipped_pairs_missing_graph_node",
        "pairs_missing_cause",
        "pairs_missing_effect",
        "pairs_with_bfs_path",
        "pairs_without_bfs_path",
        "pairs_cutoff",
        "reachable_pairs",
        "reachable_positive_pairs",
        "reachable_negative_pairs",
        "avg_bfs_path_nodes_reachable",
        "avg_bfs_path_hops_reachable",
        "avg_bfs_path_nodes_positive_reachable",
        "avg_bfs_path_hops_positive_reachable",
        "avg_bfs_path_nodes_negative_reachable",
        "avg_bfs_path_hops_negative_reachable",
        "avg_nodes_visited_evaluated",
        "avg_nodes_visited_reachable",
        "avg_nodes_visited_unreachable",
        "avg_time_ms_evaluated",
        "avg_time_ms_reachable",
        "avg_time_ms_unreachable",
        "max_time_ms_evaluated",
        "bfs_max_visits",
        "seconds",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)


def write_reports(report, summaries, output_dir):
    json_path, csv_path, tex_path = report_paths("bfs_path_lengths", output_dir)
    write_json(json_path, report)
    write_summary_csv(csv_path, summaries)

    table_rows = []
    for row in summaries:
        cells = [
            latex_escape(get_graph_label(row["graph"])),
            latex_escape(row["dataset"]),
            latex_number(row["evaluated_pairs_both_nodes_in_graph"]),
            latex_number(row["pairs_with_bfs_path"]),
            latex_number(row["avg_bfs_path_hops_reachable"], 2),
            latex_number(row["avg_nodes_visited_evaluated"], 1),
            latex_number(row["avg_time_ms_evaluated"], 1),
        ]
        table_rows.append("    " + " & ".join(cells) + " \\\\")

    latex = "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \resizebox{\textwidth}{!}{%",
            r"  \begin{tabular}{@{}llrrrrr@{}}",
            r"    \toprule",
            (
                r"    \textbf{Graph} & \textbf{Dataset} & \textbf{Evaluated} & "
                r"\textbf{Reachable} & \textbf{Avg. hops} & "
                r"\textbf{Avg. visited} & \textbf{Avg. time [ms]} \\"
            ),
            r"    \midrule",
            *table_rows,
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            (
                r"  \caption{Breadth-first-search path lengths and search effort "
                r"for evaluation question pairs.}"
            ),
            r"  \label{tab:bfs-path-lengths}",
            r"\end{table}",
        ]
    )
    write_latex(tex_path, latex)
    return json_path, csv_path, tex_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the same BFS baseline used by evaluation on test cause/effect "
            "pairs and report path lengths, visited nodes, and runtime."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names without _filtered.json. Defaults to msmarco_test sem_test.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        type=graph_arg,
        choices=graph_choices(),
        default=list(DEFAULT_GRAPHS),
        help="Graphs to check.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing normalized *_filtered.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Report root; files are written below bfs_path_lengths/.",
    )
    parser.add_argument(
        "--bfs-max-visits",
        "--max-visited",
        dest="bfs_max_visits",
        type=int,
        default=-1,
        help="Same cap passed to bfs_traverse as bfs_max_visits. -1 means no cap.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print pair progress every N dataset rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--graph-progress-every",
        type=int,
        default=1_000_000,
        help="Print graph-load progress every N loaded edges. Use 0 to disable.",
    )
    parser.add_argument(
        "--stop-after-pairs",
        type=int,
        default=None,
        help="Debug option: only evaluate the first N rows from each dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset_dir = resolve_repo_path(args.dataset_dir)
    args.output_dir = resolve_repo_path(args.output_dir)

    datasets = {}
    dataset_paths = {}

    for dataset_name in args.datasets:
        dataset_path, examples = load_dataset(
            args.dataset_dir,
            dataset_name,
            stop_after_pairs=args.stop_after_pairs,
        )
        datasets[dataset_name] = examples
        dataset_paths[dataset_name] = str(dataset_path)
        print(f"Loaded {len(examples):,} pairs from {dataset_path}")

    report = {
        "datasets": dataset_paths,
        "bfs_implementation": "traverse_strategies.bfs.bfs_traverse",
        "bfs_max_visits": args.bfs_max_visits,
        "stop_after_pairs": args.stop_after_pairs,
        "graphs": {},
    }
    summary_rows = []

    for graph_name in args.graphs:
        graph_path = resolve_repo_path(get_graph_path(graph_name))

        if not graph_path.exists():
            raise FileNotFoundError(graph_path)

        print(f"\nLoading {get_graph_label(graph_name)} from {graph_path}")
        graph_load_start = time.perf_counter()
        graph = load_causal_graph(
            graph_path,
            use_inverse=False,
            progress_every=args.graph_progress_every,
            progress_label=f"{graph_name} NetworkX graph",
        )
        graph_load_seconds = time.perf_counter() - graph_load_start
        print(
            f"Loaded {get_graph_label(graph_name)} NetworkX graph: "
            f"{graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges in {graph_load_seconds:.1f}s"
        )

        report["graphs"][graph_name] = {
            "label": get_graph_label(graph_name),
            "path": str(graph_path),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "load_seconds": graph_load_seconds,
            "datasets": {},
        }

        for dataset_name, examples in datasets.items():
            summary, rows = evaluate_dataset_on_graph(
                dataset_name=dataset_name,
                examples=examples,
                graph_name=graph_name,
                graph=graph,
                bfs_max_visits=args.bfs_max_visits,
                progress_every=args.progress_every,
            )
            report["graphs"][graph_name]["datasets"][dataset_name] = {
                "summary": summary,
                "examples": rows,
            }
            summary_rows.append(
                {
                    "graph": graph_name,
                    "dataset": dataset_name,
                    **summary,
                }
            )
            print_summary(graph_name, dataset_name, summary)

    json_path, csv_path, tex_path = write_reports(
        report=report,
        summaries=summary_rows,
        output_dir=args.output_dir,
    )
    print(f"\nWrote JSON report: {json_path}")
    print(f"Wrote CSV report:  {csv_path}")
    print(f"Wrote LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
