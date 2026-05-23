import argparse
import csv
import json
import time
from pathlib import Path

from core.graph_config import get_graph_label, get_graph_path, graph_choices
from core.utils import load_causal_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHS = ("causenet", "causenet_full", "causalbank")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reports" / "graph_sizes"


def resolve_repo_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def count_graph(graph_name, progress_every=None):
    label = get_graph_label(graph_name)
    graph_path = resolve_repo_path(get_graph_path(graph_name))

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    print(f"Loading {label} from {display_path(graph_path)}...", flush=True)
    start_time = time.perf_counter()
    graph = load_causal_graph(
        graph_path,
        progress_every=progress_every,
        progress_label=label,
    )
    load_seconds = time.perf_counter() - start_time

    return {
        "graph": graph_name,
        "label": label,
        "path": display_path(graph_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "load_seconds": load_seconds,
    }


def write_reports(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "graph_sizes.json"
    csv_path = output_dir / "graph_sizes.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "graph",
                "label",
                "path",
                "nodes",
                "edges",
                "load_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def print_table(rows):
    print()
    print(f"{'Graph':<18} {'Nodes':>14} {'Edges':>14} {'Load seconds':>14}")
    print("-" * 64)

    for row in rows:
        print(
            f"{row['label']:<18} "
            f"{row['nodes']:>14,} "
            f"{row['edges']:>14,} "
            f"{row['load_seconds']:>14.1f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count nodes and directed edges for the configured causal graphs. "
            "Counts are based on core.utils.load_causal_graph, matching the "
            "NetworkX graph used by BFS, A*, and Dijkstra."
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
        help="Directory for graph_sizes.json and graph_sizes.csv.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print graph loading progress every N input edges. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    progress_every = args.progress_every if args.progress_every > 0 else None

    rows = [
        count_graph(graph_name, progress_every=progress_every)
        for graph_name in args.graphs
    ]

    print_table(rows)
    json_path, csv_path = write_reports(rows, output_dir)

    print()
    print(f"Wrote JSON report: {display_path(json_path)}")
    print(f"Wrote CSV report:  {display_path(csv_path)}")


if __name__ == "__main__":
    main()
