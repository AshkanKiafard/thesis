"""Generate causal-graph size statistics for the thesis."""

import argparse
import csv
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from core.constants import REPORTS_DIR
from core.graph_config import graph_arg, get_graph_label, get_graph_path, graph_choices
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

DEFAULT_GRAPHS = ("causenet", "causenet_full", "ceg", "ceg_full")
DEFAULT_OUTPUT_DIR = REPORTS_DIR
DEFAULT_BATCH_SIZE = 100_000
COUNT_MODE = "unique_directed_edges"


@dataclass
class CountResult:
    graph: str
    label: str
    path: str
    nodes: int
    edges: int
    parsed_relation_records: int
    duplicate_relation_records: int
    count_mode: str
    count_seconds: float
    temp_database_peak_bytes: int

    def as_dict(self):
        return {
            "graph": self.graph,
            "label": self.label,
            "path": self.path,
            "nodes": self.nodes,
            "edges": self.edges,
            "parsed_relation_records": self.parsed_relation_records,
            "duplicate_relation_records": self.duplicate_relation_records,
            "count_mode": self.count_mode,
            "count_seconds": self.count_seconds,
            "temp_database_peak_bytes": self.temp_database_peak_bytes,
        }


def create_count_database(db_path: Path):
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute(
        """
        CREATE TABLE edges (
            cause TEXT NOT NULL,
            effect TEXT NOT NULL,
            PRIMARY KEY (cause, effect)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE nodes (
            node TEXT NOT NULL PRIMARY KEY
        ) WITHOUT ROWID
        """
    )
    return connection


def flush_batch(connection, edge_batch, node_batch):
    if not edge_batch:
        return

    connection.executemany(
        "INSERT OR IGNORE INTO edges (cause, effect) VALUES (?, ?)",
        edge_batch,
    )
    connection.executemany(
        "INSERT OR IGNORE INTO nodes (node) VALUES (?)",
        ((node,) for node in node_batch),
    )
    connection.commit()
    edge_batch.clear()
    node_batch.clear()


def count_graph(
    graph_name,
    progress_every=None,
    batch_size=DEFAULT_BATCH_SIZE,
    temporary_directory=None,
):
    label = get_graph_label(graph_name)
    graph_path = resolve_repo_path(get_graph_path(graph_name))

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    print(f"Counting {label} from {display_path(graph_path)}...", flush=True)
    start_time = time.perf_counter()
    temporary_directory = (
        resolve_repo_path(temporary_directory)
        if temporary_directory is not None
        else None
    )
    if temporary_directory is not None:
        temporary_directory.mkdir(parents=True, exist_ok=True)

    fd, db_name = tempfile.mkstemp(
        prefix=f"graph_statistics_{graph_name}_",
        suffix=".sqlite",
        dir=temporary_directory,
    )
    os.close(fd)
    db_path = Path(db_name)
    print(f"Temporary deduplication database: {display_path(db_path)}", flush=True)

    connection = None
    parsed_relation_records = 0
    temp_database_peak_bytes = 0

    try:
        connection = create_count_database(db_path)
        edge_batch = []
        node_batch = []

        for parsed_relation_records, (cause, effect, _edge_attrs) in enumerate(
            _iter_graph_edges(graph_path),
            start=1,
        ):
            edge_batch.append((cause, effect))
            node_batch.extend((cause, effect))

            if len(edge_batch) >= batch_size:
                flush_batch(connection, edge_batch, node_batch)
                temp_database_peak_bytes = max(
                    temp_database_peak_bytes,
                    db_path.stat().st_size,
                )

            if progress_every and parsed_relation_records % progress_every == 0:
                print(
                    "Parsed "
                    f"{parsed_relation_records:,} relation records from {label}...",
                    flush=True,
                )

        flush_batch(connection, edge_batch, node_batch)
        temp_database_peak_bytes = max(
            temp_database_peak_bytes,
            db_path.stat().st_size,
        )

        nodes = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        duplicate_relation_records = parsed_relation_records - edge_count
        count_seconds = time.perf_counter() - start_time

        return CountResult(
            graph=graph_name,
            label=label,
            path=display_path(graph_path),
            nodes=nodes,
            edges=edge_count,
            parsed_relation_records=parsed_relation_records,
            duplicate_relation_records=duplicate_relation_records,
            count_mode=COUNT_MODE,
            count_seconds=count_seconds,
            temp_database_peak_bytes=temp_database_peak_bytes,
        ).as_dict()
    finally:
        if connection is not None:
            connection.close()

        try:
            db_path.unlink()
            print(
                "Removed temporary deduplication database: "
                f"{display_path(db_path)}",
                flush=True,
            )
        except FileNotFoundError:
            pass


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
                "parsed_relation_records",
                "duplicate_relation_records",
                "count_mode",
                "count_seconds",
                "temp_database_peak_bytes",
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
            (
                r"  \caption{Number of nodes and unique directed edges "
                r"in each causal graph.}"
            ),
            r"  \label{tab:graph-statistics}",
            r"\end{table}",
        ]
    )
    write_latex(tex_path, latex)

    return json_path, csv_path, tex_path


def print_table(rows):
    print()
    print(
        f"{'Graph':<18} {'Nodes':>14} {'Unique edges':>14} "
        f"{'Parsed records':>16} {'Duplicates':>14} {'Count seconds':>14}"
    )
    print("-" * 100)

    for row in rows:
        print(
            f"{row['label']:<18} "
            f"{row['nodes']:>14,} "
            f"{row['edges']:>14,} "
            f"{row['parsed_relation_records']:>16,} "
            f"{row['duplicate_relation_records']:>14,} "
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
        type=graph_arg,
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of parsed relation records to insert per SQLite batch.",
    )
    parser.add_argument(
        "--temporary-directory",
        type=Path,
        default=None,
        help=(
            "Directory for the temporary SQLite deduplication database. "
            "Use a drive with enough free disk space for full graphs."
        ),
    )
    args = parser.parse_args()

    if args.progress_every < 0:
        parser.error("--progress-every must be greater than or equal to 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    return args


def main():
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    progress_every = args.progress_every if args.progress_every > 0 else None

    rows = [
        count_graph(
            graph_name,
            progress_every=progress_every,
            batch_size=args.batch_size,
            temporary_directory=args.temporary_directory,
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
