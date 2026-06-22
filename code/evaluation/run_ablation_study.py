import argparse
import subprocess
import sys
from pathlib import Path

from core.graph_config import graph_choices
from core.utils import get_ablation_model_names, get_ablation_reference_model_name


DEFAULT_GRAPHS = ["causenet", "causalbank", "causenet_full"]
DEFAULT_TEST_DATASETS = [
    "data/datasets/filtered/msmarco_test_filtered.json",
    "data/datasets/filtered/sem_test_filtered.json",
]
DEFAULT_P95_DATASET = "data/datasets/filtered/msmarco_valid_filtered.json"
DEFAULT_ABLATION_CAP_SOURCE_DATASET = "msmarco_valid"
DEFAULT_ABLATION_CAP_SOURCE_GRAPH = "causenet"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Granite activation/distance ablation workflow: "
            "pre-embed, collect p95 visited-node caps, evaluate test sets, "
            "and generate ablation plots."
        )
    )
    parser.add_argument("--run-suffix", default="v3")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=graph_choices(),
        default=DEFAULT_GRAPHS,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_TEST_DATASETS,
        help="Test datasets to evaluate and visualize.",
    )
    parser.add_argument(
        "--p95-dataset",
        default=DEFAULT_P95_DATASET,
        help=(
            "Dataset used only when --run-ablation-visited-analysis is set. "
            "Ablation evaluation itself uses the main-model shared cap source."
        ),
    )
    parser.add_argument(
        "--ablation-cap-source-dataset",
        default=DEFAULT_ABLATION_CAP_SOURCE_DATASET,
        help=(
            "Normal evaluation dataset whose main-model p95 caps are shared "
            "by all ablation models."
        ),
    )
    parser.add_argument(
        "--ablation-cap-source-graph",
        choices=graph_choices(),
        default=DEFAULT_ABLATION_CAP_SOURCE_GRAPH,
        help=(
            "Normal evaluation graph namespace whose main-model p95 caps are "
            "shared by all ablation models."
        ),
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subprocess calls.",
    )
    parser.add_argument("--skip-preembed", action="store_true")
    parser.add_argument(
        "--run-ablation-visited-analysis",
        action="store_true",
        help=(
            "Optionally run visited-node analysis for the ablation models. "
            "Not needed for the default fixed-budget thesis comparison."
        ),
    )
    parser.add_argument(
        "--skip-visited",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument(
        "--skip-dijkstra",
        action="store_true",
        help="Forward --skip-dijkstra to evaluation.",
    )
    parser.add_argument(
        "--force-model-results",
        action="store_true",
        help="Forward --force-model-results to evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be greater than 0")
    if args.embedding_batch_size <= 0:
        parser.error("--embedding-batch-size must be greater than 0")

    return args


def run_command(command, dry_run=False):
    print("\n" + "=" * 80)
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(command, check=True)


def dataset_name_from_path(dataset_path):
    return Path(dataset_path).stem.replace("_filtered", "")


def get_ablation_p95_file(graph, p95_dataset, run_suffix):
    return (
        Path("data/evaluation")
        / "ablation"
        / graph
        / dataset_name_from_path(p95_dataset)
        / run_suffix
        / "visited_nodes_analysis.json"
    )


def build_visited_command(args, graph):
    return [
        args.python,
        "-m",
        "evaluation.visited_nodes_analysis",
        args.p95_dataset,
        "--run-suffix",
        args.run_suffix,
        "--graph",
        graph,
        "--ablation",
        "--dim",
        str(args.dim),
        "--embedding-device",
        args.embedding_device,
        "--embedding-batch-size",
        str(args.embedding_batch_size),
    ]


def ensure_ablation_p95_file(args, graph):
    p95_file = get_ablation_p95_file(graph, args.p95_dataset, args.run_suffix)

    if p95_file.exists():
        print(f"Found p95 config for {graph}: {p95_file}")
        return

    if args.skip_visited:
        command = build_visited_command(args, graph)
        command_text = " ".join(command)
        raise FileNotFoundError(
            f"Missing ablation p95 config for graph '{graph}': {p95_file}\n"
            "Run visited-node analysis first, or remove --skip-visited.\n"
            f"Command: {command_text}"
        )

    print(f"Missing p95 config for {graph}; running visited-node analysis now.")
    run_command(build_visited_command(args, graph), dry_run=args.dry_run)

    if not args.dry_run and not p95_file.exists():
        raise FileNotFoundError(
            f"Visited-node analysis finished but did not create expected file: {p95_file}"
        )


def main():
    args = parse_args()

    reference_model = get_ablation_reference_model_name(args.run_suffix)
    ablation_models = get_ablation_model_names(args.run_suffix)

    print("Ablation study configuration")
    print(f"Run suffix: {args.run_suffix}")
    print(f"Dim: {args.dim}")
    print(f"Graphs: {args.graphs}")
    print(f"Datasets: {args.datasets}")
    print(
        "Shared cap source: "
        f"{args.ablation_cap_source_graph}/"
        f"{args.ablation_cap_source_dataset}/{args.run_suffix}"
    )
    print(f"Optional ablation visited-analysis dataset: {args.p95_dataset}")
    print(f"Embedding device: {args.embedding_device}")
    print(f"Embedding batch size: {args.embedding_batch_size}")
    print(f"Reference model: {reference_model}")
    print("Ablation models:")
    for model_name in ablation_models:
        print(f"  - {model_name}")

    if not args.skip_preembed:
        preembed_targets = [
            [],
            ["--graph", "causenet_full"],
        ]

        for graph_args in preembed_targets:
            command = [
                args.python,
                "-m",
                "core.pre_embed",
                "--run-suffix",
                args.run_suffix,
                "--ablation",
                "--dim",
                str(args.dim),
                "--embedding-device",
                args.embedding_device,
                "--batch-size",
                str(args.embedding_batch_size),
                *graph_args,
            ]
            run_command(command, dry_run=args.dry_run)

    if args.run_ablation_visited_analysis and not args.skip_visited:
        for graph in args.graphs:
            ensure_ablation_p95_file(args, graph)

    if not args.skip_eval:
        for graph in args.graphs:
            for dataset in args.datasets:
                command = [
                    args.python,
                    "-m",
                    "evaluation.evaluation",
                    dataset,
                    "--run-suffix",
                    args.run_suffix,
                    "--graph",
                    graph,
                    "--ablation",
                    "--dim",
                    str(args.dim),
                    "--embedding-device",
                    args.embedding_device,
                    "--embedding-batch-size",
                    str(args.embedding_batch_size),
                    "--ablation-cap-source-dataset",
                    args.ablation_cap_source_dataset,
                    "--ablation-cap-source-graph",
                    args.ablation_cap_source_graph,
                ]

                if args.skip_dijkstra:
                    command.append("--skip-dijkstra")
                if args.force_model_results:
                    command.append("--force-model-results")

                run_command(command, dry_run=args.dry_run)

    if not args.skip_viz:
        for graph in args.graphs:
            for dataset in args.datasets:
                command = [
                    args.python,
                    "-m",
                    "evaluation.evaluation_viz",
                    dataset,
                    "--run-suffix",
                    args.run_suffix,
                    "--graph",
                    graph,
                    "--ablation",
                    "--dim",
                    str(args.dim),
                ]
                run_command(command, dry_run=args.dry_run)

    print("\nAblation study workflow complete.")


if __name__ == "__main__":
    main()
