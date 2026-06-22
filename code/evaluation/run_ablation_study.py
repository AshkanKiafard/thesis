import argparse
import json
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
DEFAULT_ABLATION_CAP_SOURCE_DATASET = "msmarco_train"
DEFAULT_ABLATION_CAP_SOURCE_GRAPH = "causenet"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Granite activation/distance ablation workflow: "
            "pre-embed, evaluate test sets with a shared main-model A* cap, "
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
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
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


def get_shared_cap_source_file(args):
    return (
        Path("data/evaluation")
        / args.ablation_cap_source_graph
        / args.ablation_cap_source_dataset
        / args.run_suffix
        / "visited_nodes_analysis.json"
    )


def validate_shared_cap_source(args, reference_model):
    cap_file = get_shared_cap_source_file(args)

    if args.dry_run:
        print(f"Expected shared cap source: {cap_file}")
        return

    if not cap_file.exists():
        dataset_path = (
            f"data/datasets/filtered/"
            f"{args.ablation_cap_source_dataset}_filtered.json"
        )
        command = " ".join(
            [
                args.python,
                "-m",
                "evaluation.visited_nodes_analysis",
                dataset_path,
                "--run-suffix",
                args.run_suffix,
                "--graph",
                args.ablation_cap_source_graph,
            ]
        )
        raise FileNotFoundError(
            "Missing shared main-model cap source for ablation evaluation:\n"
            f"{cap_file}\n\n"
            "This is not an ablation visited-node file. It is the normal "
            "reference cap file used to keep the ablation comparison fixed-budget.\n"
            "Create it once with:\n"
            f"{command}"
        )

    with open(cap_file, "r", encoding="utf-8") as file:
        analysis_results = json.load(file)

    has_reference_cap = any(
        entry.get("model") == reference_model
        and entry.get("dimension") == args.dim
        and entry.get("analysis", {}).get("strategy") == "A*"
        and entry.get("analysis", {}).get("p95_visited_successful_only") is not None
        for entry in analysis_results
    )

    if not has_reference_cap:
        raise KeyError(
            "Shared cap source exists, but it does not contain the required "
            f"A* cap for {reference_model} dim {args.dim}:\n{cap_file}"
        )

    print(f"Found shared main-model A* cap source: {cap_file}")


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
    print(f"Embedding device: {args.embedding_device}")
    print(f"Embedding batch size: {args.embedding_batch_size}")
    print(f"Reference model: {reference_model}")
    print("Ablation models:")
    for model_name in ablation_models:
        print(f"  - {model_name}")

    if not args.skip_eval:
        validate_shared_cap_source(args, reference_model)

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
                    "--skip-dijkstra",
                ]

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
