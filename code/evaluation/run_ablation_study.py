import argparse
import subprocess
import sys

from core.graph_config import graph_choices
from core.utils import get_ablation_model_names, get_ablation_reference_model_name


DEFAULT_GRAPHS = ["causenet", "causalbank", "causenet_full"]
DEFAULT_TEST_DATASETS = [
    "data/datasets/filtered/msmarco_test_filtered.json",
    "data/datasets/filtered/sem_test_filtered.json",
]
DEFAULT_P95_DATASET = "data/datasets/filtered/msmarco_train_filtered.json"


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
        help="Training split used to collect uncapped visited-node p95 caps.",
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
    parser.add_argument("--skip-visited", action="store_true")
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


def main():
    args = parse_args()

    reference_model = get_ablation_reference_model_name(args.run_suffix)
    ablation_models = get_ablation_model_names(args.run_suffix)

    print("Ablation study configuration")
    print(f"Run suffix: {args.run_suffix}")
    print(f"Dim: {args.dim}")
    print(f"Graphs: {args.graphs}")
    print(f"Datasets: {args.datasets}")
    print(f"P95 dataset: {args.p95_dataset}")
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

    if not args.skip_visited:
        for graph in args.graphs:
            command = [
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
                    "--fallback-config-source-dataset",
                    "msmarco_train",
                    "--fallback-config-source-graph",
                    graph,
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
