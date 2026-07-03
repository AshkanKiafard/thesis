import argparse
import subprocess
import sys
from pathlib import Path

from core.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_P95_CONFIG_SOURCE_DATASET,
    DEFAULT_P95_CONFIG_SOURCE_GRAPH,
    DEFAULT_RUN_SUFFIX,
    DEFAULT_TEST_DATASETS,
    DEFAULT_TEST_GRAPHS,
    DEFAULT_VALIDATION_DATASET,
    DEFAULT_VALIDATION_GRAPH,
)
from core.graph_config import graph_choices
from evaluation.select_best_model import print_selection, select_best_astar_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the main evaluation workflow: all models on MSMARCO validation "
            "with CauseNet, then the selected best validation model on MSMARCO "
            "test and SemEval test across the default three graphs."
        )
    )
    parser.add_argument("--run-suffix", default=DEFAULT_RUN_SUFFIX)
    parser.add_argument(
        "--validation-graph",
        choices=graph_choices(),
        default=DEFAULT_VALIDATION_GRAPH,
        help="Graph used for the full MSMARCO validation run and model selection.",
    )
    parser.add_argument(
        "--test-graphs",
        nargs="+",
        choices=graph_choices(),
        default=DEFAULT_TEST_GRAPHS,
        help="Graphs used for the best-model test evaluations.",
    )
    parser.add_argument(
        "--validation-dataset",
        default=DEFAULT_VALIDATION_DATASET,
        help="Dataset used for the full validation evaluation.",
    )
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=DEFAULT_TEST_DATASETS,
        help="Datasets used for the best-model test evaluations.",
    )
    parser.add_argument(
        "--test-config-source-dataset",
        default=DEFAULT_P95_CONFIG_SOURCE_DATASET,
        help=(
            "Visited-node p95 dataset source forced for best-model test "
            "evaluations."
        ),
    )
    parser.add_argument(
        "--test-config-source-graph",
        choices=graph_choices(),
        default=DEFAULT_P95_CONFIG_SOURCE_GRAPH,
        help=(
            "Visited-node p95 graph source forced for best-model test "
            "evaluations."
        ),
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for subprocess calls.",
    )
    parser.add_argument(
        "--variant-filter",
        default="finetuned",
        help="Model/path substring used by select_best_model.py.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.8,
        help="Minimum validation F1 for best-model selection.",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not pass force flags to evaluation.py.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Reuse existing validation results instead of rerunning validation.",
    )
    parser.add_argument(
        "--skip-dijkstra",
        action="store_true",
        help=(
            "Skip Dijkstra in every evaluation command. A* model results and "
            "baselines can still be forced normally."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    if args.embedding_batch_size <= 0:
        parser.error("--embedding-batch-size must be greater than 0")
    if args.min_f1 < 0 or args.min_f1 > 1:
        parser.error("--min-f1 must be between 0 and 1")

    return args


def run_command(command, dry_run=False):
    print("\n" + "=" * 80)
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(command, check=True)


def force_args(args):
    if args.no_force:
        return []

    return ["--force-baselines", "--force-model-results"]


def evaluation_command(args, dataset, graph, extra_args=None):
    command = [
        args.python,
        "-m",
        "evaluation.evaluation",
        dataset,
        "--run-suffix",
        args.run_suffix,
        "--graph",
        graph,
        "--embedding-device",
        args.embedding_device,
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        *force_args(args),
    ]

    if args.skip_dijkstra:
        command.append("--skip-dijkstra")

    if extra_args:
        command.extend(extra_args)

    return command


def validation_results_path(args):
    dataset_name = Path(args.validation_dataset).stem.replace("_filtered", "")
    return (
        Path("data/evaluation")
        / args.validation_graph
        / dataset_name
        / args.run_suffix
        / "evaluation_results.json"
    )


def select_best_model(args):
    results_path = validation_results_path(args)

    print(f"\nBest-model selection source: {results_path}")

    if args.dry_run and not results_path.exists():
        print("Dry run: validation results do not exist yet; using placeholders.")
        return {
            "model_path": "<selected-best-model-path>",
            "dimension": "<selected-best-model-dim>",
        }
    if args.dry_run:
        print("Dry run: selecting from existing validation results.")

    selection = select_best_astar_model(
        results_path,
        min_f1=args.min_f1,
        variant_filter=args.variant_filter,
    )
    print_selection(selection)

    return selection["best"]


def main():
    args = parse_args()

    print("Main evaluation workflow")
    print(f"Run suffix: {args.run_suffix}")
    print(f"Validation graph: {args.validation_graph}")
    print(f"Validation dataset: {args.validation_dataset}")
    print(f"Test graphs: {args.test_graphs}")
    print(f"Test datasets: {args.test_datasets}")
    print(
        "Test p95 source: "
        f"{args.test_config_source_graph}/"
        f"{args.test_config_source_dataset}/{args.run_suffix}"
    )
    print(f"Embedding device: {args.embedding_device}")
    print(f"Embedding batch size: {args.embedding_batch_size}")
    print(f"Force results: {not args.no_force}")
    print(f"Skip Dijkstra: {args.skip_dijkstra}")

    if not args.skip_validation:
        run_command(
            evaluation_command(
                args,
                args.validation_dataset,
                args.validation_graph,
            ),
            dry_run=args.dry_run,
        )
    else:
        print("\nSkipping validation evaluation and reusing existing results.")

    best_model = select_best_model(args)
    best_model_args = [
        "--best-model-path",
        best_model["model_path"],
        "--best-model-dim",
        str(best_model["dimension"]),
        "--config-source-dataset",
        args.test_config_source_dataset,
        "--config-source-graph",
        args.test_config_source_graph,
    ]

    for graph in args.test_graphs:
        for dataset in args.test_datasets:
            run_command(
                evaluation_command(
                    args,
                    dataset,
                    graph,
                    extra_args=best_model_args,
                ),
                dry_run=args.dry_run,
            )

    print("\nMain evaluation workflow complete.")


if __name__ == "__main__":
    main()
