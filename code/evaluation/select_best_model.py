import argparse
import json
import math
from pathlib import Path


def get_metric(metrics, key, default=0.0):
    value = metrics.get(key)

    if value is None:
        return default

    return value


def get_avg_time_ms(metrics):
    if metrics.get("avg_time_ms") is not None:
        return metrics["avg_time_ms"]

    if metrics.get("avg_time_sec") is not None:
        return metrics["avg_time_sec"] * 1000.0

    return float("inf")


def get_astar_metrics(entry):
    evaluation = entry.get("evaluation", {})

    if "A*" not in evaluation:
        return None

    return evaluation["A*"].get("metrics", {})


def load_astar_candidates(evaluation_results_path):
    evaluation_results_path = Path(evaluation_results_path)

    if not evaluation_results_path.exists():
        raise FileNotFoundError(f"File not found: {evaluation_results_path}")

    with open(evaluation_results_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    candidates = []

    for entry in results:
        metrics = get_astar_metrics(entry)

        if metrics is None:
            continue

        model = entry.get("model")
        model_path = entry.get("model_path")
        dimension = entry.get("dimension")

        # Skip baselines and incomplete entries.
        if model is None or model_path is None or dimension is None:
            continue

        candidates.append(
            {
                "model": model,
                "model_path": model_path,
                "dimension": dimension,
                "f1_score": get_metric(metrics, "f1_score"),
                "accuracy": get_metric(metrics, "accuracy"),
                "recall": get_metric(metrics, "recall"),
                "precision": get_metric(metrics, "precision"),
                "avg_nodes_visited": get_metric(
                    metrics,
                    "avg_nodes_visited",
                    float("inf"),
                ),
                "avg_time_ms": get_avg_time_ms(metrics),
                "num_examples": get_metric(metrics, "num_examples", 0),
            }
        )

    if not candidates:
        raise ValueError("No A* candidates found.")

    return candidates


def select_best_astar_model(
    evaluation_results_path,
    f1_tolerance=0.02,
):
    """
    Select the A* model/dimension used for test evaluation.

    Default rule:
    1. Find the best validation F1 among A* candidates.
    2. Choose the fastest candidate, then fewer visited nodes, then higher F1.
    """
    candidates = load_astar_candidates(evaluation_results_path)

    best_f1 = max(candidate["f1_score"] for candidate in candidates)
    f1_cutoff = best_f1 - f1_tolerance

    eligible_candidates = [
        candidate for candidate in candidates if candidate["f1_score"] >= f1_cutoff
    ]

    ranked = sorted(
        eligible_candidates,
        key=lambda x: (
            x["avg_time_ms"],
            x["avg_nodes_visited"],
            -x["f1_score"],
        ),
    )

    best = ranked[0]

    return {
        "best": best,
        "candidates": candidates,
        "eligible_candidates": eligible_candidates,
        "efficient_pool": eligible_candidates,
        "ranked": ranked,
        "best_f1": best_f1,
        "f1_cutoff": f1_cutoff,
        "f1_tolerance": f1_tolerance,
        "selection_rule": "f1_constrained_fastest",
    }


def print_selection(selection_result, top_k=20):
    best = selection_result["best"]
    candidates = selection_result["candidates"]
    ranked = selection_result["ranked"]
    selection_rule = selection_result.get(
        "selection_rule",
        "legacy_efficiency_pool",
    )

    print("\nMODEL SELECTION")
    print("=" * 60)
    print(f"Total A* candidates: {len(candidates)}")

    if selection_rule == "f1_constrained_fastest":
        eligible_candidates = selection_result["eligible_candidates"]
        print("Selection rule: F1-constrained fastest candidate")
        print(f"Best validation F1: {selection_result['best_f1']:.6f}")
        print(f"F1 tolerance:       {selection_result['f1_tolerance']:.6f}")
        print(f"F1 cutoff:          {selection_result['f1_cutoff']:.6f}")
        print(f"Eligible models:    {len(eligible_candidates)}")
    else:
        efficient_pool = selection_result["efficient_pool"]
        print("Selection rule: cheapest fraction, then highest F1")
        print(
            f"Efficiency pool fraction: {selection_result['efficiency_pool_fraction']:.2f}"
        )
        print(f"Efficiency pool size: {selection_result['pool_size']}")

    print("\nVALIDATION-SELECTED A* MODEL")
    print("=" * 60)
    print(f"Model:              {best['model']}")
    print(f"Model path:         {best['model_path']}")
    print(f"Dimension:          {best['dimension']}")
    print(f"F1:                 {best['f1_score']:.6f}")
    print(f"Accuracy:           {best['accuracy']:.6f}")
    print(f"Recall:             {best['recall']:.6f}")
    print(f"Precision:          {best['precision']:.6f}")
    print(f"Avg visited nodes:  {best['avg_nodes_visited']:.2f}")
    print(f"Avg time ms:        {best['avg_time_ms']:.2f}")
    print(f"Num examples:       {best['num_examples']}")

    if selection_rule == "f1_constrained_fastest":
        print(f"\nFASTEST ELIGIBLE A* CANDIDATES")
        display_candidates = ranked
    else:
        efficient_pool = selection_result["efficient_pool"]
        print(f"\nCHEAPEST {len(efficient_pool)} A* CANDIDATES")
        display_candidates = efficient_pool
    print("=" * 60)

    for i, candidate in enumerate(display_candidates[:top_k], start=1):
        print(
            f"{i:02d}. "
            f"{candidate['model']} | "
            f"dim={candidate['dimension']} | "
            f"f1={candidate['f1_score']:.6f} | "
            f"time={candidate['avg_time_ms']:.2f} ms | "
            f"nodes={candidate['avg_nodes_visited']:.2f} | "
            f"accuracy={candidate['accuracy']:.6f}"
        )

    if selection_rule != "f1_constrained_fastest":
        print("\nRANKED WITHIN EFFICIENCY POOL")
        print("=" * 60)

        for i, candidate in enumerate(ranked[:top_k], start=1):
            print(
                f"{i:02d}. "
                f"{candidate['model']} | "
                f"dim={candidate['dimension']} | "
                f"f1={candidate['f1_score']:.6f} | "
                f"time={candidate['avg_time_ms']:.2f} ms | "
                f"nodes={candidate['avg_nodes_visited']:.2f} | "
                f"accuracy={candidate['accuracy']:.6f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the validation-best A* model for test evaluation."
    )
    parser.add_argument(
        "evaluation_results_path",
        nargs="?",
        default="data/evaluation/msmarco_valid/best_v2/evaluation_results.json",
        help="Path to validation evaluation_results.json.",
    )
    parser.add_argument(
        "--f1-tolerance",
        type=float,
        default=0.02,
        help="Absolute F1 loss allowed from the best validation F1.",
    )
    parser.add_argument(
        "--legacy-efficiency-pool-fraction",
        type=float,
        default=None,
        help=(
            "Use the old cheapest-fraction-then-F1 rule with this fraction "
            "instead of the default F1-constrained fastest rule."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of ranked candidates to print.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selection = select_best_astar_model(
        args.evaluation_results_path,
        efficiency_pool_fraction=args.legacy_efficiency_pool_fraction,
        f1_tolerance=args.f1_tolerance,
    )
    print_selection(selection, top_k=args.top_k)
