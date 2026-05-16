import json
import math
from pathlib import Path


def get_astar_metrics(entry):
    evaluation = entry.get("evaluation", {})

    if "A*" not in evaluation:
        return None

    return evaluation["A*"].get("metrics", {})


def select_best_astar_model(
    evaluation_results_path,
    efficiency_pool_fraction=0.25,
):
    evaluation_results_path = Path(evaluation_results_path)

    if not evaluation_results_path.exists():
        raise FileNotFoundError(
            f"File not found: {evaluation_results_path}"
        )

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
                "f1_score": metrics.get("f1_score", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "recall": metrics.get("recall", 0.0),
                "precision": metrics.get("precision", 0.0),
                "avg_nodes_visited": metrics.get(
                    "avg_nodes_visited",
                    float("inf"),
                ),
                "avg_time_ms": metrics.get(
                    "avg_time_ms",
                    float("inf"),
                ),
                "num_examples": metrics.get("num_examples", 0),
            }
        )

    if not candidates:
        raise ValueError(
            "No A* candidates found."
        )

    cheapest_candidates = sorted(
        candidates,
        key=lambda x: (
            x["avg_time_ms"],
            x["avg_nodes_visited"],
            -x["f1_score"],
        ),
    )

    pool_size = max(
        1,
        math.ceil(len(candidates) * efficiency_pool_fraction),
    )

    efficient_pool = cheapest_candidates[:pool_size]

    ranked = sorted(
        efficient_pool,
        key=lambda x: (
            -x["f1_score"],
            x["avg_time_ms"],
            x["avg_nodes_visited"],
        ),
    )

    best = ranked[0]

    return {
        "best": best,
        "candidates": candidates,
        "efficient_pool": efficient_pool,
        "ranked": ranked,
        "pool_size": pool_size,
        "efficiency_pool_fraction": efficiency_pool_fraction,
    }


def print_selection(selection_result, top_k=20):
    best = selection_result["best"]
    candidates = selection_result["candidates"]
    efficient_pool = selection_result["efficient_pool"]
    ranked = selection_result["ranked"]

    print("\nEFFICIENCY POOL")
    print("=" * 60)
    print(f"Total A* candidates: {len(candidates)}")
    print(f"Efficiency pool fraction: {selection_result['efficiency_pool_fraction']:.2f}")
    print(f"Efficiency pool size: {selection_result['pool_size']}")
    print(
        "Selection rule: cheapest 25% by avg visited nodes, "
        "then highest F1"
    )

    print("\nBEST MODEL FOR TEST")
    print("=" * 60)
    print(f"Model:              {best['model']}")
    print(f"Model path:         {best['model_path']}")
    print(f"Dimension:          {best['dimension']}")
    print(f"F1:                 {best['f1_score']:.6f}")
    print(f"Accuracy:           {best['accuracy']:.6f}")
    print(f"Recall:             {best['recall']:.6f}")
    print(f"Precision:          {best['precision']:.6f}")
    print(f"Avg visited nodes:  {best['avg_nodes_visited']:.2f}")
    print(f"Avg time sec:       {best['avg_time_ms']:.6f}")
    print(f"Num examples:       {best['num_examples']}")

    print(f"\nCHEAPEST {len(efficient_pool)} A* CANDIDATES")
    print("=" * 60)

    for i, candidate in enumerate(efficient_pool, start=1):
        print(
            f"{i:02d}. "
            f"{candidate['model']} | "
            f"dim={candidate['dimension']} | "
            f"nodes={candidate['avg_nodes_visited']:.2f} | "
            f"time={candidate['avg_time_ms']:.6f} | "
            f"f1={candidate['f1_score']:.6f} | "
            f"accuracy={candidate['accuracy']:.6f}"
        )

    print("\nRANKED WITHIN EFFICIENCY POOL")
    print("=" * 60)

    for i, candidate in enumerate(ranked[:top_k], start=1):
        print(
            f"{i:02d}. "
            f"{candidate['model']} | "
            f"dim={candidate['dimension']} | "
            f"f1={candidate['f1_score']:.6f} | "
            f"nodes={candidate['avg_nodes_visited']:.2f} | "
            f"time={candidate['avg_time_ms']:.6f} | "
            f"accuracy={candidate['accuracy']:.6f}"
        )


if __name__ == "__main__":
    selection = select_best_astar_model(
        "data/evaluation/msmarco_valid/evaluation_results.json"
    )
    print_selection(selection)