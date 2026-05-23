import argparse
import json
from pathlib import Path


DEFAULT_MIN_F1 = 0.8
DEFAULT_VARIANT_FILTER = "finetuned"


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


def normalize_path(value):
    if value is None:
        return None

    return str(value).replace("\\", "/").rstrip("/")


def infer_model_family(model, model_path):
    text = f"{model} {model_path}".lower()

    if "qwen" in text:
        return "Qwen"
    if "mxbai" in text or "mixedbread" in text:
        return "MXBAI"
    if "mpnet" in text:
        return "MPNet"
    if "bge" in text:
        return "BGE"
    if "granite" in text:
        return "Granite"

    return Path(normalize_path(model_path)).name


def infer_model_variant(model, model_path):
    text = f"{model} {model_path}".lower()

    if "finetuned" in text:
        return "finetuned"

    return "base"


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
                "family": infer_model_family(model, model_path),
                "variant": infer_model_variant(model, model_path),
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


def candidate_sort_key(candidate):
    return (
        candidate["avg_time_ms"],
        candidate["avg_nodes_visited"],
        -candidate["f1_score"],
        -candidate["accuracy"],
        candidate["family"],
        candidate["dimension"],
    )


def filter_by_variant(candidates, variant_filter):
    if variant_filter is None:
        return candidates

    variant_filter = variant_filter.strip().lower()

    if variant_filter in {"", "all", "*"}:
        return candidates

    return [
        candidate
        for candidate in candidates
        if variant_filter in f"{candidate['model']} {candidate['model_path']}".lower()
    ]


def same_candidate(left, right):
    return (
        normalize_path(left["model_path"]) == normalize_path(right["model_path"])
        and left["dimension"] == right["dimension"]
    )


def build_family_summaries(pool_candidates, viable_candidates, selected, min_f1):
    summaries = []
    families = sorted({candidate["family"] for candidate in pool_candidates})

    for family in families:
        family_candidates = [
            candidate
            for candidate in pool_candidates
            if candidate["family"] == family
        ]
        family_viable = [
            candidate
            for candidate in viable_candidates
            if candidate["family"] == family
        ]

        best_effectiveness = max(
            family_candidates,
            key=lambda candidate: (
                candidate["f1_score"],
                candidate["accuracy"],
                -candidate["avg_time_ms"],
            ),
        )
        fastest_candidate = min(family_candidates, key=candidate_sort_key)
        fastest_viable = (
            min(family_viable, key=candidate_sort_key)
            if family_viable else None
        )

        if fastest_viable is None:
            decision = f"rejected: best F1 is below {min_f1:.2f}"
        elif same_candidate(fastest_viable, selected):
            decision = "selected: effective enough and fastest viable"
        elif fastest_viable["f1_score"] > selected["f1_score"]:
            decision = "more effective, but slower"
        else:
            decision = "effective enough, but slower"

        summaries.append(
            {
                "family": family,
                "num_candidates": len(family_candidates),
                "best_effectiveness": best_effectiveness,
                "fastest_candidate": fastest_candidate,
                "fastest_viable": fastest_viable,
                "decision": decision,
            }
        )

    return summaries


def select_best_astar_model(
    evaluation_results_path,
    min_f1=DEFAULT_MIN_F1,
    variant_filter=DEFAULT_VARIANT_FILTER,
):
    """
    Select the A* model/dimension using an effectiveness-efficiency tradeoff.

    Rule:
    1. Restrict the candidate pool to the requested model variant.
    2. Keep only candidates with acceptable effectiveness (F1 >= min_f1).
    3. Select the fastest viable candidate, using visited nodes and F1 as
       tie-breakers.
    """
    candidates = load_astar_candidates(evaluation_results_path)
    pool_candidates = filter_by_variant(candidates, variant_filter)

    if not pool_candidates:
        raise ValueError(
            f"No A* candidates matched variant filter {variant_filter!r}."
        )

    viable_candidates = [
        candidate
        for candidate in pool_candidates
        if candidate["f1_score"] >= min_f1
    ]

    if not viable_candidates:
        raise ValueError(
            f"No A* candidates in variant filter {variant_filter!r} reached "
            f"F1 >= {min_f1:.3f}."
        )

    ranked = sorted(viable_candidates, key=candidate_sort_key)
    selected = ranked[0]
    family_summaries = build_family_summaries(
        pool_candidates,
        viable_candidates,
        selected,
        min_f1,
    )

    return {
        "best": selected,
        "candidates": candidates,
        "pool_candidates": pool_candidates,
        "viable_candidates": viable_candidates,
        "ranked": ranked,
        "family_summaries": family_summaries,
        "min_f1": min_f1,
        "variant_filter": variant_filter,
        "selection_rule": "effectiveness_gate_then_fastest",
    }


def print_selection(selection_result, top_k=20):
    best = selection_result["best"]
    candidates = selection_result["candidates"]
    pool_candidates = selection_result["pool_candidates"]
    viable_candidates = selection_result["viable_candidates"]
    ranked = selection_result["ranked"]

    print("\nMODEL SELECTION")
    print("=" * 60)
    print(f"Total A* candidates: {len(candidates)}")
    print("Selection rule: effectiveness gate, then fastest viable candidate")
    print(f"Variant filter:      {selection_result['variant_filter']}")
    print(f"Candidates in pool:  {len(pool_candidates)}")
    print(f"Effectiveness gate:  F1 >= {selection_result['min_f1']:.3f}")
    print(f"Viable candidates:   {len(viable_candidates)}")
    print("Tie-breakers:        avg time, visited nodes, F1, accuracy")

    print("\nSELECTED A* MODEL")
    print("=" * 60)
    print(f"Family:             {best['family']}")
    print(f"Variant:            {best['variant']}")
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

    print("\nTEST EVALUATION PARAMETERS")
    print("=" * 60)
    print(f"--best-model-path {best['model_path']}")
    print(f"--best-model-dim {best['dimension']}")

    print("\nFAMILY TRADEOFF SUMMARY")
    print("=" * 60)

    for summary in selection_result["family_summaries"]:
        best_effectiveness = summary["best_effectiveness"]
        fastest_candidate = summary["fastest_candidate"]
        fastest_viable = summary["fastest_viable"]

        viable_text = "none"
        if fastest_viable is not None:
            viable_text = (
                f"dim={fastest_viable['dimension']}, "
                f"f1={fastest_viable['f1_score']:.6f}, "
                f"time={fastest_viable['avg_time_ms']:.2f} ms, "
                f"nodes={fastest_viable['avg_nodes_visited']:.2f}"
            )

        print(
            f"{summary['family']}: "
            f"best_f1=dim {best_effectiveness['dimension']} "
            f"({best_effectiveness['f1_score']:.6f}, "
            f"{best_effectiveness['avg_time_ms']:.2f} ms); "
            f"fastest=dim {fastest_candidate['dimension']} "
            f"({fastest_candidate['f1_score']:.6f}, "
            f"{fastest_candidate['avg_time_ms']:.2f} ms); "
            f"fastest_viable={viable_text}; "
            f"{summary['decision']}"
        )

    print(f"\nRANKED VIABLE A* CANDIDATES")
    print("=" * 60)

    for i, candidate in enumerate(ranked[:top_k], start=1):
        print(
            f"{i:02d}. "
            f"{candidate['family']} | "
            f"{candidate['model']} | "
            f"dim={candidate['dimension']} | "
            f"f1={candidate['f1_score']:.6f} | "
            f"time={candidate['avg_time_ms']:.2f} ms | "
            f"nodes={candidate['avg_nodes_visited']:.2f} | "
            f"accuracy={candidate['accuracy']:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the A* model using an effectiveness-efficiency tradeoff."
    )
    parser.add_argument(
        "evaluation_results_path",
        nargs="?",
        default="data/evaluation/causenet/msmarco_valid/best_v2/evaluation_results.json",
        help="Path to validation evaluation_results.json.",
    )
    parser.add_argument(
        "--variant-filter",
        type=str,
        default=DEFAULT_VARIANT_FILTER,
        help=(
            "Case-insensitive model/path substring used to define the candidate "
            "pool. Use 'all' to include base and fine-tuned models."
        ),
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=DEFAULT_MIN_F1,
        help="Minimum validation F1 required before efficiency is considered.",
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
        min_f1=args.min_f1,
        variant_filter=args.variant_filter,
    )
    print_selection(selection, top_k=args.top_k)
