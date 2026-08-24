import argparse
import json
import math
from pathlib import Path

from core.config import DEFAULT_RUN_SUFFIX
from core.constants import EVALUATION_DIR

DEFAULT_VARIANT_FILTER = "finetuned"
DEFAULT_BUDGET_MODE_FILTER = "capped"
SELECTION_RULE = "pareto_knee_f1_log_nodes"
KNEE_SCORE_EPSILON = 1e-12


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
                "astar_max_visits": entry.get("used_config", {}).get(
                    "astar_max_visits"
                ),
                "budget_mode": (
                    "uncapped"
                    if entry.get("used_config", {}).get("astar_max_visits") == -1
                    else "capped"
                ),
            }
        )

    if not candidates:
        raise ValueError("No A* candidates found.")

    return candidates


def runtime_tiebreak_key(candidate):
    return (
        candidate["avg_time_ms"],
        candidate["astar_max_visits"]
        if candidate["astar_max_visits"] is not None
        else float("inf"),
        candidate["dimension"],
        -candidate["f1_score"],
        candidate["avg_nodes_visited"],
        candidate["family"],
        normalize_path(candidate["model_path"]),
    )


def dominates(left, right):
    """Return whether left is at least as effective and no more expensive."""
    no_worse = (
        left["f1_score"] >= right["f1_score"]
        and left["avg_nodes_visited"] <= right["avg_nodes_visited"]
    )
    strictly_better = (
        left["f1_score"] > right["f1_score"]
        or left["avg_nodes_visited"] < right["avg_nodes_visited"]
    )
    return no_worse and strictly_better


def build_pareto_front(candidates):
    """Build the F1/visited-node frontier without imposing an F1 cliff."""
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
    ]
    return sorted(
        frontier,
        key=lambda candidate: (
            candidate["avg_nodes_visited"],
            candidate["f1_score"],
            runtime_tiebreak_key(candidate),
        ),
    )


def rank_pareto_knee(candidates):
    """Rank the Pareto frontier by its F1/log-visited-nodes knee.

    When the frontier has no positive interior knee, prefer its most effective
    endpoint instead of mistaking the cheapest low-F1 endpoint for a knee.
    """
    frontier = build_pareto_front(candidates)
    if not frontier:
        raise ValueError("No candidates available for Pareto-knee selection.")

    for candidate in frontier:
        nodes = candidate["avg_nodes_visited"]
        f1_score = candidate["f1_score"]
        if not math.isfinite(nodes) or nodes <= 0:
            raise ValueError(
                "Pareto-knee selection requires positive finite average "
                f"visited nodes, got {nodes!r} for {candidate['model']}."
            )
        if not math.isfinite(f1_score) or not 0 <= f1_score <= 1:
            raise ValueError(
                "Pareto-knee selection requires finite F1 in [0, 1], "
                f"got {f1_score!r} for {candidate['model']}."
            )

    if len(frontier) == 1:
        selected = dict(frontier[0])
        selected.update(
            knee_score=0.0,
            normalized_f1=1.0,
            normalized_log_nodes=0.0,
        )
        return [selected]

    log_nodes = [
        math.log(candidate["avg_nodes_visited"])
        for candidate in frontier
    ]
    f1_scores = [candidate["f1_score"] for candidate in frontier]
    log_nodes_span = max(log_nodes) - min(log_nodes)
    f1_span = max(f1_scores) - min(f1_scores)

    if log_nodes_span == 0 or f1_span == 0:
        selected = dict(
            max(
                frontier,
                key=lambda candidate: (
                    candidate["f1_score"],
                    -candidate["avg_nodes_visited"],
                ),
            )
        )
        selected.update(
            knee_score=0.0,
            normalized_f1=1.0,
            normalized_log_nodes=0.0,
        )
        return [selected]

    ranked = []
    for candidate, candidate_log_nodes in zip(frontier, log_nodes):
        normalized_f1 = (
            candidate["f1_score"] - min(f1_scores)
        ) / f1_span
        normalized_log_nodes = (
            candidate_log_nodes - min(log_nodes)
        ) / log_nodes_span
        annotated = dict(candidate)
        annotated.update(
            knee_score=(normalized_f1 - normalized_log_nodes)
            / math.sqrt(2.0),
            normalized_f1=normalized_f1,
            normalized_log_nodes=normalized_log_nodes,
        )
        ranked.append(annotated)

    max_knee_score = max(candidate["knee_score"] for candidate in ranked)
    if len(ranked) == 2 or max_knee_score <= KNEE_SCORE_EPSILON:
        return sorted(
            ranked,
            key=lambda candidate: (
                -candidate["f1_score"],
                runtime_tiebreak_key(candidate),
            ),
        )

    tied_knees = [
        candidate
        for candidate in ranked
        if max_knee_score - candidate["knee_score"] <= KNEE_SCORE_EPSILON
    ]
    remaining = [
        candidate
        for candidate in ranked
        if max_knee_score - candidate["knee_score"] > KNEE_SCORE_EPSILON
    ]
    return sorted(tied_knees, key=runtime_tiebreak_key) + sorted(
        remaining,
        key=lambda candidate: (
            -candidate["knee_score"],
            runtime_tiebreak_key(candidate),
        ),
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


def filter_by_budget_mode(candidates, budget_mode_filter):
    if budget_mode_filter is None:
        return candidates

    budget_mode_filter = budget_mode_filter.strip().lower()
    if budget_mode_filter in {"", "all", "*"}:
        return candidates

    return [
        candidate
        for candidate in candidates
        if candidate["budget_mode"] == budget_mode_filter
    ]


def build_family_summaries(pool_candidates, selected):
    summaries = []
    families = sorted({candidate["family"] for candidate in pool_candidates})

    for family in families:
        family_candidates = [
            candidate
            for candidate in pool_candidates
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
        fastest_candidate = min(
            family_candidates,
            key=runtime_tiebreak_key,
        )
        family_ranked = rank_pareto_knee(family_candidates)
        family_knee = family_ranked[0]
        is_selected_family = family == selected["family"]
        tradeoff_candidate = selected if is_selected_family else family_knee

        if is_selected_family:
            decision = "selected: global Pareto knee"
        else:
            decision = "family Pareto knee; not selected globally"

        summaries.append(
            {
                "family": family,
                "num_candidates": len(family_candidates),
                "best_effectiveness": best_effectiveness,
                "fastest_candidate": fastest_candidate,
                "family_knee": family_knee,
                "tradeoff_candidate": tradeoff_candidate,
                "pareto_candidates": family_ranked,
                "decision": decision,
            }
        )

    return summaries


def select_best_astar_model(
    evaluation_results_path,
    variant_filter=DEFAULT_VARIANT_FILTER,
    budget_mode_filter=DEFAULT_BUDGET_MODE_FILTER,
):
    """
    Select the A* model/dimension using an effectiveness-efficiency tradeoff.

    Rule:
    1. Restrict the candidate pool to the requested model variant.
    2. Remove candidates dominated in validation F1 and average visited nodes.
    3. Select the knee of the normalized F1/log-visited-nodes Pareto frontier.
    4. Use measured runtime, p95 cap, and dimension only as tie-breakers.
    """
    candidates = load_astar_candidates(evaluation_results_path)
    pool_candidates = filter_by_variant(candidates, variant_filter)
    pool_candidates = filter_by_budget_mode(
        pool_candidates,
        budget_mode_filter,
    )

    if not pool_candidates:
        raise ValueError(
            f"No A* candidates matched variant filter {variant_filter!r}."
        )

    ranked = rank_pareto_knee(pool_candidates)
    selected = ranked[0]
    family_summaries = build_family_summaries(
        pool_candidates,
        selected,
    )

    return {
        "best": selected,
        "candidates": candidates,
        "pool_candidates": pool_candidates,
        "pareto_candidates": ranked,
        "ranked": ranked,
        "family_summaries": family_summaries,
        "variant_filter": variant_filter,
        "budget_mode_filter": budget_mode_filter,
        "selection_rule": SELECTION_RULE,
    }


def print_selection(selection_result, top_k=20):
    best = selection_result["best"]
    candidates = selection_result["candidates"]
    pool_candidates = selection_result["pool_candidates"]
    ranked = selection_result["ranked"]

    print("\nMODEL SELECTION")
    print("=" * 60)
    print(f"Total A* candidates: {len(candidates)}")
    print("Selection rule: F1/log-visited-nodes Pareto knee")
    print(f"Variant filter:      {selection_result['variant_filter']}")
    print(f"Budget mode filter:  {selection_result['budget_mode_filter']}")
    print(f"Candidates in pool:  {len(pool_candidates)}")
    print(f"Pareto candidates:   {len(ranked)}")
    print("Primary objectives:  maximize F1, minimize average visited nodes")
    print("Knee axes:           normalized F1 and log(average visited nodes)")
    print("No-knee fallback:    most effective frontier endpoint")
    print("Tie-breakers:        runtime, p95 cap, dimension, F1, nodes")

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
    print(f"p95 visit budget:   {best['astar_max_visits']}")
    print(f"Pareto knee score:  {best['knee_score']:.6f}")
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
        family_knee = summary["family_knee"]
        tradeoff_candidate = summary["tradeoff_candidate"]

        print(
            f"{summary['family']}: "
            f"best_f1=dim {best_effectiveness['dimension']} "
            f"({best_effectiveness['f1_score']:.6f}, "
            f"{best_effectiveness['avg_time_ms']:.2f} ms); "
            f"fastest=dim {fastest_candidate['dimension']} "
            f"({fastest_candidate['f1_score']:.6f}, "
            f"{fastest_candidate['avg_time_ms']:.2f} ms); "
            f"family_knee=dim {family_knee['dimension']} "
            f"(f1={family_knee['f1_score']:.6f}, "
            f"nodes={family_knee['avg_nodes_visited']:.2f}, "
            f"time={family_knee['avg_time_ms']:.2f} ms); "
            f"tradeoff_config=dim {tradeoff_candidate['dimension']}; "
            f"{summary['decision']}"
        )

    print("\nRANKED PARETO A* CANDIDATES")
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
            f"knee={candidate['knee_score']:.6f} | "
            f"accuracy={candidate['accuracy']:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the A* model using an effectiveness-efficiency tradeoff."
    )
    parser.add_argument(
        "evaluation_results_path",
        nargs="?",
        default=str(
            EVALUATION_DIR
            / "causenet"
            / "msmarco_valid"
            / DEFAULT_RUN_SUFFIX
            / "evaluation_results.json"
        ),
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
        "--top-k",
        type=int,
        default=20,
        help="Number of ranked candidates to print.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=("capped", "uncapped", "all"),
        default=DEFAULT_BUDGET_MODE_FILTER,
        help=(
            "A* budget variant considered during model selection. Defaults "
            "to capped so uncapped validation runs cannot affect selection."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selection = select_best_astar_model(
        args.evaluation_results_path,
        variant_filter=args.variant_filter,
        budget_mode_filter=args.budget_mode,
    )
    print_selection(selection, top_k=args.top_k)
