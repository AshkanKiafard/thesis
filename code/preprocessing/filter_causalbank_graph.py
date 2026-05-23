import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CAUSALBANK_GRAPH_PATH = (
    REPO_ROOT / "data" / "graphs" / "Lexical_Cause_Effect_Graph.txt"
)
FILTERED_CAUSALBANK_GRAPH_PATH = (
    REPO_ROOT / "data" / "graphs" / "Lexical_Cause_Effect_Graph.filtered.txt"
)

DEFAULT_MIN_COUNT = 0
DEFAULT_MIN_CAUSAL_STRENGTH = 0.10
DEFAULT_NECESSITY_WEIGHT = 0.90


def normalize_causalbank_concept(value):
    return value.replace("_", " ").strip().lower()


def compute_causal_strength(
    necessity_score,
    sufficiency_score,
    necessity_weight=DEFAULT_NECESSITY_WEIGHT,
):
    return (
        necessity_score ** necessity_weight
        * sufficiency_score ** (1.0 - necessity_weight)
    )


def edge_passes_filter(
    count,
    necessity_score,
    sufficiency_score,
    min_count=DEFAULT_MIN_COUNT,
    min_causal_strength=DEFAULT_MIN_CAUSAL_STRENGTH,
    necessity_weight=DEFAULT_NECESSITY_WEIGHT,
):
    if count < min_count:
        return False

    causal_strength = compute_causal_strength(
        necessity_score=necessity_score,
        sufficiency_score=sufficiency_score,
        necessity_weight=necessity_weight,
    )
    return causal_strength >= min_causal_strength


def iter_filtered_causalbank_edges(
    input_path,
    min_count=DEFAULT_MIN_COUNT,
    min_causal_strength=DEFAULT_MIN_CAUSAL_STRENGTH,
    necessity_weight=DEFAULT_NECESSITY_WEIGHT,
    remove_self_loops=True,
):
    with open(input_path, encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 4 or "->" not in parts[0]:
                continue

            try:
                count = int(parts[1])
                necessity_score = float(parts[2])
                sufficiency_score = float(parts[3])
            except ValueError:
                continue

            if not edge_passes_filter(
                count=count,
                necessity_score=necessity_score,
                sufficiency_score=sufficiency_score,
                min_count=min_count,
                min_causal_strength=min_causal_strength,
                necessity_weight=necessity_weight,
            ):
                continue

            cause, effect = parts[0].split("->", 1)
            cause = normalize_causalbank_concept(cause)
            effect = normalize_causalbank_concept(effect)

            if not cause or not effect:
                continue

            if remove_self_loops and cause == effect:
                continue

            yield cause, effect, count, necessity_score, sufficiency_score


def filter_causalbank_graph(
    input_path=RAW_CAUSALBANK_GRAPH_PATH,
    output_path=FILTERED_CAUSALBANK_GRAPH_PATH,
    min_count=DEFAULT_MIN_COUNT,
    min_causal_strength=DEFAULT_MIN_CAUSAL_STRENGTH,
    necessity_weight=DEFAULT_NECESSITY_WEIGHT,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges_written = 0
    nodes = set()

    with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
        for cause, effect, count, necessity_score, sufficiency_score in (
            iter_filtered_causalbank_edges(
                input_path=input_path,
                min_count=min_count,
                min_causal_strength=min_causal_strength,
                necessity_weight=necessity_weight,
            )
        ):
            output_file.write(
                f"{cause}->{effect}\t{count}\t{necessity_score}\t{sufficiency_score}\n"
            )
            edges_written += 1
            nodes.add(cause)
            nodes.add(effect)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "edges_written": edges_written,
        "nodes_written": len(nodes),
        "min_count": min_count,
        "min_causal_strength": min_causal_strength,
        "necessity_weight": necessity_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-filter the raw CausalBank/lexical CEG graph once."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_CAUSALBANK_GRAPH_PATH,
        help="Raw CausalBank graph path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FILTERED_CAUSALBANK_GRAPH_PATH,
        help="Filtered CausalBank graph path.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="Minimum CausalBank pair count.",
    )
    parser.add_argument(
        "--min-causal-strength",
        type=float,
        default=DEFAULT_MIN_CAUSAL_STRENGTH,
        help="Minimum combined CausalNet causal strength.",
    )
    parser.add_argument(
        "--necessity-weight",
        type=float,
        default=DEFAULT_NECESSITY_WEIGHT,
        help=(
            "Lambda in necessity^lambda * sufficiency^(1-lambda). "
            "The CausalNet paper reports 0.9/1.0 as best for explicit patterns."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = filter_causalbank_graph(
        input_path=args.input,
        output_path=args.output,
        min_count=args.min_count,
        min_causal_strength=args.min_causal_strength,
        necessity_weight=args.necessity_weight,
    )

    print(f"Input:  {stats['input_path']}")
    print(f"Output: {stats['output_path']}")
    print(f"Edges:  {stats['edges_written']:,}")
    print(f"Nodes:  {stats['nodes_written']:,}")
    print(
        "Filter: "
        f"count >= {stats['min_count']}, "
        "causal_strength = "
        f"necessity^{stats['necessity_weight']} * "
        f"sufficiency^{1.0 - stats['necessity_weight']:.2f}, "
        f"causal_strength >= {stats['min_causal_strength']}"
    )


if __name__ == "__main__":
    main()
