import json
from pathlib import Path

from core.utils import load_causal_graph

DATASET_DIR = Path("../data/datasets/filtered")
GRAPH_PATH = Path("../data/graphs/causenet-precision.jsonl")


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_split_name(file_path):
    name = file_path.stem.lower()

    if "train_valid" in name:
        return "Train+Valid"
    if "validation" in name or "valid" in name:
        return "Validation"
    if "train" in name:
        return "Train"
    if "test" in name:
        return "Test"

    return file_path.stem


def count_split(data, graph):
    total_pos = 0
    total_neg = 0

    causenet_pos = 0
    causenet_neg = 0

    for item in data:
        label = item.get("answer")

        if label is True:
            total_pos += 1
        elif label is False:
            total_neg += 1
        else:
            print(f"Warning: invalid label in {item.get('id', 'unknown')}")
            continue

        cause = item["cause"]
        effect = item["effect"]

        # Same logic as evaluation: only count examples where both nodes exist.
        if cause in graph.nodes and effect in graph.nodes:
            if label is True:
                causenet_pos += 1
            else:
                causenet_neg += 1

    return total_pos, total_neg, causenet_pos, causenet_neg


def main():
    graph = load_causal_graph(GRAPH_PATH)

    dataset_files = sorted(DATASET_DIR.glob("*.json"))

    rows = []

    for dataset_file in dataset_files:
        data = load_dataset(dataset_file)
        stats = count_split(data, graph)

        rows.append((pretty_split_name(dataset_file), *stats))

    print("\nDataset Statistics")
    print("=" * 65)
    print(
        f"{'Split':<15}"
        f"{'Total Pos':>12}"
        f"{'Total Neg':>12}"
        f"{'CN Pos':>12}"
        f"{'CN Neg':>12}"
    )
    print("=" * 65)

    for split, total_pos, total_neg, cn_pos, cn_neg in rows:
        print(
            f"{split:<15}"
            f"{total_pos:>12}"
            f"{total_neg:>12}"
            f"{cn_pos:>12}"
            f"{cn_neg:>12}"
        )

    print("=" * 65)


if __name__ == "__main__":
    main()
