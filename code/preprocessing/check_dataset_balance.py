import json
from pathlib import Path

try:
    from preprocessing.filter_causalbank_graph import FILTERED_CAUSALBANK_GRAPH_PATH
except ImportError:
    from filter_causalbank_graph import FILTERED_CAUSALBANK_GRAPH_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = REPO_ROOT / "data" / "datasets" / "filtered"
CAUSENET_GRAPH_PATH = REPO_ROOT / "data" / "graphs" / "causenet-precision.jsonl"
CAUSALBANK_GRAPH_PATH = FILTERED_CAUSALBANK_GRAPH_PATH

GRAPH_CONFIGS = {
    "causenet": {
        "label": "CauseNet",
        "path": CAUSENET_GRAPH_PATH,
    },
    "causalbank": {
        "label": "CausalBank",
        "path": CAUSALBANK_GRAPH_PATH,
    },
}


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_causenet_concept(value):
    return value.replace("_", " ").strip()


def load_causenet_nodes(file_path):
    nodes = set()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            relation = item["causal_relation"]
            cause = normalize_causenet_concept(relation["cause"]["concept"])
            effect = normalize_causenet_concept(relation["effect"]["concept"])

            if cause == effect:
                continue

            nodes.add(cause)
            nodes.add(effect)

    return nodes


def load_causalbank_nodes(file_path):
    nodes = set()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if not parts or "->" not in parts[0]:
                continue

            cause, effect = parts[0].split("->", 1)

            if not cause or not effect or cause == effect:
                continue

            nodes.add(cause)
            nodes.add(effect)

    return nodes


def load_graph_nodes(graph_name):
    if graph_name == "causenet":
        return load_causenet_nodes(GRAPH_CONFIGS[graph_name]["path"])

    if graph_name == "causalbank":
        return load_causalbank_nodes(GRAPH_CONFIGS[graph_name]["path"])

    raise ValueError(f"Unknown graph name: {graph_name}")


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


def pretty_dataset_name(file_path):
    name = file_path.stem.lower().replace("_filtered", "")

    for suffix in ("_train_valid", "_valid", "_validation", "_train", "_test"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break

    return name


def count_split(data, graph_nodes_by_name):
    total_pos = 0
    total_neg = 0
    graph_counts = {
        graph_name: {"pos": 0, "neg": 0}
        for graph_name in graph_nodes_by_name
    }

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
        for graph_name, graph_nodes in graph_nodes_by_name.items():
            if cause in graph_nodes and effect in graph_nodes:
                if label is True:
                    graph_counts[graph_name]["pos"] += 1
                else:
                    graph_counts[graph_name]["neg"] += 1

    return total_pos, total_neg, graph_counts


def main():
    dataset_files = sorted(DATASET_DIR.glob("*.json"))
    selected_graphs = ["causenet", "causalbank"]

    graph_nodes_by_name = {}

    for graph_name in selected_graphs:
        graph_config = GRAPH_CONFIGS[graph_name]
        graph_path = graph_config["path"]

        if not graph_path.exists():
            message = f"Missing {graph_config['label']} graph: {graph_path}"
            if graph_name == "causalbank":
                message += (
                    "\nRun: python -m preprocessing.filter_causalbank_graph"
                )
            raise FileNotFoundError(message)

        print(f"Loading {graph_config['label']} nodes from {graph_path}...")
        graph_nodes = load_graph_nodes(graph_name)
        graph_nodes_by_name[graph_name] = graph_nodes
        print(f"Loaded {len(graph_nodes):,} {graph_config['label']} nodes.")

    rows = []

    for dataset_file in dataset_files:
        data = load_dataset(dataset_file)
        total_pos, total_neg, graph_counts = count_split(data, graph_nodes_by_name)

        rows.append(
            (
                pretty_dataset_name(dataset_file),
                pretty_split_name(dataset_file),
                total_pos,
                total_neg,
                graph_counts,
            )
        )

    print("\nDataset Statistics")
    header = (
        f"{'Dataset':<15}"
        f"{'Split':<15}"
        f"{'Total Pos':>12}"
        f"{'Total Neg':>12}"
    )

    for graph_name in selected_graphs:
        graph_label = GRAPH_CONFIGS[graph_name]["label"]
        header += f"{graph_label + ' Pos':>16}{graph_label + ' Neg':>16}"

    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for dataset, split, total_pos, total_neg, graph_counts in rows:
        row = (
            f"{dataset:<15}"
            f"{split:<15}"
            f"{total_pos:>12}"
            f"{total_neg:>12}"
        )

        for graph_name in selected_graphs:
            counts = graph_counts[graph_name]
            row += f"{counts['pos']:>16}{counts['neg']:>16}"

        print(row)

    print("=" * len(header))


if __name__ == "__main__":
    main()
