import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import traverse_strategies as ts
from embeddings import Embeder
from utils import get_concept, load_graph, traverse_graph

# Load test data
with open("data/datasets/msmarco_test.json") as f:
    test_data = json.load(f)
print("Test data loaded.")

model_path = "data/models/sentence-transformers/multi-qa-mpnet-base-cos-v1_fine-tuned"
distance_metric = 'cosine'
embeder = Embeder(model_path=model_path, distance_metric=distance_metric)
print("Embeder initialized.")

causal_graph = load_graph("data/graphs/causenet-precision.jsonl")
print("Causal graph loaded.")

strategies = {
    "BFS": ts.bfs_traverse,
    "A*": ts.astar_traverse,
    "Dijkstra": ts.dijkstra_traverse
}

# TODO save time needed!
results = {
    name: {"y_true": [], "y_pred": [], "nodes_visited": [], "path_lengths": []}
    for name in strategies.keys()
}

# Evaluation loop
for i, item in enumerate(test_data):
    print(f"Evaluating item {i + 1}/{len(test_data)}")

    cause = get_concept(item, 0)
    effect = get_concept(item, 1)
    true_label = item['answer:Extracted'][0] == 'Yes'

    for name, strategy in strategies.items():
        path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, strategy)
        pred_label = bool(path)

        results[name]["y_true"].append(true_label)
        results[name]["y_pred"].append(pred_label)
        results[name]["nodes_visited"].append(visited_nodes)
        results[name]["path_lengths"].append(len(path))

# Compute metrics
summary = {}

for name in strategies.keys():
    y_true = np.array(results[name]["y_true"])
    y_pred = np.array(results[name]["y_pred"])
    nodes_visited = np.array(results[name]["nodes_visited"], dtype=int)
    path_lengths = np.array(results[name]["path_lengths"], dtype=int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    avg_nodes = nodes_visited.mean()
    avg_path_len = path_lengths[path_lengths > 0].mean()

    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "nodes": float(avg_nodes),
        "path_length": float(avg_path_len)
    }

    summary[name] = metrics

    print(f"--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print()

