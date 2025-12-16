import traverse_strategies as ts
from embeddings import Embeder, DistanceMetric
from utils import load_graph, traverse_graph

if __name__ == "__main__":
    embeder = Embeder('sentence-transformers/all-mpnet-base-v2', DistanceMetric.COSINE)
    print("Embeder initialized.")

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")
    print("Causal graph loaded.")

    cause = "study"
    effect = "illness"

    print("Starting BFS traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.bfs_traverse)
    print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting A* traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.astar_traverse)
    print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting Dijkstra traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.dijkstra_traverse)
    print(f"Dijkstra path found: {path}\nVisited nodes: {visited_nodes}")
