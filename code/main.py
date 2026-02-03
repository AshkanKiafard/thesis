import traverse_strategies as ts
from embeddings import STEmbeder, GloveEmbeder, DistanceMetric
from utils import load_graph, traverse_graph

if __name__ == "__main__":
    st_embeder = STEmbeder('sentence-transformers/all-mpnet-base-v2', DistanceMetric.COSINE)
    glove_embeder = GloveEmbeder('data/embeddings/glove.6B/glove.6B.300d.txt', DistanceMetric.COSINE)
    print("Embeders initialized.")

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")
    print("Causal graph loaded.")

    cause = "study"
    effect = "illness"

    print("Starting BFS traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.bfs_traverse)
    print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting A* traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.astar_traverse)
    print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting Dijkstra traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.dijkstra_traverse)
    print(f"Dijkstra path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting RL traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, glove_embeder, ts.rl_traverse)
    print(f"RL path found: {path}\nVisited nodes: {visited_nodes}")
