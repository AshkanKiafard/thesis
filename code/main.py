import traverse_strategies as ts
from embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from utils import load_graph, traverse_graph

if __name__ == "__main__":
    st_embeder = STEmbedder('sentence-transformers/all-mpnet-base-v2', DistanceMetric.COSINE)
    glove_embeder = GloveEmbeder('data/embeddings/glove.6B/glove.6B.300d.txt', DistanceMetric.COSINE)
    print("Embeders initialized.")

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl", remove_self_loops=False)

    cause = "coughing"
    effect = "heart attacks"

    print("Starting BFS traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, None, ts.bfs_traverse, None)
    print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting A* traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.astar_traverse, None)
    print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting Dijkstra traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.dijkstra_traverse, None)
    print(f"Dijkstra path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting RL traversal...")
    rl_config = {
        'rl_model_path': 'data/models/rl/msmarco_evaluation_state_dict.pt',
        'rl_beam_width': 5,
        'rl_max_path_len': 100,
    }
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, glove_embeder, ts.rl_traverse, rl_config)
    print(f"RL path found: {path}\nVisited nodes: {visited_nodes}")
