import traverse_strategies as ts
from core.embeddings import STEmbedder, GloveEmbeder, DistanceMetric
from core.utils import load_causal_graph, load_rl_graph, traverse_graph
from evaluation.visited_nodes_analysis import GRAPH_PATH

if __name__ == "__main__":
    GRAPH_PATH = "data/graphs/causenet-precision.jsonl"

    st_embeder = STEmbedder('sentence-transformers/all-mpnet-base-v2', DistanceMetric.COSINE)
    glove_embeder = GloveEmbeder('data/embeddings/glove.6B/glove.6B.300d.txt', DistanceMetric.COSINE)
    print("Embeders initialized.")

    print("Loading graphs...")
    causal_graph = load_causal_graph(GRAPH_PATH, use_inverse=False)
    rl_graph = load_rl_graph(GRAPH_PATH, use_inverse=False)

    cause = "wine"
    effect = "migraines"

    print("Causal graph:")
    print(list(causal_graph.successors("headaches")))
    print("RL graph:")
    print(rl_graph.successors("headaches"))

    print("Starting BFS traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, None, ts.bfs_traverse, None)
    print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")

    # print("Starting A* traversal...")
    # path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.astar_traverse, None)
    # print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")
    #
    # print("Starting Dijkstra traversal...")
    # path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.dijkstra_traverse, None)
    # print(f"Dijkstra path found: {path}\nVisited nodes: {visited_nodes}")

    print("Starting RL traversal...")
    rl_config = {
        "rl_model_path": "data/models/rl/msmarco_no_inverse_state_dict.pt",
        "rl_beam_width": 50,
        "rl_max_path_len": 2,
        "rl_max_actions": 5000,
        "rl_max_visits": -1,
    }
    path, visited_nodes = traverse_graph(rl_graph, cause, effect, glove_embeder, ts.rl_traverse, rl_config)
    print(f"RL path found: {path}\nVisited nodes: {visited_nodes}")
