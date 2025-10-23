from typing import Callable

import networkx as nx

import traverse_strategies as ts
from utils import load_graph


def traverse_graph(graph: nx.DiGraph, start_node: str, end_node: str, strategy_fn: Callable):
    if start_node not in graph or end_node not in graph:
        raise ValueError("Both start_node and end_node must exist in the graph.")

    return strategy_fn(graph, start_node, end_node)


if __name__ == "__main__":
    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")
    print("Causal graph loaded.")

    cause = "study"
    effect = "success"

    print("Starting RL traversal...")
    path, visited_nodes = traverse_graph(causal_graph, cause, effect, ts.rl_traverse)
    print(f"RL path found: {path}\nVisited nodes: {visited_nodes}")

    # path, visited_nodes = traverse_graph(causal_graph, cause, effect, ts.bfs_traverse)
    # print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")

    # path, visited_nodes = traverse_graph(causal_graph, cause, effect, ts.astar_traverse)
    # print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")
    #
    # path, visited_nodes = traverse_graph(causal_graph, cause, effect, ts.dijkstra_traverse)
    # print(f"Dijkstra path found: {path}\nVisited nodes: {visited_nodes}")
