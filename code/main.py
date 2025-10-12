from typing import Callable

import networkx as nx

import traverse_strategies as ts
from utils import load_graph


def traverse_graph(graph: nx.DiGraph, start_node: str, end_node: str, strategy_fn: Callable):
    if start_node not in graph or end_node not in graph:
        raise ValueError("Both start_node and end_node must exist in the graph.")

    return strategy_fn(graph, start_node, end_node)


if __name__ == "__main__":
    causal_graph = load_graph("data/causenet-precision.jsonl")
    print(f"Graph loaded with {causal_graph.number_of_nodes()} nodes and {causal_graph.number_of_edges()} edges.")
    path, visited_nodes = traverse_graph(causal_graph, "programming", "depression", ts.bfs_traverse)
    print(f"BFS path found: {path}\nVisited nodes: {visited_nodes}")
    path, visited_nodes = traverse_graph(causal_graph, "programming", "depression", ts.astar_traverse)
    print(f"A* path found: {path}\nVisited nodes: {visited_nodes}")

