import heapq
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np
import ollama
from sortedcontainers import SortedList


@lru_cache(maxsize=None)
def embed_text(text):
    return np.array(ollama.embed(model="nomic-embed-text:latest", input=text.replace("_", "")).embeddings)


def get_embedding_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


def astar_traverse(graph: nx.DiGraph, start_node: str, end_node: str) -> tuple[list[Any], int]:
    # Priority queue: (f_score, g_score, current_node, path)
    open_set = [(0, 0, start_node, [start_node])]
    visited = set()

    start_node_embed = embed_text(start_node)
    end_node_embed = embed_text(end_node)

    while open_set:
        f_score, g_score, current_node, path = heapq.heappop(open_set)

        if current_node == end_node:
            return path, len(visited)

        if current_node in visited:
            continue
        visited.add(current_node)

        sorted_successors = SortedList(
            [(s, graph.get_edge_data(current_node, s)["support"]) for s in graph.successors(current_node)],
            key=lambda x: -x[1]
        )

        for successor, _ in sorted_successors:
            if successor in visited:
                continue

            current_node_embed = embed_text(current_node)
            successor_embed = embed_text(successor)

            edge_cost = get_embedding_distance(current_node_embed, successor_embed)
            tentative_g = g_score + edge_cost

            heuristic = get_embedding_distance(successor_embed, end_node_embed)
            tentative_f = tentative_g + heuristic

            heapq.heappush(open_set, (tentative_f, tentative_g, successor, path + [successor]))

    return [], len(visited)
