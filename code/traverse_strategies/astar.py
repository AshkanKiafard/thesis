import heapq
from typing import Any

import networkx as nx

from embeddings import Embeder


def astar_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: Embeder) -> tuple[list[Any], int]:
    # Priority queue: (f_score, g_score, current_node, path)
    open_set = [(0, 0, start_node, [start_node])]
    visited = set()

    end_node_embed = embeder.embed(end_node)

    while open_set:
        f_score, g_score, current_node, path = heapq.heappop(open_set)

        if current_node == end_node:
            return path, len(visited)

        if current_node in visited:
            continue
        visited.add(current_node)

        current_node_embed = embeder.embed(current_node)

        for successor in graph.successors(current_node):
            if successor in visited:
                continue

            successor_embed = embeder.embed(successor)

            edge_cost = embeder.get_distance(current_node_embed, successor_embed)
            tentative_g = g_score + edge_cost

            heuristic = embeder.get_distance(successor_embed, end_node_embed)
            tentative_f = tentative_g + heuristic

            heapq.heappush(open_set, (tentative_f, tentative_g, successor, path + [successor]))

    return [], len(visited)
