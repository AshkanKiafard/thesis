import heapq
from typing import Any

import networkx as nx
from sortedcontainers import SortedList

from utils import get_embedding_distance


def astar_traverse(graph: nx.DiGraph, start_node: str, end_node: str) -> tuple[list[Any], int]:
    # Priority queue: (f_score, g_score, current_node, path)
    open_set = [(0, 0, start_node, [start_node])]
    visited = set()

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

        for neighbor, _ in sorted_successors[:10]:
            if neighbor in visited:
                continue

            edge_cost = get_embedding_distance(current_node, neighbor)
            tentative_g = g_score + edge_cost

            heuristic = get_embedding_distance(neighbor, end_node)
            tentative_f = tentative_g + heuristic

            heapq.heappush(open_set, (tentative_f, tentative_g, neighbor, path + [neighbor]))

    # No path found
    return [], len(visited)