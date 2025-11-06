import heapq
from typing import Any

import networkx as nx

from embeddings import Embeder


def dijkstra_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: Embeder) -> tuple[list[Any], int]:
    # Priority queue: (distance, current_node, path)
    open_set = [(0, start_node, [start_node])]
    visited = set()
    distances = {start_node: 0}

    while open_set:
        distance, current_node, path = heapq.heappop(open_set)

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
            new_distance = distance + edge_cost

            if successor not in distances or new_distance < distances[successor]:
                distances[successor] = new_distance
                heapq.heappush(open_set, (new_distance, successor, path + [successor]))

    return [], len(visited)
