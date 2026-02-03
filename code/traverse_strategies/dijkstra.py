import heapq
from typing import Any

import networkx as nx

from embeddings import STEmbeder


def dijkstra_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: STEmbeder) -> tuple[list[Any], int]:
    # Priority queue: (distance, current_node, path)
    open_set = [(0, start_node, [start_node])]
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0
    distances = {start_node: 0}

    while open_set:
        distance, current_node, path = heapq.heappop(open_set)

        if current_node == end_node:
            return path, visited_count

        if graph.nodes[current_node]["visited"]:
            continue
        nx.set_node_attributes(graph, {current_node: {"visited": True}})
        visited_count += 1

        current_node_embed = embeder.embed(current_node)

        for successor in graph.successors(current_node):
            if graph.nodes[successor]["visited"]:
                continue

            successor_embed = embeder.embed(successor)

            edge_cost = embeder.get_distance(current_node_embed, successor_embed)
            new_distance = distance + edge_cost

            if successor not in distances or new_distance < distances[successor]:
                distances[successor] = new_distance
                heapq.heappush(open_set, (new_distance, successor, path + [successor]))

    return [], visited_count
