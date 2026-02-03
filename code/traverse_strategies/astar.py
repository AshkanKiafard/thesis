import heapq
from typing import Any

import networkx as nx

from embeddings import STEmbeder


def astar_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: STEmbeder) -> tuple[list[Any], int]:
    # Priority queue: (f_score, g_score, current_node, path)
    open_set = [(0, 0, start_node, [start_node])]
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0

    end_node_embed = embeder.embed(end_node)

    while open_set:
        f_score, g_score, current_node, path = heapq.heappop(open_set)

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
            tentative_g = g_score + edge_cost

            heuristic = embeder.get_distance(successor_embed, end_node_embed)
            tentative_f = tentative_g + heuristic

            heapq.heappush(open_set, (tentative_f, tentative_g, successor, path + [successor]))

    return [], visited_count
