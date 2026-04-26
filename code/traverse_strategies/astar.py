import heapq
from typing import Any, Dict

import networkx as nx

from core.embeddings import STEmbedder


def astar_traverse(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    embeder: STEmbedder,
    config: Dict[str, Any] = None
) -> tuple[list[Any], int]:
    # Allow optional runtime config (for example a max visit limit).
    if config is None:
        config = {}

    max_visits = config.get('astar_max_visits', -1)

    # Priority queue entries are:
    # (f_score, g_score, current_node, path_so_far)
    #
    # f = g + h
    # g = current path cost from start to current node
    # h = heuristic estimate from current node to end node
    open_set = [(0, 0, start_node, [start_node])]

    # Reset visited flags before each traversal.
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0

    # Embed the target node once so we do not recompute it for every expansion.
    end_node_embed = embeder.embed(end_node)

    while open_set:
        f_score, g_score, current_node, path = heapq.heappop(open_set)

        # Goal reached -> return the path and the number of expanded nodes.
        if current_node == end_node:
            return path, visited_count

        # Skip nodes that were already finalized.
        if graph.nodes[current_node]["visited"]:
            continue

        nx.set_node_attributes(graph, {current_node: {"visited": True}})
        visited_count += 1

        # Optional safety cap for evaluation / runtime control.
        if max_visits != -1 and visited_count > max_visits:
            return [], visited_count

        current_node_embed = embeder.embed(current_node)

        for successor in graph.successors(current_node):
            if graph.nodes[successor]["visited"]:
                continue

            successor_embed = embeder.embed(successor)

            # Edge cost is the embedding distance between current node and successor.
            edge_cost = embeder.get_distance(current_node_embed, successor_embed)
            tentative_g = g_score + edge_cost

            # Heuristic is the embedding distance from successor to goal.
            heuristic = embeder.get_distance(successor_embed, end_node_embed)
            tentative_f = tentative_g + heuristic

            heapq.heappush(
                open_set,
                (tentative_f, tentative_g, successor, path + [successor])
            )

    # No path found.
    return [], visited_count