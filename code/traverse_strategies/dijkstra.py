import heapq
from typing import Any, Dict

import networkx as nx

from core.embeddings import STEmbedder


def dijkstra_traverse(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    embeder: STEmbedder,
    config: Dict[str, Any] = None
) -> tuple[list[Any], int]:
    # Allow optional runtime config such as a visit limit.
    if config is None:
        config = {}

    max_visits = config.get('dijkstra_max_visits', -1)

    # Priority queue entries are:
    # (current_distance, current_node, path_so_far)
    open_set = [(0, start_node, [start_node])]

    # Reset visited flags before each traversal.
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0

    # Best known distance from the start node to each node.
    distances = {start_node: 0}

    while open_set:
        distance, current_node, path = heapq.heappop(open_set)

        # Target reached -> return path and number of expanded nodes.
        if current_node == end_node:
            return path, visited_count

        # Skip nodes that were already finalized.
        if graph.nodes[current_node]["visited"]:
            continue

        nx.set_node_attributes(graph, {current_node: {"visited": True}})
        visited_count += 1

        # Optional limit used during evaluation to prevent extremely expensive runs.
        if max_visits != -1 and visited_count > max_visits:
            return [], visited_count

        current_node_embed = embeder.embed(current_node)

        for successor in graph.successors(current_node):
            if graph.nodes[successor]["visited"]:
                continue

            successor_embed = embeder.embed(successor)

            # Edge weights are defined by embedding distance between connected nodes.
            edge_cost = embeder.get_distance(current_node_embed, successor_embed)
            new_distance = distance + edge_cost

            # Relax edge if this route is better than the best known one.
            if successor not in distances or new_distance < distances[successor]:
                distances[successor] = new_distance
                heapq.heappush(open_set, (new_distance, successor, path + [successor]))

    # No path found.
    return [], visited_count