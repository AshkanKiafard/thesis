import heapq
from typing import Any, Dict

import networkx as nx

from core.embeddings import STEmbedder


def _reconstruct_path(parents, end_node):
    path = []
    current_node = end_node

    while current_node is not None:
        path.append(current_node)
        current_node = parents[current_node]

    path.reverse()
    return path


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

    max_visits = config.get("dijkstra_max_visits", -1)

    # Priority queue entries are:
    # (current_distance, current_node)
    open_set = [(0, start_node)]

    # Local closed set.
    # Faster than writing "visited" metadata into the NetworkX graph.
    visited = set()
    visited_count = 0

    # Best known distance from the start node to each node.
    distances = {start_node: 0}
    parents = {start_node: None}

    while open_set:
        distance, current_node = heapq.heappop(open_set)

        # Target reached -> return path and number of expanded nodes.
        if current_node == end_node:
            return _reconstruct_path(parents, end_node), visited_count

        # Skip nodes that were already finalized.
        if current_node in visited:
            continue

        visited.add(current_node)
        visited_count += 1

        # Optional limit used during evaluation to prevent extremely expensive runs.
        if max_visits != -1 and visited_count > max_visits:
            return [], visited_count

        current_node_embed = embeder.embed(current_node)
        successors = [
            successor
            for successor in graph.successors(current_node)
            if successor not in visited
        ]
        successor_embeds = [embeder.embed(successor) for successor in successors]
        edge_costs = embeder.get_distances(current_node_embed, successor_embeds)

        for successor, edge_cost in zip(successors, edge_costs):
            # Edge weights are defined by embedding distance between connected nodes.
            new_distance = distance + edge_cost

            # Relax edge if this route is better than the best known one.
            if successor not in distances or new_distance < distances[successor]:
                distances[successor] = new_distance
                parents[successor] = current_node
                heapq.heappush(open_set, (new_distance, successor))

    # No path found.
    return [], visited_count
