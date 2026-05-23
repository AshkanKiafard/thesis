import heapq
from typing import Any, Dict

import networkx as nx

from core.embeddings import STEmbedder


def _reconstruct_path(path_links, path_index):
    path = []

    while path_index is not None:
        node, path_index = path_links[path_index]
        path.append(node)

    path.reverse()
    return path


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

    max_visits = config.get("astar_max_visits", -1)

    # Priority queue entries are:
    # (f_score, g_score, current_node, path_link_index)
    #
    # f = g + h
    # g = current path cost from start to current node
    # h = heuristic estimate from current node to end node
    path_links = [(start_node, None)]
    open_set = [(0, 0, start_node, 0)]

    # Local closed set.
    # Faster than writing "visited" metadata into the NetworkX graph.
    visited = set()
    visited_count = 0

    # Embed the target node once so we do not recompute it for every expansion.
    end_node_embed = embeder.embed(end_node)

    while open_set:
        f_score, g_score, current_node, path_index = heapq.heappop(open_set)

        # Goal reached -> return the path and the number of expanded nodes.
        if current_node == end_node:
            return _reconstruct_path(path_links, path_index), visited_count

        # Skip nodes that were already finalized.
        if current_node in visited:
            continue

        visited.add(current_node)
        visited_count += 1

        # Optional safety cap for evaluation / runtime control.
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
        heuristic_costs = embeder.get_distances(end_node_embed, successor_embeds)

        for successor, edge_cost, heuristic in zip(
            successors,
            edge_costs,
            heuristic_costs,
        ):
            # Edge cost is the embedding distance between current node and successor.
            tentative_g = g_score + edge_cost

            # Heuristic is the embedding distance from successor to goal.
            tentative_f = tentative_g + heuristic

            path_links.append((successor, path_index))
            successor_path_index = len(path_links) - 1

            heapq.heappush(
                open_set,
                (tentative_f, tentative_g, successor, successor_path_index)
            )

    # No path found.
    return [], visited_count
