from collections import deque
from typing import Any

import networkx as nx
from sortedcontainers import SortedList

from core.embeddings import STEmbedder


def bfs_traverse(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    _embeder: STEmbedder,
    config=None
) -> tuple[list[Any], int]:
    if config is None:
        config = {}

    max_visits = config.get("bfs_max_visits", -1)

    # Standard BFS queue:
    # each entry stores the current node and the full path to it.
    queue = deque([(start_node, [start_node])])

    # Local visited set.
    # Faster than writing "visited" metadata into the NetworkX graph.
    visited = set()
    visited_count = 0

    while queue:
        current_node, path = queue.popleft()

        # If we reached the target, return the path and number of visited nodes.
        if current_node == end_node:
            return path, visited_count

        if current_node in visited:
            continue

        visited.add(current_node)
        visited_count += 1

        if max_visits != -1 and visited_count > max_visits:
            return [], visited_count

        # BFS itself is unweighted, but we still sort successors by edge support
        # so that stronger / better-supported edges are explored first.
        sorted_successors = SortedList(
            [
                (s, graph.get_edge_data(current_node, s)["support"])
                for s in graph.successors(current_node)
            ],
            key=lambda x: -x[1]
        )

        for successor, _ in sorted_successors:
            if successor not in visited:
                queue.append((successor, path + [successor]))

    # No path found.
    return [], visited_count