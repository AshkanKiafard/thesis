from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx


def _reconstruct_path(parents, end_node):
    path = []
    current_node = end_node

    while current_node is not None:
        path.append(current_node)
        current_node = parents[current_node]

    path.reverse()
    return path


def bfs_traverse(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    _embeder: Any,
    config=None
) -> tuple[list[Any] | bool, int]:
    if config is None:
        config = {}

    max_visits = config.get("bfs_max_visits", -1)
    reachability_only = config.get("reachability_only", False)

    if reachability_only and start_node == end_node:
        return True, 0

    # Standard BFS queue with parent pointers.
    # Mark nodes when discovered so duplicate paths do not pile up in the queue.
    queue = deque([start_node])
    parents = None if reachability_only else {start_node: None}
    visited = {start_node}
    visited_count = 0

    while queue:
        current_node = queue.popleft()

        # If we reached the target, return the path and number of visited nodes.
        if not reachability_only and current_node == end_node:
            return _reconstruct_path(parents, end_node), visited_count

        visited_count += 1

        if max_visits != -1 and visited_count > max_visits:
            return (False if reachability_only else []), visited_count

        for successor in graph.successors(current_node):
            if successor not in visited:
                visited.add(successor)

                if reachability_only and successor == end_node:
                    return True, visited_count

                if parents is not None:
                    parents[successor] = current_node
                queue.append(successor)

    # No path found.
    return (False if reachability_only else []), visited_count
