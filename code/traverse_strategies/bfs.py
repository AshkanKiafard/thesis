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
    _config=None
) -> tuple[list[Any], int]:
    # Standard BFS queue:
    # each entry stores the current node and the full path to it.
    queue = deque([(start_node, [start_node])])

    # Reset visited flags before every traversal.
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0

    while queue:
        current_node, path = queue.popleft()

        # If we reached the target, return the path and number of visited nodes.
        if current_node == end_node:
            return path, visited_count

        if not graph.nodes[current_node]["visited"]:
            nx.set_node_attributes(graph, {current_node: {"visited": True}})
            visited_count += 1

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
                if not graph.nodes[successor]["visited"]:
                    queue.append((successor, path + [successor]))

    # No path found.
    return [], visited_count