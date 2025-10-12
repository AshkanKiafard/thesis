from collections import deque
from typing import Any

import networkx as nx
from sortedcontainers import SortedList


def bfs_traverse(graph: nx.DiGraph, start_node: str, end_node: str) -> tuple[list[Any], int]:
    visited = set()
    queue = deque([(start_node, [start_node])])

    while queue:
        current_node, path = queue.popleft()

        if current_node == end_node:
            return path, len(visited)

        if current_node not in visited:
            visited.add(current_node)

            sorted_successors = SortedList(
                [(s, graph.get_edge_data(current_node, s)["support"]) for s in graph.successors(current_node)],
                key=lambda x: -x[1]
            )

            for successor, _ in sorted_successors[:10]:
                if successor not in visited:
                    queue.append((successor, path + [successor]))

    return [], len(visited)