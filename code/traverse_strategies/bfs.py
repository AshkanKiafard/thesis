from collections import deque
from typing import Any

import networkx as nx
from sortedcontainers import SortedList

from embeddings import Embeder


def bfs_traverse(graph: nx.DiGraph, start_node: str, end_node: str, _embeder: Embeder) -> tuple[list[Any], int]:
    queue = deque([(start_node, [start_node])])
    nx.set_node_attributes(graph, False, "visited")
    visited_count = 0

    while queue:
        current_node, path = queue.popleft()

        if current_node == end_node:
            return path, visited_count

        if not graph.nodes[current_node]["visited"]:
            nx.set_node_attributes(graph, {current_node: {"visited": True}})
            visited_count += 1

            sorted_successors = SortedList(
                [(s, graph.get_edge_data(current_node, s)["support"]) for s in graph.successors(current_node)],
                key=lambda x: -x[1]
            )

            for successor, _ in sorted_successors:
                if not graph.nodes[successor]["visited"]:
                    queue.append((successor, path + [successor]))

    return [], visited_count