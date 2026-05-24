from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class IndexedGraph:
    node_to_idx: Dict[str, int]
    idx_to_node: List[str]
    adjacency: List[Tuple[int, ...]]

    def has_node(self, node: str) -> bool:
        return node in self.node_to_idx

    def node_index(self, node: str) -> int:
        return self.node_to_idx[node]

    def node_text(self, index: int) -> str:
        return self.idx_to_node[index]

    def successors(self, index: int) -> Tuple[int, ...]:
        return self.adjacency[index]


def build_indexed_graph(graph, text_to_idx) -> IndexedGraph:
    idx_to_node = [None] * len(text_to_idx)

    for node, index in text_to_idx.items():
        idx_to_node[index] = node

    missing_nodes = [
        node
        for node in graph.nodes
        if node not in text_to_idx
    ]

    if missing_nodes:
        preview = ", ".join(repr(node) for node in missing_nodes[:5])
        raise ValueError(
            "Cannot build indexed graph because graph nodes are missing from "
            f"the embedding index. First missing nodes: {preview}"
        )

    adjacency = [[] for _ in idx_to_node]
    graph_adjacency = graph._succ

    for node, index in text_to_idx.items():
        adjacency[index] = tuple(
            text_to_idx[successor]
            for successor in graph_adjacency.get(node, ())
        )

    return IndexedGraph(
        node_to_idx=text_to_idx,
        idx_to_node=idx_to_node,
        adjacency=adjacency,
    )
