import time
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


def build_indexed_graph(graph, text_to_idx, progress_every=500_000) -> IndexedGraph:
    start_time = time.time()
    total_nodes = len(text_to_idx)
    print(
        "Building indexed graph adjacency: "
        f"{total_nodes:,} nodes.",
        flush=True,
    )

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

    for processed, (node, index) in enumerate(text_to_idx.items(), start=1):
        adjacency[index] = tuple(
            text_to_idx[successor]
            for successor in graph_adjacency.get(node, ())
        )

        if progress_every and processed % progress_every == 0:
            print(
                "Indexed graph adjacency progress: "
                f"{processed:,}/{total_nodes:,} nodes "
                f"({processed / total_nodes:.1%}), "
                f"{time.time() - start_time:.1f}s elapsed.",
                flush=True,
            )

    print(
        "Indexed graph adjacency ready: "
        f"{total_nodes:,} nodes, "
        f"{graph.number_of_edges():,} edges, "
        f"{time.time() - start_time:.1f}s elapsed.",
        flush=True,
    )

    return IndexedGraph(
        node_to_idx=text_to_idx,
        idx_to_node=idx_to_node,
        adjacency=adjacency,
    )
