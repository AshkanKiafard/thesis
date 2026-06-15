import heapq
from typing import Any, Dict

import networkx as nx

from core.constants import EMBEDDING_INDEX_MIN_SUCCESSORS
from core.embeddings import STEmbedder


def _reconstruct_path(parents, end_node):
    path = []
    current_node = end_node

    while current_node is not None:
        path.append(current_node)
        current_node = parents[current_node]

    path.reverse()
    return path


def _reconstruct_index_path(parents, end_index, indexed_graph):
    path = []
    current_index = end_index

    while current_index is not None:
        path.append(indexed_graph.node_text(current_index))
        current_index = parents[current_index]

    path.reverse()
    return path


def _embed_many(embeder, nodes, config):
    # See benchmarks/embed_many_threshold_benchmark.py: small successor batches
    # are still faster through single-row lookups, while embed_many starts
    # winning around 16 successors on the astar_gpu runtime.
    min_successors = config.get(
        "embedding_index_min_successors",
        EMBEDDING_INDEX_MIN_SUCCESSORS,
    )
    has_embedding_index = getattr(embeder, "has_embedding_index", None)
    can_use_index = (
        has_embedding_index()
        if callable(has_embedding_index)
        else True
    )

    embed_many = getattr(embeder, "embed_many", None)
    if len(nodes) >= min_successors and can_use_index and callable(embed_many):
        return embed_many(nodes)

    return [
        embeder.embed(node)
        for node in nodes
    ]


def _get_indexed_graph(config, embeder, start_node, end_node):
    indexed_graph = config.get("_indexed_graph")

    if indexed_graph is None:
        return None

    has_embedding_index = getattr(embeder, "has_embedding_index", None)
    if not callable(has_embedding_index) or not has_embedding_index():
        return None

    if not indexed_graph.has_node(start_node) or not indexed_graph.has_node(end_node):
        return None

    return indexed_graph


def _has_normalized_runtime_embeddings(embeder):
    has_normalized = getattr(embeder, "has_normalized_runtime_embeddings", None)
    return callable(has_normalized) and has_normalized()


def _get_distances(embeder, source_embed, target_embeds, assume_normalized):
    if assume_normalized:
        return embeder.get_distances(
            source_embed,
            target_embeds,
            assume_normalized=True,
        )

    return embeder.get_distances(source_embed, target_embeds)


def _dijkstra_traverse_indexed(
    indexed_graph,
    start_node: str,
    end_node: str,
    embeder: STEmbedder,
    max_visits: int,
) -> tuple[list[Any], int]:
    start_index = indexed_graph.node_index(start_node)
    end_index = indexed_graph.node_index(end_node)

    open_set = [(0, start_node, start_index)]
    visited = set()
    visited_count = 0
    distances = {start_index: 0}
    parents = {start_index: None}
    adjacency = indexed_graph.adjacency
    assume_normalized = _has_normalized_runtime_embeddings(embeder)

    while open_set:
        distance, current_node, current_index = heapq.heappop(open_set)

        if current_index == end_index:
            return _reconstruct_index_path(
                parents,
                end_index,
                indexed_graph,
            ), visited_count

        if current_index in visited:
            continue

        visited.add(current_index)
        visited_count += 1

        if max_visits != -1 and visited_count > max_visits:
            return [], visited_count

        current_node_embed = embeder.embed_index(current_index)
        successors = [
            successor
            for successor in adjacency[current_index]
            if successor not in visited
        ]

        if not successors:
            continue

        successor_embeds = embeder.embed_indices(successors)
        edge_costs = _get_distances(
            embeder,
            current_node_embed,
            successor_embeds,
            assume_normalized,
        )

        for successor, edge_cost in zip(successors, edge_costs):
            new_distance = distance + edge_cost

            if successor not in distances or new_distance < distances[successor]:
                distances[successor] = new_distance
                parents[successor] = current_index
                successor_node = indexed_graph.node_text(successor)
                heapq.heappush(open_set, (new_distance, successor_node, successor))

    return [], visited_count


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
    indexed_graph = _get_indexed_graph(config, embeder, start_node, end_node)

    if indexed_graph is not None:
        return _dijkstra_traverse_indexed(
            indexed_graph,
            start_node,
            end_node,
            embeder,
            max_visits,
        )

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
    adjacency = graph._succ
    assume_normalized = _has_normalized_runtime_embeddings(embeder)

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
            for successor in adjacency.get(current_node, ())
            if successor not in visited
        ]

        if not successors:
            continue

        successor_embeds = _embed_many(embeder, successors, config)
        edge_costs = _get_distances(
            embeder,
            current_node_embed,
            successor_embeds,
            assume_normalized,
        )

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
