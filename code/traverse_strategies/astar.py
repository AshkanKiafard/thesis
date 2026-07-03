import heapq
from typing import Any, Dict

import networkx as nx

from core.config import EMBEDDING_INDEX_MIN_SUCCESSORS
from core.embeddings import STEmbedder


def _reconstruct_path(path_links, path_index):
    path = []

    while path_index is not None:
        node, path_index = path_links[path_index]
        path.append(node)

    path.reverse()
    return path


def _reconstruct_index_path(path_links, path_index, indexed_graph):
    path = []

    while path_index is not None:
        node_index, path_index = path_links[path_index]
        path.append(indexed_graph.node_text(node_index))

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


def _get_distances_pair(
    embeder,
    source_embed1,
    source_embed2,
    target_embeds,
    assume_normalized,
):
    # STEmbedder keeps both rows on the accelerator until one host transfer.
    # Other embedders retain the previous two-call behavior through the fallback.
    get_distances_pair = getattr(embeder, "get_distances_pair", None)

    if callable(get_distances_pair):
        if assume_normalized:
            return get_distances_pair(
                source_embed1,
                source_embed2,
                target_embeds,
                assume_normalized=True,
            )

        return get_distances_pair(
            source_embed1,
            source_embed2,
            target_embeds,
        )

    return (
        _get_distances(
            embeder,
            source_embed1,
            target_embeds,
            assume_normalized,
        ),
        _get_distances(
            embeder,
            source_embed2,
            target_embeds,
            assume_normalized,
        ),
    )


def _astar_traverse_indexed(
    indexed_graph,
    start_node: str,
    end_node: str,
    embeder: STEmbedder,
    max_visits: int,
) -> tuple[list[Any], int]:
    start_index = indexed_graph.node_index(start_node)
    end_index = indexed_graph.node_index(end_node)

    path_links = [(start_index, None)]
    open_set = [(0, 0, start_node, start_index, 0)]
    best_g = {start_index: 0.0}
    visited = set()
    visited_count = 0

    end_node_embed = embeder.embed_index(end_index)
    adjacency = indexed_graph.adjacency
    assume_normalized = _has_normalized_runtime_embeddings(embeder)

    while open_set:
        f_score, g_score, current_node, current_index, path_index = (
            heapq.heappop(open_set)
        )

        if current_index == end_index:
            return _reconstruct_index_path(
                path_links,
                path_index,
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
        edge_costs, heuristic_costs = _get_distances_pair(
            embeder,
            current_node_embed,
            end_node_embed,
            successor_embeds,
            assume_normalized,
        )

        for successor, edge_cost, heuristic in zip(
            successors,
            edge_costs,
            heuristic_costs,
        ):
            tentative_g = g_score + edge_cost

            if tentative_g >= best_g.get(successor, float("inf")):
                continue

            best_g[successor] = tentative_g
            tentative_f = tentative_g + heuristic

            path_links.append((successor, path_index))
            successor_path_index = len(path_links) - 1
            successor_node = indexed_graph.node_text(successor)

            heapq.heappush(
                open_set,
                (
                    tentative_f,
                    tentative_g,
                    successor_node,
                    successor,
                    successor_path_index,
                ),
            )

    return [], visited_count


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
    indexed_graph = _get_indexed_graph(config, embeder, start_node, end_node)

    if indexed_graph is not None:
        return _astar_traverse_indexed(
            indexed_graph,
            start_node,
            end_node,
            embeder,
            max_visits,
        )

    # Priority queue entries are:
    # (f_score, g_score, current_node, path_link_index)
    #
    # f = g + h
    # g = current path cost from start to current node
    # h = heuristic estimate from current node to end node
    path_links = [(start_node, None)]
    open_set = [(0, 0, start_node, 0)]
    best_g = {start_node: 0.0}

    # Local closed set.
    # Faster than writing "visited" metadata into the NetworkX graph.
    visited = set()
    visited_count = 0

    # Embed the target node once so we do not recompute it for every expansion.
    end_node_embed = embeder.embed(end_node)
    adjacency = graph._succ
    assume_normalized = _has_normalized_runtime_embeddings(embeder)

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
            for successor in adjacency.get(current_node, ())
            if successor not in visited
        ]

        if not successors:
            continue

        successor_embeds = _embed_many(embeder, successors, config)
        edge_costs, heuristic_costs = _get_distances_pair(
            embeder,
            current_node_embed,
            end_node_embed,
            successor_embeds,
            assume_normalized,
        )

        for successor, edge_cost, heuristic in zip(
            successors,
            edge_costs,
            heuristic_costs,
        ):
            # Edge cost is the embedding distance between current node and successor.
            tentative_g = g_score + edge_cost

            if tentative_g >= best_g.get(successor, float("inf")):
                continue

            best_g[successor] = tentative_g

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
