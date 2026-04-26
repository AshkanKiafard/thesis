import json
from typing import Callable

import networkx as nx

from embeddings import STEmbedder


def load_graph(file_path, remove_self_loops=True):
    # Loads the causal graph from a JSONL file (one JSON object per line).
    # Each entry represents a causal relation (cause -> effect).
    #
    # The graph is constructed as a directed graph where:
    # - nodes = concepts (strings)
    # - edges = causal relations
    # - edge attributes store metadata like support and example sentence

    with open(file_path) as f:
        return nx.DiGraph([
            (
                c,  # cause node
                e,  # effect node
                {
                    # Number of supporting sources for this causal relation
                    "support": d.get("support", 0),

                    # Example sentence (if available) showing the relation
                    "sentence": d.get("sources", [{}])[0].get("payload", {}).get("sentence", "")
                }
            )
            # Read file line-by-line and parse JSON
            for d in map(json.loads, f)

            # Extract cause and effect concepts and normalize them
            # (replace "_" with spaces to get readable text)
            if (
                       (c := d["causal_relation"]["cause"]["concept"].replace('_', ' ')) !=
                       (e := d["causal_relation"]["effect"]["concept"].replace('_', ' '))
               )
               # Optionally remove self-loops (cause == effect)
               or not remove_self_loops
        ])


def traverse_graph(
        graph: nx.DiGraph,
        start_node: str,
        end_node: str,
        embeder: STEmbedder,
        strategy_fn: Callable,
        config: dict = None
):
    # Generic wrapper for running a traversal/search strategy.
    #
    # strategy_fn can be:
    # - BFS
    # - A*
    # - Dijkstra
    # - RL-based search
    #
    # The function just validates inputs and delegates to the strategy.

    # If either node is not in the graph, no path can exist.
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return [], 0

    # Strategy returns:
    # - path (list of nodes)
    # - number of visited nodes (used as cost/efficiency metric)
    return strategy_fn(graph, start_node, end_node, embeder, config)


def get_concept(question: dict, concept_type: int) -> str:
    # Extracts a concept (cause or effect) from a dataset example.
    #
    # concept_type:
    # - 0 → cause
    # - 1 → effect
    #
    # The dataset stores token spans, so we reconstruct the phrase here.

    # Get start/end indices of the concept span
    start = question['query'][concept_type][0]
    end = question['query'][concept_type][1] + 1

    # Extract tokens from POS-tagged question representation
    # Each entry looks like: (word, POS_tag)
    concept = [t[0] for t in question['question:POS'][start:end]]

    # Join tokens into a readable string
    concept = ' '.join(concept)

    return concept


def get_matryoshka_dims(model_dim: int) -> list[int]:
    # Generates a list of embedding dimensions for Matryoshka training.
    #
    # The goal is to evaluate truncated embeddings at multiple sizes while:
    # - preserving consistency across models
    # - enabling fair comparison (especially at 768 dims)
    #
    # The resulting list includes:
    # - the full embedding dimension (model_dim)
    # - powers of two (64 → ... → <= model_dim)
    # - a fixed anchor dimension (768) for cross-model comparison

    # Use a set to avoid duplicate dimensions.
    dims = {model_dim}

    # Generate powers-of-two dimensions starting from 64.
    # These represent progressively compressed embedding sizes.
    base_dim = 64
    while base_dim < model_dim:
        dims.add(base_dim)
        base_dim *= 2

    # Add 768 as a fixed comparison point across models.
    # Only include it if the model's embedding size supports it.
    if 768 <= model_dim:
        dims.add(768)

    # Return dimensions sorted from largest to smallest.
    # This ordering matches the truncation logic used during training.
    return sorted(dims, reverse=True)
