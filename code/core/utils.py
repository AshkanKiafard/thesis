import bz2
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import networkx as nx

from core.constants import ActivationFunc, DistanceMetric, LIGHTNING_DIR
from core.embeddings import STEmbedder


@dataclass
class RLGraph:
    # Original-style adjacency list for the RL baseline.
    #
    # Important:
    # - preserves duplicate neighbor actions
    # - preserves insertion order from CauseNet
    # - stores source/relation text per directed edge
    #
    # This is intentionally not a NetworkX graph because nx.DiGraph collapses
    # duplicate (cause, effect) edges.
    adjacency: Dict[str, List[str]]
    edge_sources: Dict[Tuple[str, str], str]
    nodes: set

    def successors(self, node: str) -> List[str]:
        return self.adjacency.get(node, [])

    def get_edge_data(self, source: str, target: str, default=None):
        sentence = self.edge_sources.get((source, target))

        if sentence is None:
            return default if default is not None else {}

        return {"sentence": sentence}


def _open_graph_file(file_path):
    path = str(file_path)

    if path.endswith(".bz2"):
        return bz2.open(path, mode="rt", encoding="utf-8")

    return open(path, encoding="utf-8")


def _build_source(connection, cause, effect):
    # Match original causal-qa-rl graph_utils._build_source.
    if connection == "causes":
        source = cause + " causes " + effect
    elif connection == "Cause":
        source = cause + " can cause " + effect
    elif connection == "cause":
        source = cause + " can cause " + effect
    elif connection == "risks":
        source = cause + " risks " + effect
    elif connection == "symptoms":
        source = effect + " is a symptom of " + cause
    elif connection == "Symptoms":
        source = effect + " is a symptom of " + cause
    elif connection == "Signs and symptoms":
        source = effect + " is a sign or symptom of " + cause
    elif connection == "Causes":
        source = cause + " causes " + effect
    elif connection == "Risk factor":
        source = cause + " is a risk factor for " + effect
    else:
        raise ValueError(f"No source with {connection} connection")

    return source


def _build_causenet_source(d, cause, effect):
    # Match original causal-qa-rl graph_utils._get_source as closely as possible.
    #
    # Original behavior:
    # 1. Return the first clueweb12_sentence or wikipedia_sentence.
    # 2. If none exists, use the LAST source object from the loop.
    #    This is a bit weird, but it is what their code does because the
    #    variable `source` remains bound after the loop.
    sources = d.get("sources", [])

    if not sources:
        return f"{cause} can cause {effect}"

    last_source = None

    for source in sources:
        last_source = source

        if source.get("type") in {"clueweb12_sentence", "wikipedia_sentence"}:
            sentence = source.get("payload", {}).get("sentence", "")
            if sentence:
                return sentence

    source_type = last_source.get("type")
    payload = last_source.get("payload", {})

    if source_type == "wikipedia_list":
        connection = payload.get("list_toc_section_heading")
    else:
        connection = payload.get("infobox_argument")

    if connection is None:
        return f"{cause} can cause {effect}"

    try:
        return _build_source(connection, cause, effect)
    except ValueError:
        # Safer than crashing your whole evaluation if CauseNet has an
        # unexpected source field. The original would crash here.
        return f"{cause} can cause {effect}"


def load_causal_graph(file_path, remove_self_loops=True, use_inverse=False):
    # Loads the causal graph from a JSONL file (one JSON object per line).
    # This graph is used for BFS and A*.
    #
    # It intentionally uses nx.DiGraph:
    # - normal graph traversal behavior
    # - one edge per (cause, effect)
    # - edge attributes store metadata like support and example sentence

    graph = nx.DiGraph()

    with _open_graph_file(file_path) as f:
        for d in map(json.loads, f):
            c = d["causal_relation"]["cause"]["concept"].replace("_", " ")
            e = d["causal_relation"]["effect"]["concept"].replace("_", " ")

            if remove_self_loops and c == e:
                continue

            sentence = _build_causenet_source(d, c, e)

            graph.add_edge(
                c,
                e,
                support=d.get("support", 0),
                sentence=sentence,
            )

            if use_inverse:
                graph.add_edge(
                    e,
                    c,
                    support=d.get("support", 0),
                    sentence=sentence,
                    inverse=True,
                )

    return graph


def load_rl_graph(file_path, remove_self_loops=True, use_inverse=False):
    # Loads the causal graph in a format closer to the original RL repo.
    #
    # Original causal-qa-rl behavior:
    # - graph is a defaultdict(list)
    # - duplicate neighbor entries are preserved
    # - neighbor order follows the CauseNet file order
    # - inverse edges are disabled by default during RL evaluation
    #
    # We do NOT insert the stop action here because rl.py currently prepends
    # the stop action inside _build_action_tensor(). This keeps stop handling
    # centralized in one place.

    adjacency = defaultdict(list)
    edge_sources = {}
    nodes = set()

    with _open_graph_file(file_path) as f:
        for d in map(json.loads, f):
            c = d["causal_relation"]["cause"]["concept"].replace("_", " ")
            e = d["causal_relation"]["effect"]["concept"].replace("_", " ")

            if remove_self_loops and c == e:
                continue

            sentence = _build_causenet_source(d, c, e)

            nodes.add(c)
            nodes.add(e)

            # Preserve duplicate actions and insertion order.
            adjacency[c].append(e)

            # Match original graph_sources behavior:
            # if duplicate (c, e) exists, the last source wins.
            edge_sources[(c, e)] = sentence

            if use_inverse:
                adjacency[e].append(c)
                edge_sources[(e, c)] = sentence

    # Make sure every node has an adjacency list.
    for node in nodes:
        adjacency.setdefault(node, [])

    # Keep the artificial stop node available for embeddings/fallbacks.
    nodes.add("stop stop action")
    adjacency.setdefault("stop stop action", [])

    return RLGraph(
        adjacency=dict(adjacency),
        edge_sources=edge_sources,
        nodes=nodes,
    )


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
    base_dim = 2
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


def parse_activation_func(value: str) -> ActivationFunc:
    value = value.strip().lower()
    if value == "relu":
        return ActivationFunc.RELU
    elif value == "gelu":
        return ActivationFunc.GELU
    else:
        raise ValueError(f"Unsupported activation function: {value}")


def parse_distance_metric(value: str) -> DistanceMetric:
    # Convert a string distance name into the DistanceMetric enum used by STEmbedder.
    value = value.strip().lower()

    if value == "cosine":
        return DistanceMetric.COSINE

    if value in {"euclid", "euclidean"}:
        return DistanceMetric.EUCLIDEAN

    raise ValueError(f"Unsupported distance metric: {value}")


def get_model_distance_metric(model_path: str) -> DistanceMetric:
    # Fine-tuned models exported by finetune_best.py contain training_metadata.json.
    # That file records whether the model was trained/evaluated with cosine or Euclidean distance.
    #
    # Base models from Hugging Face do not have this metadata locally,
    # so we default them to cosine.
    metadata_path = Path(model_path) / "training_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        distance = metadata.get("distance")

        if distance is None:
            raise ValueError(f"Missing distance field in {metadata_path}")

        return parse_distance_metric(distance)

    return DistanceMetric.COSINE


def get_fine_tuned_models(run_suffix: str):
    """
    Load only fine-tuned models belonging to this run suffix.

    Expected final-training export pattern:
    <model>_<activation>_<distance>_<norm>_<mrl>_<run_suffix>_finetuned
    """
    if not os.path.exists(LIGHTNING_DIR):
        return []

    expected_suffix = f"_{run_suffix}_finetuned"

    return [
        os.path.join(LIGHTNING_DIR, name).replace("\\", "/")
        for name in os.listdir(LIGHTNING_DIR)
        if os.path.isdir(os.path.join(LIGHTNING_DIR, name))
           and name != "old"
           and name.endswith(expected_suffix)
    ]
