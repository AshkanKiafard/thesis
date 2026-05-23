import bz2
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx

from core.constants import ActivationFunc, DistanceMetric, LIGHTNING_DIR


def get_embedding_cache_suffix(graph_name):
    if graph_name == "causenet_full":
        return graph_name

    return None


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


def _detect_graph_format(file_path, graph_format="auto"):
    graph_format = graph_format.lower()

    if graph_format != "auto":
        if graph_format not in {"causenet", "lexical_ceg"}:
            raise ValueError("graph_format must be one of: auto, causenet, lexical_ceg")
        return graph_format

    file_name = Path(file_path).name.lower()

    if file_name.endswith(".jsonl") or file_name.endswith(".jsonl.bz2"):
        return "causenet"

    if file_name.endswith(".txt"):
        return "lexical_ceg"

    raise ValueError(
        f"Could not infer graph format from '{file_path}'. "
        "Pass graph_format='causenet' or graph_format='lexical_ceg'."
    )


def _normalize_causenet_concept(value):
    return value.replace("_", " ").strip()


def _normalize_lexical_ceg_concept(value):
    return value.replace("_", " ").strip().lower()


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


def _iter_causenet_edges(file_path, remove_self_loops=True):
    with _open_graph_file(file_path) as f:
        for d in map(json.loads, f):
            cause = _normalize_causenet_concept(
                d["causal_relation"]["cause"]["concept"]
            )
            effect = _normalize_causenet_concept(
                d["causal_relation"]["effect"]["concept"]
            )

            if remove_self_loops and cause == effect:
                continue

            yield (
                cause,
                effect,
                {
                    "support": d.get("support", 0),
                    "sentence": _build_causenet_source(d, cause, effect),
                    "graph_source": "causenet",
                },
            )


def _iter_lexical_ceg_edges(
    file_path,
    remove_self_loops=True,
):
    # CEG format:
    # cause->effect<TAB>count<TAB>necessity_score<TAB>sufficiency_score
    with _open_graph_file(file_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) < 4 or "->" not in parts[0]:
                continue

            try:
                count = int(parts[1])
                necessity_score = float(parts[2])
                sufficiency_score = float(parts[3])
            except ValueError:
                continue

            cause, effect = parts[0].split("->", 1)
            cause = _normalize_lexical_ceg_concept(cause)
            effect = _normalize_lexical_ceg_concept(effect)

            if not cause or not effect:
                continue

            if remove_self_loops and cause == effect:
                continue

            yield (
                cause,
                effect,
                {
                    "support": count,
                    "sentence": f"{cause} can cause {effect}",
                    "graph_source": "lexical_ceg",
                    "count": count,
                    "necessity_score": necessity_score,
                    "sufficiency_score": sufficiency_score,
                    "causality_score": max(necessity_score, sufficiency_score),
                },
            )


def _iter_graph_edges(
    file_path,
    remove_self_loops=True,
    graph_format="auto",
):
    graph_format = _detect_graph_format(file_path, graph_format)

    if graph_format == "causenet":
        return _iter_causenet_edges(
            file_path=file_path,
            remove_self_loops=remove_self_loops,
        )

    return _iter_lexical_ceg_edges(
        file_path=file_path,
        remove_self_loops=remove_self_loops,
    )


def load_causal_graph(
    file_path,
    remove_self_loops=True,
    use_inverse=False,
    graph_format="auto",
    progress_every=None,
    progress_label="causal graph",
):
    # Loads CauseNet JSONL or lexical CEG TXT for BFS, A*, and Dijkstra.
    #
    # It intentionally uses nx.DiGraph:
    # - normal graph traversal behavior
    # - one edge per (cause, effect)
    # - edge attributes store metadata like support and example sentence

    graph = nx.DiGraph()

    for edge_index, (cause, effect, edge_attrs) in enumerate(
        _iter_graph_edges(
            file_path=file_path,
            remove_self_loops=remove_self_loops,
            graph_format=graph_format,
        ),
        start=1,
    ):
        if progress_every and edge_index % progress_every == 0:
            print(f"Loaded {edge_index:,} edges into {progress_label}...", flush=True)

        graph.add_edge(cause, effect, **edge_attrs)

        if use_inverse:
            graph.add_edge(
                effect,
                cause,
                **edge_attrs,
                inverse=True,
            )

    return graph


def load_graph_nodes(
    file_path,
    remove_self_loops=True,
    graph_format="auto",
):
    nodes = set()

    for cause, effect, _ in _iter_graph_edges(
        file_path=file_path,
        remove_self_loops=remove_self_loops,
        graph_format=graph_format,
    ):
        nodes.add(cause)
        nodes.add(effect)

    return nodes


def load_rl_graph(
    file_path,
    remove_self_loops=True,
    use_inverse=False,
    graph_format="auto",
    progress_every=None,
    progress_label="RL graph",
):
    # Loads CauseNet JSONL or lexical CEG TXT in the RL baseline format.
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

    for edge_index, (cause, effect, edge_attrs) in enumerate(
        _iter_graph_edges(
            file_path=file_path,
            remove_self_loops=remove_self_loops,
            graph_format=graph_format,
        ),
        start=1,
    ):
        if progress_every and edge_index % progress_every == 0:
            print(f"Loaded {edge_index:,} edges into {progress_label}...", flush=True)

        sentence = edge_attrs["sentence"]

        nodes.add(cause)
        nodes.add(effect)

        # Preserve duplicate actions and insertion order.
        adjacency[cause].append(effect)

        # Match original graph_sources behavior:
        # if duplicate (cause, effect) exists, the last source wins.
        edge_sources[(cause, effect)] = sentence

        if use_inverse:
            adjacency[effect].append(cause)
            edge_sources[(effect, cause)] = sentence

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
    embeder: Any,
    strategy_fn: Callable,
    config: dict = None,
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
    start = question["query"][concept_type][0]
    end = question["query"][concept_type][1] + 1

    # Extract tokens from POS-tagged question representation
    # Each entry looks like: (word, POS_tag)
    concept = [t[0] for t in question["question:POS"][start:end]]

    # Join tokens into a readable string
    concept = " ".join(concept)

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


def sort_model_queue(model_queue, run_suffix):
    """
    Sort models so each base model is followed by its fine-tuned variant.

    Example:
    MPNet base
    MPNet finetuned
    BGE base
    BGE finetuned
    ...

    Base models come before fine-tuned variants.
    """

    def sort_key(model_path):
        model_name = model_path.split("/")[-1]

        normalized_name = (
            model_name.replace(
                f"_relu_cosine_nonorm_matryoshka_{run_suffix}_finetuned", ""
            )
            .replace(f"_relu_euclid_nonorm_matryoshka_{run_suffix}_finetuned", "")
            .replace(f"_gelu_cosine_nonorm_matryoshka_{run_suffix}_finetuned", "")
            .replace(f"_gelu_euclid_nonorm_matryoshka_{run_suffix}_finetuned", "")
        )

        is_finetuned = "finetuned" in model_name

        return normalized_name, is_finetuned

    return sorted(model_queue, key=sort_key)
