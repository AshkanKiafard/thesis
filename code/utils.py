import json
from typing import Callable

import networkx as nx

from embeddings import STEmbeder


def load_graph(file_path, remove_self_loops=True):
    with open(file_path) as f:
        return nx.DiGraph([
            (
                c,
                e,
                {
                    "support": d.get("support", 0),
                    "sentence": d.get("sources", [{}])[0].get("payload", {}).get("sentence", "")
                }
            )
            for d in map(json.loads, f)
            if ((c := d["causal_relation"]["cause"]["concept"].replace('_', ' ')) !=
                (e := d["causal_relation"]["effect"]["concept"].replace('_', ' '))) or not remove_self_loops
        ])


def traverse_graph(graph: nx.DiGraph, start_node: str, end_node: str, embeder: STEmbeder, config, strategy_fn: Callable):
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return [], 0

    return strategy_fn(graph, start_node, end_node, embeder, config)


def get_concept(question: dict, concept_type: int) -> str:
    start = question['query'][concept_type][0]
    end = question['query'][concept_type][1] + 1
    concept = [t[0] for t in question['question:POS'][start:end]]
    concept = ' '.join(concept)
    return concept
