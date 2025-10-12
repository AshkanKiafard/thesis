import json
from functools import lru_cache

import networkx as nx
import numpy as np
import ollama


def load_graph(file_path):
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
            if (c := d["causal_relation"]["cause"]["concept"]) != (e := d["causal_relation"]["effect"]["concept"])
        ])


@lru_cache(maxsize=None)
def embed_text(text):
    return np.array(ollama.embed(model="nomic-embed-text:latest", input=text.replace("_", "")).embeddings)


def get_embedding_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)
