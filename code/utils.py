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
    return np.array(ollama.embed(model="nomic-embed-text:latest", input=text.replace("_", "")).embeddings).squeeze()


def get_embedding_distance(embedding1, embedding2):
    dot = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return 1 - (dot / norm)