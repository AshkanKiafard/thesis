import json
import zipfile
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
            if (c := d["causal_relation"]["cause"]["concept"].replace('_', ' ')) != (e := d["causal_relation"]["effect"]["concept"].replace('_', ' '))
        ])


@lru_cache(maxsize=None)
def embed_text(text, model="nomic-embed-text:latest"):
    return np.array(ollama.embed(model=model, input=text.replace("_", "")).embeddings)


def get_embedding_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


def load_embeddings(file_path="data/embeddings/glove.6B.zip"):
    embeddings = {}
    with zipfile.ZipFile(file_path) as z:
        with z.open("glove.6B.300d.txt", 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip().split(' ')
                embeddings[line[0]] = [float(value) for value in line[1:]]
    return embeddings