import numpy as np
from embeddings import Embeder
from utils import load_graph

graph = load_graph("data/graphs/causenet-precision.jsonl")
model_path = "data/models/sentence-transformers/multi-qa-mpnet-base-cos-v1_fine-tuned"
model_name = model_path.split("/")[-1]
batch_size = 64
embeder = Embeder(model_path=model_path, distance_metric='cosine')

try:
    embeddings = np.load(f"data/embeddings/{model_name}_embeddings.npy", allow_pickle=True).item()
except FileNotFoundError:
    embeddings = {}

uncached_nodes = [node for node in graph.nodes if node not in embeddings]
print(f"Embedding {len(uncached_nodes)} uncached nodes...")

for i in range(0, len(uncached_nodes), batch_size):
    batch = uncached_nodes[i:i + batch_size]

    batch_embeddings = [embeder.embed(node) for node in batch]

    for node, emb in zip(batch, batch_embeddings):
        embeddings[node] = emb

    print(f"Embedded {i + len(batch)}/{len(uncached_nodes)} nodes")

np.save(f"data/embeddings/{model_name}_embeddings.npy", embeddings)
print(f"All embeddings saved, total nodes: {len(embeddings)}")
