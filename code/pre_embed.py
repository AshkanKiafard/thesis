import os
import gc
import torch
import numpy as np
from embeddings import Embeder
from utils import load_graph

os.makedirs("data/embeddings", exist_ok=True)

graph = load_graph("data/graphs/causenet-precision.jsonl")

base_models = [
    "sentence-transformers/all-mpnet-base-v2",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "sentence-transformers/multi-qa-mpnet-base-cos-v1"
]

lightning_dir = "data/models/lightning"
fine_tuned_models = []

if os.path.exists(lightning_dir):
    fine_tuned_models = [
        os.path.join(lightning_dir, name).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if os.path.isdir(os.path.join(lightning_dir, name))
    ]
    print(f"Found {len(fine_tuned_models)} fine-tuned models in {lightning_dir}")
else:
    print(f"Warning: Directory {lightning_dir} not found.")

model_queue = base_models + fine_tuned_models

batch_size = 64

for model_path in model_queue:
    print(f"\n{'=' * 50}")
    print(f"PROCESSING MODEL: {model_path}")
    print(f"{'=' * 50}")

    if os.path.sep in model_path:
        raw_name = model_path.split(os.path.sep)[-1]
    else:
        raw_name = model_path.split("/")[-1]

    save_path = f"data/embeddings/{raw_name}_embeddings.npy"

    try:
        embeddings = np.load(save_path, allow_pickle=True).item()
        print(f"Loaded {len(embeddings)} existing embeddings from {save_path}")
    except FileNotFoundError:
        print("No existing cache found. Starting fresh.")
        embeddings = {}

    uncached_nodes = [node for node in graph.nodes if node not in embeddings]
    print(f"Found {len(uncached_nodes)} nodes that need embedding.")

    if len(uncached_nodes) > 0:
        print("Loading model...")
        embeder = Embeder(model_path=model_path, distance_metric='cosine')

        total_batches = (len(uncached_nodes) + batch_size - 1) // batch_size

        for i in range(0, len(uncached_nodes), batch_size):
            batch = uncached_nodes[i:i + batch_size]

            batch_embeddings = [embeder.embed(node) for node in batch]

            for node, emb in zip(batch, batch_embeddings):
                embeddings[node] = emb

            if (i // batch_size) % 10 == 0:
                print(
                    f"Processed batch {i // batch_size + 1}/{total_batches} (Total: {i + len(batch)}/{len(uncached_nodes)})")

        print(f"Saving embeddings to {save_path}...")
        np.save(save_path, embeddings)
        print("Save complete.")

        print(f"Cleaning up memory ...")
        del embeder
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("All nodes already cached. Skipping computation.")

print("\nAll models processed.")