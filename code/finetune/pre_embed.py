import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

# code/finetune/pre_embed.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from core.embeddings import STEmbedder, DistanceMetric
from core.utils import load_graph

embeddings_dir = DATA_DIR / "embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

# We only need the graph structure and node names here.
graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl", False)

base_models = [
    # Strong general-purpose baseline
    "sentence-transformers/all-mpnet-base-v2",

    # Lightweight embedding models
    "BAAI/bge-base-en-v1.5",
    "ibm-granite/granite-embedding-small-english-r2",

    # Higher-capacity embedding models
    "BAAI/bge-large-en-v1.5",
    "ibm-granite/granite-embedding-english-r2",
    "mixedbread-ai/mxbai-embed-large-v1",

    # Large language model-based embeddings
    "Qwen/Qwen3-Embedding-0.6B",

    # Very large model (high memory requirements)
    "Qwen/Qwen3-Embedding-4B",
]

lightning_dir = DATA_DIR / "models" / "lightning"
fine_tuned_models = []

# Collect all fine-tuned models that were exported into the Lightning model directory.
if lightning_dir.exists():
    fine_tuned_models = [
        str((lightning_dir / name).resolve()).replace("\\", "/")
        for name in os.listdir(lightning_dir)
        if (lightning_dir / name).is_dir()
    ]
    print(f"Found {len(fine_tuned_models)} fine-tuned models in {lightning_dir}")
else:
    print(f"Warning: Directory {lightning_dir} not found.")

# Process base models first, then any discovered fine-tuned models.
model_queue = base_models + fine_tuned_models

batch_size = 64

for model_path in model_queue:
    print(f"\n{'=' * 50}")
    print(f"PROCESSING MODEL: {model_path}")
    print(f"{'=' * 50}")

    # Extract a clean model name for the embedding cache filename.
    raw_name = Path(model_path).name
    save_path = embeddings_dir / f"{raw_name}_embeddings.npy"

    try:
        # Reuse an existing cache if available so interrupted runs can resume.
        embeddings = np.load(save_path, allow_pickle=True).item()
        print(f"Loaded {len(embeddings)} existing embeddings from {save_path}")
    except FileNotFoundError:
        print("No existing cache found. Starting fresh.")
        embeddings = {}

    # Only embed nodes that are still missing from the cache.
    uncached_nodes = [node for node in graph.nodes if node not in embeddings]
    print(f"Found {len(uncached_nodes)} nodes that need embedding.")

    if len(uncached_nodes) > 0:
        print("Loading model...")
        embeder = STEmbedder(model_path=model_path, distance_metric=DistanceMetric.COSINE)

        total_batches = (len(uncached_nodes) + batch_size - 1) // batch_size

        for i in range(0, len(uncached_nodes), batch_size):
            batch = uncached_nodes[i:i + batch_size]

            # Embed the current batch one node at a time through the wrapper.
            batch_embeddings = [embeder.embed(node) for node in batch]

            for node, emb in zip(batch, batch_embeddings):
                embeddings[node] = emb

            # Print progress every 10 batches so the console is still readable.
            if (i // batch_size) % 10 == 0:
                print(
                    f"Processed batch {i // batch_size + 1}/{total_batches} "
                    f"(Total: {i + len(batch)}/{len(uncached_nodes)})"
                )

        print(f"Saving embeddings to {save_path}...")
        np.save(save_path, embeddings)
        print("Save complete.")

        # Explicit cleanup helps when multiple large models are processed in one run.
        print("Cleaning up memory ...")
        del embeder
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("All nodes already cached. Skipping computation.")

print("\nAll models processed.")