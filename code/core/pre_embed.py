import argparse
import bz2
import gc
import json
import sys
from pathlib import Path

# code/core/pre_embed.py -> code root is one level above this file.
CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"

# Make code/ importable when this script is executed directly.
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.graph_config import get_graph_label, get_graph_path, graph_choices
from core.utils import get_embedding_cache_suffix, get_fine_tuned_models


BASE_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    # "BAAI/bge-base-en-v1.5",
    # "ibm-granite/granite-embedding-small-english-r2",
    "BAAI/bge-large-en-v1.5",
    "ibm-granite/granite-embedding-english-r2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen3-Embedding-0.6B",
    # "Qwen/Qwen3-Embedding-4B",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute graph-node embeddings for base and fine-tuned models."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Optional single Hugging Face model name or fine-tuned model path. "
            "When set, the normal base+fine-tuned queue is ignored."
        ),
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="best_v2",
        help="Only include fine-tuned models with this suffix, e.g. best_v2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for model encoding. Default: auto.",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        default=None,
        help=(
            "Optional single Hugging Face model name or fine-tuned model path. "
            "Alias for the positional model argument."
        ),
    )
    parser.add_argument(
        "--graph",
        choices=graph_choices(),
        default=None,
        help=(
            "Graph to pre-embed. If omitted, keep the legacy combined "
            "causenet+causalbank cache. Full graph variants are stored in "
            "separate graph-specific cache files."
        ),
    )

    args = parser.parse_args()

    if args.model is not None and args.model_path is not None:
        parser.error("Pass either positional model or --model/--model-path, not both.")

    args.single_model = args.model_path or args.model

    return args


def open_graph_file(file_path):
    file_path = Path(file_path)

    if str(file_path).endswith(".bz2"):
        return bz2.open(file_path, mode="rt", encoding="utf-8")

    return open(file_path, encoding="utf-8")


def load_graph_nodes_for_pre_embed(graph_name, graph_path):
    nodes = set()
    file_name = Path(graph_path).name.lower()

    with open_graph_file(graph_path) as file:
        if file_name.endswith(".jsonl") or file_name.endswith(".jsonl.bz2"):
            for line in file:
                if not line.strip():
                    continue

                item = json.loads(line)
                relation = item["causal_relation"]
                cause = relation["cause"]["concept"].replace("_", " ").strip()
                effect = relation["effect"]["concept"].replace("_", " ").strip()

                if cause and effect and cause != effect:
                    nodes.add(cause)
                    nodes.add(effect)

            return nodes

        if file_name.endswith(".txt"):
            for line in file:
                parts = line.rstrip("\n").split("\t")

                if not parts or "->" not in parts[0]:
                    continue

                cause, effect = parts[0].split("->", 1)
                cause = cause.replace("_", " ").strip().lower()
                effect = effect.replace("_", " ").strip().lower()

                if cause and effect and cause != effect:
                    nodes.add(cause)
                    nodes.add(effect)

            return nodes

    raise ValueError(f"Could not infer graph format for {graph_name}: {graph_path}")


def load_embedding_cache(save_path):
    from core.embeddings import load_st_embedding_cache_index

    embeddings, text_to_idx, vectors = load_st_embedding_cache_index(save_path)
    num_embeddings = len(embeddings) + len(text_to_idx)

    if num_embeddings:
        print(f"Loaded {num_embeddings} existing embeddings from {save_path}")

    else:
        print("No existing cache found. Starting fresh.")

    return embeddings, text_to_idx, vectors


def get_embedding_cache_path(embeddings_dir, model_path, cache_suffix=None):
    raw_name = Path(model_path).name

    if cache_suffix:
        raw_name = f"{raw_name}_{cache_suffix}"

    return embeddings_dir / f"{raw_name}_embeddings.npy"


def pre_embed_model(
    model_path,
    graph_nodes,
    embeddings_dir,
    batch_size,
    embedding_device,
    cache_suffix=None,
):
    import torch

    from core.embeddings import STEmbedder, save_st_embedding_cache
    from core.utils import get_model_distance_metric

    print(f"\n{'=' * 50}")
    print(f"PROCESSING MODEL: {model_path}")
    print(f"{'=' * 50}")

    save_path = get_embedding_cache_path(
        embeddings_dir,
        model_path,
        cache_suffix=cache_suffix,
    )

    embeddings, text_to_idx, vectors = load_embedding_cache(save_path)

    uncached_nodes = [
        node
        for node in graph_nodes
        if node not in embeddings and node not in text_to_idx
    ]
    print(f"Found {len(uncached_nodes)} nodes that need embedding.")

    if not uncached_nodes:
        print("All nodes already cached. Skipping computation.")
        return

    print("Loading model...")

    distance_metric = get_model_distance_metric(model_path)
    print(f"Distance metric: {distance_metric}")

    embeder = STEmbedder(
        model_path=model_path,
        distance_metric=distance_metric,
        device=embedding_device,
        cache_suffix=cache_suffix,
    )

    total_batches = (len(uncached_nodes) + batch_size - 1) // batch_size
    new_embeddings = {}

    for start in range(0, len(uncached_nodes), batch_size):
        batch = uncached_nodes[start : start + batch_size]

        # Preload on GPU for efficient model inference, then persist the CPU
        # NumPy representation used by the embedding cache.
        embeder.preload(batch, batch_size=batch_size, save=False)
        batch_embeddings = [embeder.embed_numpy(node) for node in batch]

        for node, emb in zip(batch, batch_embeddings):
            new_embeddings[node] = emb

        batch_index = start // batch_size
        if batch_index % 10 == 0:
            print(
                f"Processed batch {batch_index + 1}/{total_batches} "
                f"(Total: {start + len(batch)}/{len(uncached_nodes)})"
            )

    print(f"Saving embeddings to {save_path}...")
    embeddings.update(new_embeddings)
    save_st_embedding_cache(
        save_path,
        embeddings,
        existing_text_to_idx=text_to_idx,
        existing_vectors=vectors,
    )
    print("Save complete.")

    print("Cleaning up memory ...")
    del embeder
    del embeddings
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    embeddings_dir = DATA_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run suffix: {args.run_suffix}")
    print(f"Embedding device: {args.embedding_device}")

    selected_graphs = (
        (args.graph,)
        if args.graph is not None
        else ("causenet", "causalbank")
    )
    cache_suffix = get_embedding_cache_suffix(args.graph)

    graph_nodes = set()
    for graph_name in selected_graphs:
        graph_path = get_graph_path(graph_name)
        if not graph_path.is_absolute():
            graph_path = CODE_ROOT / graph_path

        print(f"Loading {get_graph_label(graph_name)} nodes from: {graph_path}")
        current_nodes = load_graph_nodes_for_pre_embed(graph_name, graph_path)
        graph_nodes.update(current_nodes)
        print(f"Loaded {len(current_nodes)} {graph_name} nodes.")

    graph_nodes = sorted(graph_nodes)
    if args.graph is None:
        print(f"Combined graph nodes: {len(graph_nodes)}")
    else:
        print(f"{args.graph} graph nodes: {len(graph_nodes)}")

    if cache_suffix:
        print(f"Using graph-specific embedding cache suffix: {cache_suffix}")
    else:
        print("Using legacy shared embedding cache.")

    if args.single_model is not None:
        model_queue = [args.single_model]
        print("Single model provided. Ignoring base+fine-tuned model queue.")
    else:
        fine_tuned_models = get_fine_tuned_models(args.run_suffix)
        print(
            f"Found {len(fine_tuned_models)} fine-tuned models for suffix "
            f"'{args.run_suffix}'."
        )

        model_queue = BASE_MODELS + fine_tuned_models

    print("Model queue:")
    for model_path in model_queue:
        print(f"  {model_path}")

    for model_path in model_queue:
        pre_embed_model(
            model_path=model_path,
            graph_nodes=graph_nodes,
            embeddings_dir=embeddings_dir,
            batch_size=args.batch_size,
            embedding_device=args.embedding_device,
            cache_suffix=cache_suffix,
        )

    print("\nAll models processed.")


if __name__ == "__main__":
    main()
