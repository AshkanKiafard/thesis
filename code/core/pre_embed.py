import argparse
import bz2
import gc
import json
import sys
import time
from pathlib import Path

# code/core/pre_embed.py -> code root is one level above this file.
CODE_ROOT = Path(__file__).resolve().parents[1]

# Make code/ importable when this script is executed directly.
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.config import BASE_MODELS, DATA_DIR, EMBEDDINGS_DIR
from core.graph_config import get_graph_label, get_graph_path, graph_choices
from core.indexed_graph import build_indexed_graph
from core.utils import (
    MERGED_NODE_UNIVERSE,
    get_ablation_fine_tuned_models,
    get_embedding_cache_suffix,
    get_fine_tuned_models,
    get_node_universe_for_graph,
    get_node_universe_path,
    write_node_universe,
)


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
        default="v3",
        help="Only include fine-tuned models with this suffix, e.g. v3.",
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
            "Graph to pre-embed. If omitted, pre-embed the combined "
            "causenet+causalbank node set into the shared JSONL/mmap cache. "
            "Full graph variants are stored in separate graph-specific cache "
            "files."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Pre-embed only the activation/distance ablation models for "
            "--run-suffix. Ignores the normal base+fine-tuned queue."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help=(
            "Optional Matryoshka dimension to pre-embed for the selected "
            "single model. This writes the dim-specific cache instead of "
            "forcing a full-dimension cache."
        ),
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help=(
            "Only fill the persisted embedding cache. By default this script "
            "also builds the runtime embedding index table once, which validates "
            "that evaluation can load the same indexed cache."
        ),
    )

    args = parser.parse_args()

    if args.model is not None and args.model_path is not None:
        parser.error("Pass either positional model or --model/--model-path, not both.")

    args.single_model = args.model_path or args.model

    if args.ablation and args.single_model is not None:
        parser.error("--ablation cannot be combined with an explicit model path.")

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


def load_embedding_cache(save_path, node_universe):
    from core.embeddings import load_st_embedding_cache_index

    embeddings, text_to_idx, vectors = load_st_embedding_cache_index(
        save_path,
        node_universe=node_universe,
    )
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


def preload_graph_embeddings(embeder, graph, batch_size=64, save_cache=True):
    """
    Load or create graph-node embeddings and expose one runtime index table.

    The persisted cache is the reusable preembedding artifact. The torch
    embedding table and IndexedGraph are process-local runtime structures, so
    evaluation still has to load them before timed traversal. Missing rows are
    encoded here as a fallback when pre_embed.py was not run beforehand.
    """
    nodes = list(graph.nodes)
    cached_before = sum(
        1
        for node in nodes
        if embeder.has_cached_embedding(node)
    )
    active_dim = embeder.get_active_embedding_dim()
    model_dim = embeder.get_model_dim()

    print(
        f"Preparing graph embedding index for {embeder.get_model_name()} "
        f"({cached_before:,}/{len(nodes):,} cached, "
        f"active dim {active_dim}, model dim {model_dim})...",
        flush=True,
    )

    start_time = time.time()
    added = embeder.prepare_embedding_index(
        nodes,
        batch_size=batch_size,
        save=save_cache,
        discard_tensor_cache=True,
        populate_tensor_cache=False,
        texts_are_unique=True,
    )

    index_start = time.time()
    indexed_graph = build_indexed_graph(
        graph,
        embeder.indexed_text_to_idx,
    )
    print(
        "Indexed graph built: "
        f"{len(indexed_graph.idx_to_node):,} nodes in "
        f"{time.time() - index_start:.1f}s",
        flush=True,
    )

    elapsed = time.time() - start_time
    print(
        f"Graph embedding index ready: {added:,} added, "
        f"{cached_before + added:,}/{len(nodes):,} graph nodes cached, "
        f"{len(embeder.indexed_text_to_idx):,} indexed, "
        f"{elapsed:.1f}s",
        flush=True,
    )

    return indexed_graph


def preload_rl_embeddings(embeder, graph, data=None, batch_size=4096):
    """
    Preload the GloVe embedding caches used by the RL traversal baseline.

    RL uses separate entity/question/relation embedding paths, so warming these
    caches before timed evaluation keeps preprocessing out of traversal timing.
    """
    nodes = list(graph.nodes)
    entity_texts = nodes + ["stop stop action"]

    print(
        "Preloading RL GloVe entity embeddings "
        f"({len(entity_texts):,} entities)..."
    )
    start_time = time.time()
    added_entities = embeder.preload_entities(
        entity_texts,
        batch_size=batch_size,
    )

    question_texts = []
    if data is not None:
        graph_nodes = set(graph.nodes)

        for item in data:
            cause = item["cause"]
            effect = item["effect"]

            if cause not in graph_nodes or effect not in graph_nodes:
                continue

            question_texts.append(
                item.get("question", f"can {cause} cause {effect}?")
            )

    added_questions = embeder.preload_questions(question_texts)
    added_relations = embeder.preload_relations(["stop"])
    elapsed = time.time() - start_time

    print(
        "RL GloVe preload complete: "
        f"{added_entities:,} entities added, "
        f"{added_questions:,} questions added, "
        f"{added_relations:,} relations added, "
        f"{elapsed:.1f}s"
    )


def pre_embed_model(
    model_path,
    graph_nodes,
    embeddings_dir,
    batch_size,
    embedding_device,
    cache_suffix=None,
    node_universe=None,
    dim=None,
    build_index=True,
):
    import torch

    from core.embeddings import STEmbedder, save_st_embedding_cache
    from core.utils import get_model_distance_metric

    print(f"\n{'=' * 50}")
    print(f"PROCESSING MODEL: {model_path}")
    print(f"{'=' * 50}")

    print("Loading model...")

    distance_metric = get_model_distance_metric(model_path)
    print(f"Distance metric: {distance_metric}")

    embeder = STEmbedder(
        model_path=model_path,
        distance_metric=distance_metric,
        device=embedding_device,
        cache_suffix=cache_suffix,
        node_universe=node_universe,
    )

    model_dim = embeder.get_model_dim()
    if dim is not None:
        if dim > model_dim:
            raise ValueError(
                f"Requested dim {dim}, but {model_path} only has "
                f"{model_dim} embedding dimensions."
            )
        embeder.set_matryoshka_dim(dim)

    active_dim = embeder.get_active_embedding_dim()
    print(f"Model dim: {model_dim}")
    print(f"Active dim: {active_dim}")

    if build_index:
        cached_before = sum(
            1
            for node in graph_nodes
            if embeder.has_cached_embedding(node)
        )
        print(
            "Preparing embedding cache and runtime index table "
            f"({cached_before:,}/{len(graph_nodes):,} cached).",
            flush=True,
        )
        added = embeder.prepare_embedding_index(
            graph_nodes,
            batch_size=batch_size,
            save=True,
            discard_tensor_cache=True,
            populate_tensor_cache=False,
            texts_are_unique=True,
        )
        print(
            "Preembedding complete: "
            f"{added:,} newly encoded, "
            f"{len(embeder.indexed_text_to_idx):,} indexed rows.",
            flush=True,
        )
        print("Cleaning up memory ...")
        del embeder
        gc.collect()
        torch.cuda.empty_cache()
        return

    save_path = get_embedding_cache_path(
        embeddings_dir,
        model_path,
        cache_suffix=cache_suffix,
    )

    embeddings, text_to_idx, vectors = load_embedding_cache(
        save_path,
        node_universe,
    )

    uncached_nodes = [
        node
        for node in graph_nodes
        if node not in embeddings and node not in text_to_idx
    ]
    print(f"Found {len(uncached_nodes)} nodes that need embedding.")

    if not uncached_nodes:
        print("All nodes already cached. Skipping computation.")
        del embeder
        gc.collect()
        torch.cuda.empty_cache()
        return

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
                f"(Total: {start + len(batch)}/{len(uncached_nodes)})",
                flush=True,
            )

    print(f"Saving embeddings to {save_path}...")
    embeddings.update(new_embeddings)
    save_st_embedding_cache(
        save_path,
        embeddings,
        existing_text_to_idx=text_to_idx,
        existing_vectors=vectors,
        node_universe=node_universe,
        node_order=graph_nodes,
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
    if args.dim is not None and args.dim <= 0:
        raise ValueError("--dim must be greater than 0")
    if args.dim is not None and args.single_model is None and not args.ablation:
        raise ValueError("--dim requires a single model path or --ablation")
    if args.dim is not None and args.no_index:
        raise ValueError("--dim-specific preembedding requires index mode")

    embeddings_dir = EMBEDDINGS_DIR
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run suffix: {args.run_suffix}")
    print(f"Embedding device: {args.embedding_device}")
    if args.dim is not None:
        print(f"Matryoshka dim: {args.dim}")
    print(f"Build runtime embedding index: {not args.no_index}")

    node_universe = get_node_universe_for_graph(args.graph)
    selected_graphs = (
        ("causenet", "causalbank")
        if node_universe == MERGED_NODE_UNIVERSE
        else (args.graph,)
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
    if node_universe == MERGED_NODE_UNIVERSE:
        print(
            "Merged CauseNet precision + CausalBank graph nodes: "
            f"{len(graph_nodes)}"
        )
    else:
        print(f"{args.graph} graph nodes: {len(graph_nodes)}")

    node_universe_path = get_node_universe_path(embeddings_dir, node_universe)
    write_node_universe(node_universe_path, graph_nodes)
    print(
        "Wrote universal node-order JSONL for "
        f"{node_universe}: {node_universe_path}"
    )

    if cache_suffix:
        print(f"Using graph-specific embedding cache suffix: {cache_suffix}")
    else:
        print("Using shared JSONL/mmap embedding cache.")

    if args.single_model is not None:
        model_queue = [args.single_model]
        print("Single model provided. Ignoring base+fine-tuned model queue.")
    elif args.ablation:
        model_queue = get_ablation_fine_tuned_models(args.run_suffix)
        if not model_queue:
            raise FileNotFoundError(
                "No ablation fine-tuned models found for run suffix "
                f"'{args.run_suffix}' in {DATA_DIR / 'models' / 'lightning'}"
            )
        print("Ablation mode enabled. Ignoring base+standard fine-tuned models.")
    else:
        fine_tuned_models = get_fine_tuned_models(args.run_suffix)
        print(
            f"Found {len(fine_tuned_models)} fine-tuned models for suffix "
            f"'{args.run_suffix}'."
        )

        model_queue = list(BASE_MODELS) + fine_tuned_models

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
            node_universe=node_universe,
            dim=args.dim,
            build_index=not args.no_index,
        )

    print("\nAll models processed.")


if __name__ == "__main__":
    main()
