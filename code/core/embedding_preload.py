import time
from pathlib import Path

from core.config import DEFAULT_EMBEDDING_BATCH_SIZE
from core.embeddings import load_st_embedding_cache_index
from core.indexed_graph import build_indexed_graph
from core.utils import is_valid_graph_node


def load_embedding_cache(save_path, node_universe):
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

    return Path(embeddings_dir) / f"{raw_name}_embeddings.npy"


def preload_graph_embeddings(
    embeder,
    graph,
    batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
    save_cache=True,
):
    """
    Load or create graph-node embeddings and expose one runtime index table.

    The persisted cache is the reusable preembedding artifact. The torch
    embedding table and IndexedGraph are process-local runtime structures, so
    evaluation still has to load them before timed traversal. Missing rows are
    encoded here as a fallback when pre_embed.py was not run beforehand.
    """
    nodes = [
        node
        for node in graph.nodes
        if is_valid_graph_node(node)
    ]
    skipped_nodes = graph.number_of_nodes() - len(nodes)
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
    if skipped_nodes:
        print(
            "Skipping blank or punctuation-only graph node(s) during "
            f"embedding index preload: {skipped_nodes:,}.",
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
    nodes = [
        node
        for node in graph.nodes
        if is_valid_graph_node(node)
    ]
    skipped_nodes = len(graph.nodes) - len(nodes)
    entity_texts = nodes + ["stop stop action"]

    print(
        "Preloading RL GloVe entity embeddings "
        f"({len(entity_texts):,} entities)..."
    )
    if skipped_nodes:
        print(
            "Skipping blank or punctuation-only RL graph node(s) during "
            f"embedding preload: {skipped_nodes:,}.",
            flush=True,
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
