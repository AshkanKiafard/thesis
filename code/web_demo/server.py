from __future__ import annotations

import json
import math
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

# Transformers may import model classes that use torch.compile at import time.
# The web demo does inference only, and disabling TorchDynamo avoids fragile
# Inductor imports on Windows environments with mixed torch package versions.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import (
    BASE_MODELS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_RUN_SUFFIX as CONFIG_DEFAULT_RUN_SUFFIX,
    EMBEDDING_INDEX_MIN_SUCCESSORS,
)
from core.constants import (
    EMBEDDINGS_DIR,
    EVALUATION_DIR,
    LIGHTNING_MODELS_DIR,
    REPO_ROOT,
    WEB_DEMO_GRAPH_CACHE_DIR,
    WEB_DEMO_GRAPH_PREVIEW_CACHE_DIR,
)
from core.graph_config import DEFAULT_GRAPH_NAME, GRAPH_CONFIGS, get_graph_label
from core.utils import (
    format_model_display_name,
    get_embedding_cache_suffix,
    get_fine_tuned_models,
    get_model_config_labels,
    get_matryoshka_dims,
    get_model_distance_metric,
    get_node_universe_for_graph,
    get_node_universe_path,
    load_causal_graph,
    traverse_graph,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
GRAPH_CACHE_DIR = WEB_DEMO_GRAPH_CACHE_DIR
VISUAL_CACHE_DIR = WEB_DEMO_GRAPH_PREVIEW_CACHE_DIR

DEFAULT_RUN_SUFFIX = os.environ.get("WEB_DEMO_RUN_SUFFIX", CONFIG_DEFAULT_RUN_SUFFIX)
EMBEDDING_DEVICE = os.environ.get("WEB_DEMO_EMBEDDING_DEVICE", "auto")
ASTAR_MAX_VISITS = int(os.environ.get("WEB_DEMO_ASTAR_MAX_VISITS", "-1"))
SUBGRAPH_LIMIT = int(os.environ.get("WEB_DEMO_SUBGRAPH_LIMIT", "240"))
MODEL_WARMUP_BATCH_SIZE = int(
    os.environ.get(
        "WEB_DEMO_MODEL_WARMUP_BATCH_SIZE",
        str(DEFAULT_EMBEDDING_BATCH_SIZE),
    )
)
EDGE_LIMIT_FACTOR = 4
GRAPH_CACHE_VERSION = 1
VISUAL_CACHE_VERSION = 2
DISABLE_GRAPH_DISK_CACHE = (
    os.environ.get("WEB_DEMO_DISABLE_GRAPH_DISK_CACHE", "").lower()
    in {"1", "true", "yes"}
)
PRELOAD_MODELS_SETTING = os.environ.get("WEB_DEMO_PRELOAD_MODELS", "all").strip()
PRELOAD_MODEL_RUNTIMES = (
    PRELOAD_MODELS_SETTING.lower()
    not in {"0", "false", "no", "none", "off"}
)
SUPPORTED_DEMO_GRAPHS = ("causenet", "causalbank")


def parse_demo_graph_choices() -> tuple[str, ...]:
    requested_graphs = os.environ.get(
        "WEB_DEMO_GRAPHS",
        ",".join(SUPPORTED_DEMO_GRAPHS),
    )
    choices = []

    for raw_name in requested_graphs.split(","):
        graph_name = raw_name.strip()
        if not graph_name:
            continue
        if graph_name not in SUPPORTED_DEMO_GRAPHS:
            print(
                "Ignoring unsupported web-demo graph "
                f"'{graph_name}'. Supported demo graphs: "
                f"{', '.join(SUPPORTED_DEMO_GRAPHS)}.",
                flush=True,
            )
            continue
        if graph_name not in choices:
            choices.append(graph_name)

    return tuple(choices or SUPPORTED_DEMO_GRAPHS)


DEMO_GRAPH_CHOICES = parse_demo_graph_choices()

MODEL_DIM_HINTS = {
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
}


@dataclass
class GraphBundle:
    name: str
    label: str
    path: Path
    graph: Any
    nodes: list[str]
    lowercase_to_node: dict[str, str]


@dataclass
class ModelRuntime:
    model_path: str
    cache_suffix: str | None
    node_universe: str
    embedder: Any
    indexed_graphs: dict[str, Any]
    lock: Any = field(default_factory=threading.RLock)


class AStarConfig(BaseModel):
    astar_max_visits: int | None = None
    embedding_index_min_successors: int | None = None


class AStarRequest(BaseModel):
    graph: str = Field(default=DEFAULT_GRAPH_NAME)
    model: str
    dim: int
    source: str
    target: str
    config: AStarConfig = Field(default_factory=AStarConfig)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    preload_inference_modules()
    preload_demo_graphs()
    preload_demo_model_runtimes()
    yield


app = FastAPI(
    title="Causal Graph A* Demo",
    description="Interactive Webis-style demo for causal-graph A* inference.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# The web demo intentionally exposes only the small/filtered graph variants.
# They are preloaded during FastAPI startup so A* timing is not polluted by
# graph deserialization. The parsed graph cache is still lazy per graph name
# inside get_graph_bundle(), but startup calls it for every exposed demo graph
# before the UI can respond. By default every model exposed by the UI is also
# warmed once: one runtime per model/cache suffix, with a full-dimension
# embedding table that lower matryoshka dims reuse by slicing. CLI
# training/evaluation code does not use this module, so its loading behavior is
# unchanged.
_graph_cache: dict[str, GraphBundle] = {}
_graph_lock = threading.Lock()
_graph_warmup_edges: dict[str, tuple[str, str]] = {}
_preload_status: dict[str, dict[str, Any]] = {}
_preload_complete = False
_model_cache: dict[tuple[str, str | None], ModelRuntime] = {}
_model_lock = threading.Lock()
_model_preload_status: dict[str, Any] = {
    "enabled": PRELOAD_MODEL_RUNTIMES,
    "setting": PRELOAD_MODELS_SETTING or "all",
    "loaded": False,
    "models": [],
}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def options():
    models = discover_models()
    available_graphs = get_available_demo_graphs()
    graphs = [
        {
            "id": graph_name,
            "label": get_graph_label(graph_name),
            "path": str(resolve_code_path(config["path"])),
        }
        for graph_name in available_graphs
        for config in (GRAPH_CONFIGS[graph_name],)
    ]
    default_graph = (
        DEFAULT_GRAPH_NAME
        if DEFAULT_GRAPH_NAME in available_graphs
        else available_graphs[0] if available_graphs else None
    )

    return {
        "graphs": graphs,
        "models": models,
        "defaults": {
            "graph": default_graph,
            "model": models[0]["id"] if models else None,
            "dim": models[0]["dims"][0] if models else None,
        },
        "advanced": {
            "run_suffix": DEFAULT_RUN_SUFFIX,
            "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
        },
        "limits": {
            "initial_subgraph_nodes": SUBGRAPH_LIMIT,
            "astar_max_visits": ASTAR_MAX_VISITS,
        },
    }


@app.get("/api/preload-status")
def preload_status():
    return {
        "complete": _preload_complete,
        "graphs": [
            {
                "id": graph_name,
                "label": get_graph_label(graph_name),
                **_preload_status.get(graph_name, {"loaded": False}),
            }
            for graph_name in DEMO_GRAPH_CHOICES
        ],
        "models": _model_preload_status,
    }


@app.get("/api/config")
def config_defaults(
    graph: str = Query(default=DEFAULT_GRAPH_NAME),
    model: str = Query(...),
    dim: int = Query(..., ge=1),
):
    validate_demo_graph_name(graph)

    model_option = get_model_option(model)
    if model_option is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    if dim not in model_option["dims"]:
        raise HTTPException(
            status_code=400,
            detail=f"Dimension {dim} is not available for {model_option['label']}.",
        )

    cap = get_default_astar_max_visits(graph, model, dim)
    return {
        "astar_max_visits": cap["value"],
        "astar_max_visits_source": cap["source"],
        "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
    }


@app.get("/api/nodes")
def nodes(
    graph: str = Query(default=DEFAULT_GRAPH_NAME),
    q: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=100),
):
    bundle = get_loaded_graph_bundle(graph)
    return {"nodes": search_nodes(bundle, q, limit)}


@app.get("/api/subgraph")
def subgraph(
    graph: str = Query(default=DEFAULT_GRAPH_NAME),
    center: str | None = Query(default=None),
    source: str | None = Query(default=None),
    target: str | None = Query(default=None),
    depth: int = Query(default=1, ge=0, le=2),
    limit: int = Query(default=SUBGRAPH_LIMIT, ge=20, le=600),
):
    bundle = get_loaded_graph_bundle(graph)

    if not center and not source and not target:
        cached_overview = load_overview_visual_cache(bundle.name, bundle.path, limit)
        if cached_overview is not None:
            return cached_overview

    center_node = canonical_node(bundle, center) if center else None
    source_node = canonical_node(bundle, source) if source else None
    target_node = canonical_node(bundle, target) if target else None

    if center and center_node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{center}' is not available in {bundle.label}.",
        )
    if source and source_node is None:
        raise HTTPException(
            status_code=404,
            detail=missing_node_message(bundle, source, "start"),
        )
    if target and target_node is None:
        raise HTTPException(
            status_code=404,
            detail=missing_node_message(bundle, target, "target"),
        )

    selected = (
        collect_neighborhood(bundle.graph, [center_node], depth, limit)
        if center_node
        else sample_overview_nodes(bundle.graph, limit)
    )
    if source_node:
        selected.add(source_node)
    if target_node:
        selected.add(target_node)

    visual_graph = build_visual_graph(
        bundle.graph,
        selected,
        source=source_node,
        target=target_node,
        edge_limit=preview_edge_budget(bundle.graph, len(selected))
        if not center_node and not source_node and not target_node
        else None,
        meta=visual_graph_meta(bundle, "live", limit),
    )
    if not center_node and not source_node and not target_node:
        save_overview_visual_cache(bundle.name, bundle.path, limit, visual_graph)

    return visual_graph


@app.post("/api/astar")
def astar(request: AStarRequest):
    bundle = get_loaded_graph_bundle(request.graph)
    model_option = get_model_option(request.model)

    if model_option is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or unavailable model '{request.model}'.",
        )

    if request.dim not in model_option["dims"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dimension {request.dim} is not available for "
                f"{model_option['label']}."
            ),
        )

    source = canonical_node(bundle, request.source)
    target = canonical_node(bundle, request.target)

    if source is None:
        raise HTTPException(
            status_code=400,
            detail=missing_node_message(bundle, request.source, "start"),
        )
    if target is None:
        raise HTTPException(
            status_code=400,
            detail=missing_node_message(bundle, request.target, "target"),
        )

    config = build_astar_runtime_config(bundle.name, model_option, request)
    runtime = get_model_runtime(
        model_option["id"],
        request.dim,
        get_embedding_cache_suffix(bundle.name),
        get_node_universe_for_graph(bundle.name),
    )
    with runtime.lock:
        try:
            runtime.embedder.set_matryoshka_dim(request.dim)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not set embedding dimension {request.dim}: {exc}",
            ) from exc

        indexed_graph = runtime.indexed_graphs.get(bundle.name)
        if indexed_graph is not None:
            config["_indexed_graph"] = indexed_graph

        started = time.perf_counter()
        try:
            import traverse_strategies as ts

            # This is the single integration point with the existing inference
            # code. The runtime lock protects STEmbedder's mutable active
            # matryoshka dimension while preserving one loaded model per
            # model/cache suffix.
            path, visited_nodes = traverse_graph(
                bundle.graph,
                source,
                target,
                runtime.embedder,
                ts.astar_traverse,
                config,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"A* inference failed: {exc}",
            ) from exc

        runtime_ms = (time.perf_counter() - started) * 1000.0
    path_edges = path_to_edges(path)
    selected = collect_result_nodes(bundle.graph, path, source, target)

    return {
        "found": bool(path),
        "path": path,
        "path_edges": path_edges,
        "hops": max(len(path) - 1, 0) if path else 0,
        "visited_nodes": visited_nodes,
        "runtime_ms": round(runtime_ms, 2),
        "source": source,
        "target": target,
        "used_config": public_astar_config(config),
        "graph": build_visual_graph(
            bundle.graph,
            selected,
            source=source,
            target=target,
            path_edges=path_edges,
        ),
    }


def resolve_code_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_available_demo_graphs() -> tuple[str, ...]:
    return tuple(
        graph_name
        for graph_name in DEMO_GRAPH_CHOICES
        if resolve_code_path(GRAPH_CONFIGS[graph_name]["path"]).exists()
    )


def validate_demo_graph_name(graph_name: str) -> None:
    if graph_name not in DEMO_GRAPH_CHOICES:
        choices = ", ".join(DEMO_GRAPH_CHOICES)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Graph '{graph_name}' is not enabled for this web demo. "
                f"Enabled graphs: {choices}."
            ),
        )


def preload_inference_modules() -> None:
    started = time.perf_counter()

    # Import the traversal package during startup so the first UI click does not
    # pay Python import/module-initialization cost inside the timed A* endpoint.
    import traverse_strategies  # noqa: F401

    elapsed = time.perf_counter() - started
    _model_preload_status["inference_modules"] = {
        "loaded": True,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(
        "Preloaded A* inference modules in "
        f"{elapsed:.2f}s.",
        flush=True,
    )


def preload_demo_graphs() -> None:
    global _preload_complete

    available_graphs = get_available_demo_graphs()
    if not available_graphs:
        raise RuntimeError("No configured web-demo graph files are available.")

    started = time.perf_counter()
    print(
        "Preloading web-demo graphs before serving UI: "
        f"{', '.join(available_graphs)}",
        flush=True,
    )

    for graph_name in available_graphs:
        graph_started = time.perf_counter()
        _preload_status[graph_name] = {
            "loaded": False,
            "path": str(resolve_code_path(GRAPH_CONFIGS[graph_name]["path"])),
        }
        bundle = get_graph_bundle(graph_name)
        elapsed = time.perf_counter() - graph_started
        _preload_status[graph_name] = {
            "loaded": True,
            "path": str(bundle.path),
            "nodes": bundle.graph.number_of_nodes(),
            "edges": bundle.graph.number_of_edges(),
            "elapsed_seconds": round(elapsed, 2),
        }
        print(
            "Preloaded web-demo graph: "
            f"{graph_name} with {bundle.graph.number_of_nodes():,} nodes and "
            f"{bundle.graph.number_of_edges():,} edges in {elapsed:.2f}s.",
            flush=True,
        )

    _preload_complete = True
    print(
        "Finished web-demo graph preload in "
        f"{time.perf_counter() - started:.2f}s.",
        flush=True,
    )


def preload_demo_model_runtimes() -> None:
    if not PRELOAD_MODEL_RUNTIMES:
        _model_preload_status.update(
            {
                "loaded": False,
                "skipped": True,
                "reason": "WEB_DEMO_PRELOAD_MODELS is disabled",
            }
        )
        return

    models = get_model_options_for_preload(discover_models())
    if not models:
        _model_preload_status.update(
            {
                "loaded": False,
                "skipped": True,
                "reason": "no available embedding models",
            }
        )
        print("Skipping web-demo model warmup: no available models.", flush=True)
        return

    graph_names = get_available_demo_graphs()
    started = time.perf_counter()
    _model_preload_status.update(
        {
            "loaded": False,
            "skipped": False,
            "setting": PRELOAD_MODELS_SETTING or "all",
            "model_count": len(models),
            "models": [],
        }
    )

    print(
        "Preloading A* models before serving UI: "
        f"{len(models)} model(s) for {', '.join(graph_names)}.",
        flush=True,
    )

    preload_astar_config_defaults(models, graph_names)

    loaded_models = []
    failed_models = []
    for model_index, model_option in enumerate(models, start=1):
        dim = model_option["dims"][0]
        model_started = time.perf_counter()
        model_status = {
            "id": model_option["id"],
            "label": model_option["label"],
            "dim": dim,
            "warmed_dims": model_option["dims"],
            "loaded": False,
        }
        _model_preload_status["models"].append(model_status)

        print(
            "Preloading A* model "
            f"{model_index}/{len(models)}: {model_option['label']} "
            f"(full dim {dim}; warms dims {model_option['dims']}).",
            flush=True,
        )

        try:
            runtime = get_model_runtime(
                model_option["id"],
                dim,
                get_embedding_cache_suffix(DEFAULT_GRAPH_NAME),
                get_node_universe_for_graph(DEFAULT_GRAPH_NAME),
            )
            prepare_runtime_indexes(runtime, graph_names)
            warm_runtime_dimensions(runtime, model_option["dims"])
            warm_runtime_traversals(runtime, graph_names, model_option["dims"])
        except Exception as exc:
            model_status.update(
                {
                    "loaded": False,
                    "error": exception_message(exc),
                    "elapsed_seconds": round(time.perf_counter() - model_started, 2),
                }
            )
            failed_models.append(model_status)
            print(
                "A* model warmup failed; selecting this model may still pay "
                f"lazy load cost: {model_option['label']}: {exception_message(exc)}",
                flush=True,
            )
            continue

        model_status.update(
            {
                "loaded": True,
                "indexed_graphs": sorted(runtime.indexed_graphs),
                "dimension_warmup": True,
                "traversal_warmup": True,
                "elapsed_seconds": round(time.perf_counter() - model_started, 2),
            }
        )
        loaded_models.append(model_status)
        print(
            "Finished A* model warmup: "
            f"{model_option['label']} in {model_status['elapsed_seconds']:.2f}s.",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    _model_preload_status.update(
        {
            "loaded": bool(loaded_models) and not failed_models,
            "complete": True,
            "loaded_count": len(loaded_models),
            "failed_count": len(failed_models),
            "elapsed_seconds": round(elapsed, 2),
        }
    )
    print(
        "Finished A* model warmup in "
        f"{elapsed:.2f}s: {len(loaded_models)} loaded, "
        f"{len(failed_models)} failed.",
        flush=True,
    )


def get_model_options_for_preload(
    models: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    setting = PRELOAD_MODELS_SETTING.strip()
    if not setting or setting.lower() in {"all", "*"}:
        return models

    requested = {
        item.strip()
        for item in setting.split(",")
        if item.strip()
    }
    selected = []
    for model in models:
        aliases = {
            model["id"],
            model["cache_name"],
            model["label"],
            model["base_label"],
            Path(model["id"]).name,
        }
        if aliases & requested:
            selected.append(model)

    missing = sorted(requested - {
        alias
        for model in selected
        for alias in {
            model["id"],
            model["cache_name"],
            model["label"],
            model["base_label"],
            Path(model["id"]).name,
        }
    })
    if missing:
        print(
            "WEB_DEMO_PRELOAD_MODELS ignored unknown model selector(s): "
            f"{', '.join(missing)}",
            flush=True,
        )

    return tuple(selected)


def exception_message(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if detail:
        return str(detail)
    return str(exc) or exc.__class__.__name__


def preload_astar_config_defaults(
    models: tuple[dict[str, Any], ...],
    graph_names: tuple[str, ...],
) -> None:
    started = time.perf_counter()
    warmed = 0
    print(
        "Preloading A* config defaults for web-demo model/dim choices.",
        flush=True,
    )

    for graph_name in graph_names:
        for model in models:
            for dim in model["dims"]:
                get_default_astar_max_visits(graph_name, model["id"], dim)
                warmed += 1

    elapsed = time.perf_counter() - started
    _model_preload_status["config_defaults"] = {
        "loaded": True,
        "warmed_count": warmed,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(
        "Finished A* config default warmup: "
        f"{warmed} graph/model/dim combinations in {elapsed:.2f}s.",
        flush=True,
    )


def prepare_runtime_indexes(
    runtime: ModelRuntime,
    graph_names: tuple[str, ...],
) -> None:
    missing_graph_names = [
        graph_name
        for graph_name in graph_names
        if graph_name not in runtime.indexed_graphs
    ]
    if not missing_graph_names:
        return

    ordered_nodes = []
    seen_nodes = set()
    for graph_name in missing_graph_names:
        bundle = get_loaded_graph_bundle(graph_name)
        for node in bundle.nodes:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            ordered_nodes.append(node)

    print(
        "Preparing A* model embedding index for web-demo graphs: "
        f"{len(ordered_nodes):,} unique nodes across "
        f"{', '.join(missing_graph_names)}.",
        flush=True,
    )

    runtime.embedder.prepare_embedding_index(
        ordered_nodes,
        batch_size=MODEL_WARMUP_BATCH_SIZE,
        save=True,
        discard_tensor_cache=True,
        populate_tensor_cache=False,
        texts_are_unique=True,
    )

    from core.indexed_graph import build_indexed_graph

    for graph_name in missing_graph_names:
        bundle = get_loaded_graph_bundle(graph_name)
        runtime.indexed_graphs[graph_name] = build_indexed_graph(
            bundle.graph,
            runtime.embedder.indexed_text_to_idx,
        )


def warm_runtime_dimensions(runtime: ModelRuntime, dims: list[int]) -> None:
    if runtime.embedder.embedding_table is None:
        return

    for dim in dims:
        runtime.embedder.set_matryoshka_dim(dim)
        # Touch one indexed row per dimension. Lower dims are slices of the
        # full-dimension runtime table, so this is cheap but removes the first
        # request's lazy tensor/indexing setup.
        runtime.embedder.embed_index(0)

    runtime.embedder.set_matryoshka_dim(dims[0])


def warm_runtime_traversals(
    runtime: ModelRuntime,
    graph_names: tuple[str, ...],
    dims: list[int],
) -> None:
    import traverse_strategies as ts

    for graph_name in graph_names:
        bundle = get_loaded_graph_bundle(graph_name)
        warmup_edge = get_graph_warmup_edge(bundle)
        if warmup_edge is None:
            continue

        source, target = warmup_edge
        config = {
            "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
            "astar_max_visits": 10,
        }
        indexed_graph = runtime.indexed_graphs.get(graph_name)
        if indexed_graph is not None:
            config["_indexed_graph"] = indexed_graph

        for dim in dims:
            runtime.embedder.set_matryoshka_dim(dim)
            traverse_graph(
                bundle.graph,
                source,
                target,
                runtime.embedder,
                ts.astar_traverse,
                config,
            )

    runtime.embedder.set_matryoshka_dim(dims[0])


def get_graph_warmup_edge(bundle: GraphBundle) -> tuple[str, str] | None:
    cached = _graph_warmup_edges.get(bundle.name)
    if cached is not None:
        return cached

    best_edge = None
    best_degree = None
    for source in bundle.nodes:
        out_degree = bundle.graph.out_degree(source)
        if out_degree <= 0:
            continue

        try:
            target = next(iter(bundle.graph.successors(source)))
        except StopIteration:
            continue

        if best_degree is None or out_degree < best_degree:
            best_edge = (source, target)
            best_degree = out_degree
            if out_degree == 1:
                break

    if best_edge is not None:
        _graph_warmup_edges[bundle.name] = best_edge

    return best_edge


def get_loaded_graph_bundle(graph_name: str) -> GraphBundle:
    validate_demo_graph_name(graph_name)
    cached = _graph_cache.get(graph_name)
    if cached is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Graph '{graph_name}' is enabled but not preloaded. "
                "Restart the demo so startup can load graph data before A* runs."
            ),
        )
    return cached


def get_graph_bundle(graph_name: str) -> GraphBundle:
    validate_demo_graph_name(graph_name)

    with _graph_lock:
        cached = _graph_cache.get(graph_name)
        if cached is not None:
            return cached

        config = GRAPH_CONFIGS[graph_name]
        graph_path = resolve_code_path(config["path"])
        if not graph_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Graph file not found: {graph_path}",
            )

        graph = load_graph_for_demo(graph_name, graph_path)
        nodes = sorted(str(node) for node in graph.nodes)
        lowercase_to_node = {}
        for node in nodes:
            lowercase_to_node.setdefault(node.lower(), node)

        bundle = GraphBundle(
            name=graph_name,
            label=config["label"],
            path=graph_path,
            graph=graph,
            nodes=nodes,
            lowercase_to_node=lowercase_to_node,
        )
        _graph_cache[graph_name] = bundle
        return bundle


def load_graph_for_demo(graph_name: str, graph_path: Path):
    graph = load_graph_from_disk_cache(graph_name, graph_path)
    if graph is not None:
        return graph

    start_time = time.perf_counter()
    graph = load_causal_graph(graph_path)
    elapsed = time.perf_counter() - start_time
    print(
        "Loaded graph from source: "
        f"{graph_name} with {graph.number_of_nodes():,} nodes and "
        f"{graph.number_of_edges():,} edges in {elapsed:.2f}s.",
        flush=True,
    )
    save_graph_to_disk_cache(graph_name, graph_path, graph)
    return graph


def graph_cache_path(graph_name: str) -> Path:
    return GRAPH_CACHE_DIR / f"{graph_name}.pickle"


def graph_source_signature(graph_path: Path) -> dict[str, Any]:
    stat = graph_path.stat()
    return {
        "version": GRAPH_CACHE_VERSION,
        "source_path": str(graph_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_graph_from_disk_cache(graph_name: str, graph_path: Path):
    if DISABLE_GRAPH_DISK_CACHE:
        return None

    cache_path = graph_cache_path(graph_name)
    if not cache_path.exists():
        return None

    expected_signature = graph_source_signature(graph_path)
    start_time = time.perf_counter()

    try:
        with open(cache_path, "rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
        print(
            f"Ignoring unreadable graph cache {cache_path}: {exc}",
            flush=True,
        )
        return None

    if payload.get("signature") != expected_signature:
        print(f"Ignoring stale graph cache {cache_path}.", flush=True)
        return None

    graph = payload.get("graph")
    if graph is None:
        print(f"Ignoring empty graph cache {cache_path}.", flush=True)
        return None

    elapsed = time.perf_counter() - start_time
    print(
        "Loaded graph from parsed cache: "
        f"{graph_name} with {graph.number_of_nodes():,} nodes and "
        f"{graph.number_of_edges():,} edges in {elapsed:.2f}s.",
        flush=True,
    )
    return graph


def save_graph_to_disk_cache(graph_name: str, graph_path: Path, graph):
    if DISABLE_GRAPH_DISK_CACHE:
        return

    try:
        GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = graph_cache_path(graph_name)
        temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        payload = {
            "signature": graph_source_signature(graph_path),
            "graph": graph,
        }

        with open(temp_path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_path, cache_path)
        print(f"Saved parsed graph cache: {cache_path}", flush=True)
    except OSError as exc:
        print(f"Could not save graph cache for {graph_name}: {exc}", flush=True)


def overview_visual_cache_path(graph_name: str, limit: int) -> Path:
    return VISUAL_CACHE_DIR / f"{graph_name}_preview_v{VISUAL_CACHE_VERSION}_{limit}.json"


def visual_cache_signature(graph_name: str, graph_path: Path, limit: int) -> dict[str, Any]:
    signature = dict(graph_source_signature(graph_path))
    signature.update(
        {
            "visual_cache_version": VISUAL_CACHE_VERSION,
            "graph": graph_name,
            "limit": limit,
            "algorithm": "degree_seeded_neighborhood_v2",
        }
    )
    return signature


def load_overview_visual_cache(graph_name: str, graph_path: Path, limit: int):
    cache_path = overview_visual_cache_path(graph_name, limit)
    if not cache_path.exists() or not graph_path.exists():
        return None

    try:
        with open(cache_path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable visual cache {cache_path}: {exc}", flush=True)
        return None

    if payload.get("signature") != visual_cache_signature(graph_name, graph_path, limit):
        print(f"Ignoring stale visual cache {cache_path}.", flush=True)
        return None

    graph_payload = payload.get("graph")
    if not graph_payload:
        return None

    graph_payload.setdefault("meta", {})
    graph_payload["meta"].update(
        {
            "graph": graph_name,
            "label": get_graph_label(graph_name),
            "cache_status": "disk",
            "limit": limit,
        }
    )
    return graph_payload


def save_overview_visual_cache(
    graph_name: str,
    graph_path: Path,
    limit: int,
    visual_graph: dict[str, Any],
):
    try:
        VISUAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = overview_visual_cache_path(graph_name, limit)
        temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        payload = {
            "signature": visual_cache_signature(graph_name, graph_path, limit),
            "graph": visual_graph,
        }

        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)

        os.replace(temp_path, cache_path)
        print(f"Saved visual overview cache: {cache_path}", flush=True)
    except OSError as exc:
        print(f"Could not save visual overview cache for {graph_name}: {exc}", flush=True)


@lru_cache(maxsize=1)
def discover_models() -> tuple[dict[str, Any], ...]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_model(model_path: str, is_finetuned: bool = False):
        normalized_path = model_path.replace("\\", "/")
        if normalized_path in seen:
            return

        dim = infer_model_dim(normalized_path)
        if dim is None:
            return

        if not is_finetuned and not embedding_cache_exists(normalized_path):
            return

        distance_metric = get_model_distance_metric(normalized_path)
        distance = distance_metric.name.lower()
        dims = get_matryoshka_dims(dim)
        cache_name = Path(normalized_path).name
        config_labels = get_model_config_labels(cache_name)
        label = format_model_display_name(cache_name)

        models.append(
            {
                "id": normalized_path,
                "label": label,
                "base_label": format_model_display_name(
                    cache_name,
                    include_config=False,
                ),
                "config_label": " + ".join(config_labels),
                "model_dim": dim,
                "dims": dims,
                "distance": distance,
                "distance_label": distance_metric.name.title(),
                "is_finetuned": is_finetuned,
                "cache_name": cache_name,
            }
        )
        seen.add(normalized_path)

    for model_path in BASE_MODELS:
        add_model(model_path)

    for model_path in get_fine_tuned_models(DEFAULT_RUN_SUFFIX):
        add_model(model_path, is_finetuned=True)

    if LIGHTNING_MODELS_DIR.exists():
        for model_dir in sorted(LIGHTNING_MODELS_DIR.iterdir()):
            if model_dir.is_dir() and model_dir.name != "old":
                add_model(
                    str(model_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                    is_finetuned=True,
                )

    return tuple(models)


def get_model_option(model_id: str) -> dict[str, Any] | None:
    return next((model for model in discover_models() if model["id"] == model_id), None)


def get_model_runtime(
    model_id: str,
    dim: int,
    cache_suffix: str | None,
    node_universe: str,
) -> ModelRuntime:
    key = (model_id, cache_suffix, node_universe)

    with _model_lock:
        runtime = _model_cache.get(key)
        if runtime is not None:
            if dim > runtime.embedder.get_model_dim():
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Requested dim {dim}, but model has "
                        f"{runtime.embedder.get_model_dim()} dimensions."
                    ),
                )
            return runtime

        try:
            from core.embeddings import STEmbedder

            distance_metric = get_model_distance_metric(model_id)
            embedder = STEmbedder(
                model_id,
                distance_metric,
                device=EMBEDDING_DEVICE,
                cache_suffix=cache_suffix,
                node_universe=node_universe,
            )
            if dim > embedder.get_model_dim():
                raise ValueError(
                    f"Requested dim {dim}, but model has "
                    f"{embedder.get_model_dim()} dimensions."
                )
            embedder.set_matryoshka_dim(dim)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load model '{model_id}': {exc}",
            ) from exc

        runtime = ModelRuntime(model_id, cache_suffix, node_universe, embedder, {})
        _model_cache[key] = runtime
        return runtime


def build_astar_runtime_config(
    graph_name: str,
    model_option: dict[str, Any],
    request: AStarRequest,
) -> dict[str, int]:
    config = {
        "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
    }

    if request.config.astar_max_visits is None:
        max_visits = get_default_astar_max_visits(
            graph_name,
            model_option["id"],
            request.dim,
        )["value"]
    else:
        max_visits = request.config.astar_max_visits

    if max_visits < -1:
        raise HTTPException(
            status_code=400,
            detail="A* max visits must be -1 for uncapped or a non-negative integer.",
        )
    config["astar_max_visits"] = max_visits

    threshold = request.config.embedding_index_min_successors
    if threshold is None:
        threshold = EMBEDDING_INDEX_MIN_SUCCESSORS
    if threshold < 1:
        raise HTTPException(
            status_code=400,
            detail="Embedding-index successor threshold must be at least 1.",
        )
    config["embedding_index_min_successors"] = threshold

    return config


def public_astar_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }


def get_default_astar_max_visits(
    graph_name: str,
    model_id: str,
    dim: int,
) -> dict[str, Any]:
    if ASTAR_MAX_VISITS != -1:
        return {
            "value": ASTAR_MAX_VISITS,
            "source": "WEB_DEMO_ASTAR_MAX_VISITS",
        }

    model_name = Path(model_id).name
    candidate_graphs = [graph_name]
    if DEFAULT_GRAPH_NAME not in candidate_graphs:
        candidate_graphs.append(DEFAULT_GRAPH_NAME)

    for candidate_graph in candidate_graphs:
        p95_file = (
            EVALUATION_DIR
            / candidate_graph
            / "msmarco_train"
            / DEFAULT_RUN_SUFFIX
            / "visited_nodes_analysis.json"
        )
        cap = read_p95_astar_cap(p95_file, model_name, dim)
        if cap is not None:
            return {
                "value": cap,
                "source": (
                    f"{candidate_graph}/msmarco_train/{DEFAULT_RUN_SUFFIX} "
                    "p95 successful A* visits"
                ),
            }

    return {
        "value": -1,
        "source": "uncapped; no matching p95 analysis found",
    }


def read_p95_astar_cap(
    p95_file: Path,
    model_name: str,
    dim: int,
) -> int | None:
    return read_p95_astar_cap_index(p95_file).get((model_name, dim))


@lru_cache(maxsize=16)
def read_p95_astar_cap_index(p95_file: Path) -> dict[tuple[str, int], int]:
    if not p95_file.exists():
        return {}

    try:
        with open(p95_file, encoding="utf-8") as file:
            entries = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    caps = {}
    for entry in entries:
        analysis = entry.get("analysis", {})
        if analysis.get("strategy") != "A*":
            continue
        p95_value = analysis.get("p95_visited_successful_only")
        if p95_value is None:
            continue

        try:
            dimension = int(entry.get("dimension"))
        except (TypeError, ValueError):
            continue

        model_name = entry.get("model")
        if not model_name:
            continue

        caps[(str(model_name), dimension)] = int(math.ceil(p95_value))

    return caps


def embedding_cache_exists(model_path: str) -> bool:
    cache_name = f"{Path(model_path).name}_embeddings"
    node_file = get_node_universe_path(
        EMBEDDINGS_DIR,
        get_node_universe_for_graph(DEFAULT_GRAPH_NAME),
    )
    return (
        node_file.exists()
        and (EMBEDDINGS_DIR / f"{cache_name}_vectors.npy").exists()
    )


def infer_model_dim(model_path: str) -> int | None:
    if model_path in MODEL_DIM_HINTS:
        return MODEL_DIM_HINTS[model_path]

    local_path = resolve_code_path(model_path)
    config_path = local_path / "config.json"
    if config_path.exists():
        try:
            import json

            with open(config_path, encoding="utf-8") as file:
                config = json.load(file)
            hidden_size = config.get("hidden_size")
            return int(hidden_size) if hidden_size else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    return None


def canonical_node(bundle: GraphBundle, value: str | None) -> str | None:
    if value is None:
        return None

    candidate = value.strip()
    if not candidate:
        return None
    if candidate in bundle.graph.nodes:
        return candidate
    return bundle.lowercase_to_node.get(candidate.lower())


def search_nodes(bundle: GraphBundle, query: str, limit: int) -> list[str]:
    query = query.strip().lower()
    if not query:
        return bundle.nodes[:limit]

    prefix_matches = []
    contains_matches = []
    for node in bundle.nodes:
        lowered = node.lower()
        if lowered.startswith(query):
            prefix_matches.append(node)
        elif query in lowered:
            contains_matches.append(node)

        if len(prefix_matches) >= limit:
            break

    matches = prefix_matches + contains_matches
    return matches[:limit]


def missing_node_message(bundle: GraphBundle, value: str, role: str) -> str:
    suggestions = search_nodes(bundle, value, 5) if value else []
    message = (
        f"{role.title()} node '{value}' not found in graph "
        f"{bundle.label} ({bundle.name})."
    )
    if suggestions:
        message += " Closest matches: " + ", ".join(suggestions) + "."
    return message


def sample_overview_nodes(graph, limit: int) -> set[str]:
    node_budget = preview_node_budget(graph, limit)
    if graph.number_of_nodes() <= node_budget:
        return set(graph.nodes)

    seed = max(graph.nodes, key=lambda node: graph.degree(node))
    selected = collect_neighborhood(graph, [seed], depth=1, limit=node_budget)
    if len(selected) < node_budget:
        selected = collect_neighborhood(graph, [seed], depth=2, limit=node_budget)
    return selected


def preview_node_budget(graph, limit: int) -> int:
    if graph.number_of_nodes() <= limit:
        return graph.number_of_nodes()

    # Treat the query limit as an upper bound, not a target. The budget grows
    # with graph size, which keeps previews graph-specific instead of forcing
    # every dense graph into the same 240-node shape.
    edge_count = max(graph.number_of_edges(), 1)
    budget = max(96, int(math.sqrt(min(edge_count, 1_000_000)) / 3))
    return min(limit, budget)


def preview_edge_budget(graph, node_count: int) -> int:
    if node_count <= 1:
        return 0

    avg_degree = graph.number_of_edges() / max(graph.number_of_nodes(), 1)
    factor = 2.0 + min(2.5, math.log2(avg_degree + 1) / 2)
    return max(node_count - 1, int(node_count * factor))


def visual_graph_meta(bundle: GraphBundle, cache_status: str, limit: int) -> dict[str, Any]:
    return {
        "graph": bundle.name,
        "label": bundle.label,
        "cache_status": cache_status,
        "limit": limit,
        "source_path": str(bundle.path),
        "graph_nodes": bundle.graph.number_of_nodes(),
        "graph_edges": bundle.graph.number_of_edges(),
    }


def collect_neighborhood(
    graph,
    centers: list[str],
    depth: int,
    limit: int,
) -> set[str]:
    selected = {node for node in centers if node in graph.nodes}
    frontier = set(selected)

    for _ in range(depth):
        if len(selected) >= limit:
            break

        candidates = []
        for node in frontier:
            candidates.extend(graph.successors(node))
            candidates.extend(graph.predecessors(node))

        candidates = [
            node
            for node in dict.fromkeys(candidates)
            if node not in selected
        ]
        candidates.sort(
            key=lambda node: (
                graph.degree(node),
                graph.out_degree(node),
                graph.in_degree(node),
                str(node),
            ),
            reverse=True,
        )

        frontier = set()
        for node in candidates:
            if len(selected) >= limit:
                break
            selected.add(node)
            frontier.add(node)

    return selected


def collect_result_nodes(
    graph,
    path: list[str],
    source: str,
    target: str,
) -> set[str]:
    centers = path if path else [source, target]
    limit = max(SUBGRAPH_LIMIT, len(centers) + 40)
    selected = collect_neighborhood(graph, centers, depth=1, limit=limit)
    selected.update(centers)
    return selected


def path_to_edges(path: list[str]) -> list[dict[str, str]]:
    return [
        {"source": path[index], "target": path[index + 1]}
        for index in range(len(path) - 1)
    ]


def build_visual_graph(
    graph,
    selected_nodes: set[str],
    source: str | None = None,
    target: str | None = None,
    path_edges: list[dict[str, str]] | None = None,
    edge_limit: int | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path_edges = path_edges or []
    path_edge_keys = {
        (edge["source"], edge["target"])
        for edge in path_edges
    }
    path_nodes = {
        node
        for edge in path_edges
        for node in (edge["source"], edge["target"])
    }

    selected_nodes = set(selected_nodes)
    if source:
        selected_nodes.add(source)
    if target:
        selected_nodes.add(target)
    selected_nodes.update(path_nodes)

    nodes = [
        {
            "id": node,
            "label": node,
            "degree": int(graph.degree(node)),
            "status": node_status(node, source, target, path_nodes),
        }
        for node in sorted(selected_nodes)
        if node in graph.nodes
    ]

    if edge_limit is None:
        edge_limit = max(len(nodes) * EDGE_LIMIT_FACTOR, 120)
    links = []
    seen_edges = set()

    for source_node, target_node in path_edge_keys:
        if graph.has_edge(source_node, target_node):
            links.append(edge_payload(graph, source_node, target_node, is_path=True))
            seen_edges.add((source_node, target_node))

    for source_node, target_node in spanning_edges(graph, selected_nodes):
        edge_key = (source_node, target_node)
        if edge_key in seen_edges:
            continue
        links.append(
            edge_payload(
                graph,
                source_node,
                target_node,
                is_path=edge_key in path_edge_keys,
            )
        )
        seen_edges.add(edge_key)

    source_order = sorted(
        selected_nodes,
        key=lambda node: (
            graph.degree(node) if node in graph.nodes else -1,
            str(node),
        ),
        reverse=True,
    )
    for source_node in source_order:
        if source_node not in graph.nodes:
            continue
        for target_node in graph.successors(source_node):
            edge_key = (source_node, target_node)
            if target_node not in selected_nodes or edge_key in seen_edges:
                continue
            links.append(
                edge_payload(
                    graph,
                    source_node,
                    target_node,
                    is_path=edge_key in path_edge_keys,
                )
            )
            seen_edges.add(edge_key)
            if len(links) >= edge_limit:
                break
        if len(links) >= edge_limit:
            break

    payload = {"nodes": nodes, "links": links}
    if meta is not None:
        payload["meta"] = {
            **meta,
            "preview_node_count": len(nodes),
            "preview_edge_count": len(links),
        }
    return payload


def spanning_edges(graph, selected_nodes: set[str]):
    selected_nodes = {node for node in selected_nodes if node in graph.nodes}
    visited = set()

    roots = sorted(
        selected_nodes,
        key=lambda node: (graph.degree(node), str(node)),
        reverse=True,
    )

    for root in roots:
        if root in visited:
            continue

        visited.add(root)
        queue = [root]

        while queue:
            node = queue.pop(0)
            neighbors = []

            for successor in graph.successors(node):
                if successor in selected_nodes:
                    neighbors.append((node, successor, successor))

            for predecessor in graph.predecessors(node):
                if predecessor in selected_nodes:
                    neighbors.append((predecessor, node, predecessor))

            neighbors.sort(
                key=lambda item: (
                    graph.degree(item[2]),
                    graph.out_degree(item[2]),
                    graph.in_degree(item[2]),
                    str(item[2]),
                ),
                reverse=True,
            )

            for source_node, target_node, next_node in neighbors:
                if next_node in visited:
                    continue

                visited.add(next_node)
                queue.append(next_node)
                yield source_node, target_node


def node_status(
    node: str,
    source: str | None,
    target: str | None,
    path_nodes: set[str],
) -> str:
    if node == source:
        return "source"
    if node == target:
        return "target"
    if node in path_nodes:
        return "path"
    return "normal"


def edge_payload(graph, source_node: str, target_node: str, is_path: bool) -> dict[str, Any]:
    data = graph.get_edge_data(source_node, target_node, default={}) or {}
    return {
        "source": source_node,
        "target": target_node,
        "path": is_path,
        "support": data.get("support"),
        "sentence": data.get("sentence"),
    }
