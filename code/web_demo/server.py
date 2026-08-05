from __future__ import annotations

import json
import math
import os
import pickle
import threading
import time
from bisect import bisect_left
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.config import (
    BASE_MODELS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_P95_CONFIG_SOURCE_DATASET,
    DEFAULT_P95_CONFIG_SOURCE_GRAPH,
    DEFAULT_RUN_SUFFIX as CONFIG_DEFAULT_RUN_SUFFIX,
    EMBEDDING_INDEX_MIN_SUCCESSORS,
)
from core.constants import (
    BFS_CAPPED_BASELINE_MODEL,
    BFS_UNCAPPED_BASELINE_MODEL,
    EMBEDDINGS_DIR,
    EVALUATION_DIR,
    LIGHTNING_MODELS_DIR,
    REPO_ROOT,
    WEB_DEMO_GRAPH_CACHE_DIR,
    WEB_DEMO_GRAPH_PREVIEW_CACHE_DIR,
)
from core.graph_config import (
    DEFAULT_INFERENCE_GRAPH,
    GRAPH_CONFIGS,
    SUPPORTED_INFERENCE_GRAPHS,
    canonical_graph_name,
    graph_aliases_for,
    get_graph_bfs_p95_cap,
    get_graph_cache_suffix,
    get_graph_label,
    get_graph_node_universe,
    graph_supports_algorithm,
)
from core.inference_registry import (
    DEFAULT_RL_POLICY_ID,
    get_rl_policy_config,
    graph_supports_rl,
)
from core.model_registry import (
    MODEL_DIMENSIONS,
    distance_config_token,
    format_model_display_label,
    get_embedding_model,
    method_sort_key,
    parse_model_config,
    stable_config_identity,
)
from core.utils import (
    get_ablation_fine_tuned_models,
    get_fine_tuned_models,
    get_matryoshka_dims,
    get_model_distance_metric,
    get_node_universe_path,
    load_causal_graph,
    load_rl_graph,
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
LOAD_ALL_MODELS = os.environ.get("WEB_DEMO_LOAD_ALL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BEST_MODEL_KEY = "granite"
BEST_MODEL_VARIANT = "finetuned"
BEST_MODEL_ACTIVATION = "relu"
BEST_MODEL_DISTANCE = "euclid"
BEST_MODEL_DIMENSION = 32
DEFAULT_GRAPH_NAME = DEFAULT_INFERENCE_GRAPH


def get_demo_graph_choices(load_all: bool | None = None) -> tuple[str, ...]:
    if load_all is None:
        load_all = LOAD_ALL_MODELS
    return SUPPORTED_INFERENCE_GRAPHS if load_all else (DEFAULT_GRAPH_NAME,)


def get_enabled_algorithms(load_all: bool | None = None) -> tuple[str, ...]:
    if load_all is None:
        load_all = LOAD_ALL_MODELS
    return ("bfs", "rl", "astar") if load_all else ("astar",)


DEMO_GRAPH_CHOICES = get_demo_graph_choices()

MODEL_DIM_HINTS = {
    **MODEL_DIMENSIONS,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": 768,
    "BAAI/bge-base-en-v1.5": 768,
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


@dataclass
class RLRuntime:
    policy_config_id: str
    embedder: Any
    graphs: dict[str, Any]
    lock: Any = field(default_factory=threading.RLock)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BFSConfig(StrictModel):
    cap: int | None = None


class RLConfig(StrictModel):
    policy_config_id: str | None = None


class InferenceAStarConfig(StrictModel):
    model_config_id: str | None = None
    model_id: str | None = None
    dimension: int | None = None
    astar_max_visits: int | None = None
    embedding_index_min_successors: int | None = None


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


class InferenceRequest(StrictModel):
    algorithm: str
    graph_id: str | None = None
    graph: str | None = None
    source: str
    target: str
    config: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    preload_inference_modules()
    preload_demo_graphs()
    preload_demo_model_runtimes()
    preload_demo_rl_runtime()
    yield


app = FastAPI(
    title="Causal Graph Inference Demo",
    description="Interactive Webis-style demo for causal-graph inference.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Full mode exposes all evaluation graph variants; limited mode exposes only
# CauseNet Precision through DEMO_GRAPH_CHOICES.
# Startup deliberately loads and prepares every graph, graph preview, A* runtime,
# and RL runtime. A slow startup is preferable to a UI that occasionally blocks
# on a first graph/model request. Parsed graph and preview caches still make
# subsequent starts faster.
_graph_cache: dict[str, GraphBundle] = {}
_graph_lock = threading.Lock()
_graph_warmup_queries: dict[str, tuple[str, str]] = {}
_overview_visual_cache: dict[tuple[str, int], dict[str, Any]] = {}
_preload_status: dict[str, dict[str, Any]] = {}
_preload_complete = False
_model_cache: dict[tuple[str, str | None, str], ModelRuntime] = {}
_model_lock = threading.Lock()
_rl_runtime_cache: dict[str, RLRuntime] = {}
_rl_runtime_lock = threading.Lock()
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
    models = get_demo_models(discover_models())
    methods = discover_search_methods(models)
    available_graphs = get_available_demo_graphs()
    enabled_algorithms = {method["algorithm"] for method in methods}
    graphs = [
        graph_option_payload(
            graph_name,
            enabled_algorithms=enabled_algorithms,
        )
        for graph_name in available_graphs
    ]
    default_graph = (
        DEFAULT_GRAPH_NAME
        if DEFAULT_GRAPH_NAME in available_graphs
        else available_graphs[0] if available_graphs else None
    )
    default_method = next(
        (
            method["id"]
            for method in methods
            if default_graph in method.get("supported_graphs", ())
        ),
        methods[0]["id"] if methods else None,
    )

    return {
        "graphs": graphs,
        "methods": methods,
        "models": models,
        "defaults": {
            "graph": default_graph,
            "method": default_method,
            "model": models[0]["id"] if models else None,
            "dim": models[0]["dims"][0] if models else None,
        },
        "advanced": {
            "run_suffix": DEFAULT_RUN_SUFFIX,
            "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
            "bfs_cap_source_dataset": DEFAULT_P95_CONFIG_SOURCE_DATASET,
            "bfs_cap_source_graph": DEFAULT_P95_CONFIG_SOURCE_GRAPH,
            "load_all": LOAD_ALL_MODELS,
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
                "nodes": GRAPH_CONFIGS[graph_name]["nodes"],
                "edges": GRAPH_CONFIGS[graph_name]["edges"],
                **_preload_status.get(graph_name, {"loaded": False}),
            }
            for graph_name in DEMO_GRAPH_CHOICES
        ],
        "models": _model_preload_status,
    }


@app.get("/api/config")
def config_defaults(
    graph: str = Query(default=DEFAULT_GRAPH_NAME),
    algorithm: str = Query(default="astar"),
    method: str | None = Query(default=None),
    model: str | None = Query(default=None),
    dim: int | None = Query(default=None, ge=1),
):
    graph = validate_demo_graph_name(graph)
    algorithm = normalize_algorithm(algorithm)

    if algorithm == "bfs":
        cap = get_default_bfs_cap(graph)
        return {
            "algorithm": "bfs",
            "bfs_cap": cap["value"],
            "bfs_cap_source": cap["source"],
            "bfs_cap_mode": "default_p95",
        }

    if algorithm == "rl":
        policy = get_rl_policy_config()
        if not graph_supports_rl(graph, policy.id):
            raise HTTPException(
                status_code=400,
                detail=unsupported_rl_graph_message(graph, policy.id),
            )
        return {
            "algorithm": "rl",
            "policy": policy.public_config(),
        }

    model_option = resolve_astar_model_option(method, model, dim)

    cap = get_default_astar_max_visits(
        graph,
        model_option["id"],
        model_option["selected_dim"],
    )
    return {
        "algorithm": "astar",
        "model_config_id": model_option["config_id"],
        "astar_max_visits": cap["value"],
        "astar_max_visits_source": cap["source"],
        "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
        "model": model_option,
    }


@app.get("/api/nodes")
def nodes(
    graph: str = Query(default=DEFAULT_GRAPH_NAME),
    q: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=100),
):
    graph = validate_demo_graph_name(graph)
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
    graph = validate_demo_graph_name(graph)
    bundle = get_loaded_graph_bundle(graph)

    if not center and not source and not target:
        return get_overview_visual_graph(bundle, limit)

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
    return visual_graph


@app.post("/api/infer")
def infer(request: InferenceRequest):
    return run_inference(request)


@app.post("/api/astar")
def astar(request: AStarRequest):
    return run_inference(
        InferenceRequest(
            algorithm="astar",
            graph_id=request.graph,
            source=request.source,
            target=request.target,
            config={
                "model_id": request.model,
                "dimension": request.dim,
                "astar_max_visits": request.config.astar_max_visits,
                "embedding_index_min_successors": (
                    request.config.embedding_index_min_successors
                ),
            },
        )
    )


def run_inference(request: InferenceRequest) -> dict[str, Any]:
    graph_id = request.graph_id or request.graph
    if not graph_id:
        raise HTTPException(status_code=400, detail="Missing graph_id.")

    graph_id = validate_demo_graph_name(graph_id)
    algorithm = normalize_algorithm(request.algorithm)
    if algorithm not in get_enabled_algorithms():
        enabled = ", ".join(get_enabled_algorithms())
        raise HTTPException(
            status_code=400,
            detail=(
                f"{algorithm.upper()} is disabled in this web-demo mode. "
                f"Enabled algorithms: {enabled}."
            ),
        )
    if not graph_supports_algorithm(graph_id, algorithm):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{algorithm.upper()} is not supported for "
                f"{get_graph_label(graph_id)}."
            ),
        )

    bundle = get_loaded_graph_bundle(graph_id)
    source, target = resolve_endpoint_nodes(bundle, request.source, request.target)

    if algorithm == "bfs":
        return run_bfs_inference(bundle, source, target, request.config)
    if algorithm == "rl":
        return run_rl_inference(bundle, source, target, request.config)
    if algorithm == "astar":
        return run_astar_inference(bundle, source, target, request.config)

    raise HTTPException(status_code=400, detail=f"Unsupported algorithm '{algorithm}'.")


def run_bfs_inference(
    bundle: GraphBundle,
    source: str,
    target: str,
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    config_request = parse_algorithm_config(BFSConfig, raw_config)
    cap = config_request.cap
    cap_source = None
    if cap is None:
        default_cap = get_default_bfs_cap(bundle.name)
        cap = default_cap["value"]
        cap_source = default_cap["source"]

    validate_search_cap(cap, "BFS search cap")

    runtime_config = {"bfs_max_visits": cap}
    started = time.perf_counter()
    try:
        import traverse_strategies as ts

        path, visited_nodes = traverse_graph(
            bundle.graph,
            source,
            target,
            None,
            ts.bfs_traverse,
            runtime_config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BFS inference failed: {exc}") from exc

    runtime_ms = (time.perf_counter() - started) * 1000.0
    cap_mode = bfs_cap_mode(bundle.name, cap)
    return build_inference_response(
        bundle=bundle,
        algorithm="bfs",
        source=source,
        target=target,
        path=path,
        visited_nodes=visited_nodes,
        runtime_ms=runtime_ms,
        config_id="bfs",
        config_label=bfs_result_label(cap, cap_mode),
        used_config={
            **runtime_config,
            "cap_mode": cap_mode,
            "cap_source": cap_source,
        },
        applied_cap=cap,
        termination_reason=termination_reason(
            "bfs",
            bool(path),
            visited_nodes,
            cap,
        ),
    )


def run_rl_inference(
    bundle: GraphBundle,
    source: str,
    target: str,
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    config_request = parse_algorithm_config(RLConfig, raw_config)
    policy = get_rl_policy_config(config_request.policy_config_id)
    if not graph_supports_rl(bundle.name, policy.id):
        raise HTTPException(
            status_code=400,
            detail=unsupported_rl_graph_message(bundle.name, policy.id),
        )

    runtime = get_rl_runtime(policy.id)
    rl_graph = get_rl_graph(runtime, bundle)
    runtime_config = policy.runtime_config()
    runtime_config["question"] = f"can {source} cause {target}?"

    with runtime.lock:
        started = time.perf_counter()
        try:
            import traverse_strategies as ts

            path, visited_nodes = traverse_graph(
                rl_graph,
                source,
                target,
                runtime.embedder,
                ts.rl_traverse,
                runtime_config,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"RL inference failed: {exc}",
            ) from exc
        runtime_ms = (time.perf_counter() - started) * 1000.0

    public_config = policy.public_config()
    return build_inference_response(
        bundle=bundle,
        algorithm="rl",
        source=source,
        target=target,
        path=path,
        visited_nodes=visited_nodes,
        runtime_ms=runtime_ms,
        config_id=policy.id,
        config_label=policy.label,
        used_config=public_config,
        applied_cap=public_config["rl_max_visits"],
        termination_reason=termination_reason(
            "rl",
            bool(path),
            visited_nodes,
            public_config["rl_max_visits"],
        ),
    )


def run_astar_inference(
    bundle: GraphBundle,
    source: str,
    target: str,
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    config_request = parse_algorithm_config(InferenceAStarConfig, raw_config)
    model_option = resolve_astar_model_option(
        config_request.model_config_id,
        config_request.model_id,
        config_request.dimension,
    )

    runtime_config = build_astar_runtime_config(
        bundle.name,
        model_option,
        config_request,
    )
    runtime = get_model_runtime(
        model_option["id"],
        model_option["selected_dim"],
        get_graph_cache_suffix(bundle.name),
        get_graph_node_universe(bundle.name),
    )
    with runtime.lock:
        try:
            runtime.embedder.set_matryoshka_dim(model_option["selected_dim"])
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not set embedding dimension "
                    f"{model_option['selected_dim']}: {exc}"
                ),
            ) from exc

        indexed_graph = runtime.indexed_graphs.get(bundle.name)
        if indexed_graph is not None:
            runtime_config["_indexed_graph"] = indexed_graph

        started = time.perf_counter()
        try:
            import traverse_strategies as ts

            path, visited_nodes = traverse_graph(
                bundle.graph,
                source,
                target,
                runtime.embedder,
                ts.astar_traverse,
                runtime_config,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"A* inference failed: {exc}",
            ) from exc

        runtime_ms = (time.perf_counter() - started) * 1000.0

    public_config = public_astar_config(runtime_config)
    public_config.update(
        {
            "model_config_id": model_option["config_id"],
            "model_id": model_option["id"],
            "dimension": model_option["selected_dim"],
            "label": model_option["selected_label"],
        }
    )
    return build_inference_response(
        bundle=bundle,
        algorithm="astar",
        source=source,
        target=target,
        path=path,
        visited_nodes=visited_nodes,
        runtime_ms=runtime_ms,
        config_id=model_option["config_id"],
        config_label=model_option["selected_label"],
        used_config=public_config,
        applied_cap=public_config.get("astar_max_visits"),
        termination_reason=termination_reason(
            "astar",
            bool(path),
            visited_nodes,
            public_config.get("astar_max_visits"),
        ),
    )


def build_inference_response(
    *,
    bundle: GraphBundle,
    algorithm: str,
    source: str,
    target: str,
    path: list[str],
    visited_nodes: int,
    runtime_ms: float,
    config_id: str,
    config_label: str,
    used_config: dict[str, Any],
    applied_cap: int | None,
    termination_reason: str,
) -> dict[str, Any]:
    path_edges = path_to_edges(path)
    selected = collect_result_nodes(bundle.graph, path, source, target)
    return {
        "algorithm": algorithm,
        "graph_id": bundle.name,
        "graph_label": bundle.label,
        "found": bool(path),
        "path": path,
        "path_edges": path_edges,
        "hops": max(len(path) - 1, 0) if path else 0,
        "visited_nodes": visited_nodes,
        "runtime_ms": round(runtime_ms, 2),
        "source": source,
        "target": target,
        "termination_reason": termination_reason,
        "applied_cap": applied_cap,
        "search_budget": applied_cap,
        "config_id": config_id,
        "config_label": config_label,
        "used_config": used_config,
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


def normalize_algorithm(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"a*", "a-star", "astar"}:
        return "astar"
    if normalized in {"bfs", "rl"}:
        return normalized
    raise HTTPException(
        status_code=400,
        detail="Algorithm must be one of: bfs, rl, astar.",
    )


def parse_algorithm_config(model_type, raw_config: dict[str, Any]):
    try:
        return model_type.model_validate(raw_config or {})
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc


def graph_option_payload(
    graph_name: str,
    *,
    enabled_algorithms: set[str] | None = None,
) -> dict[str, Any]:
    graph_name = canonical_graph_name(graph_name)
    config = GRAPH_CONFIGS[graph_name]
    return {
        "id": graph_name,
        "label": config["label"],
        "nodes": config["nodes"],
        "edges": config["edges"],
        "size_label": format_graph_size(config["nodes"], config["edges"]),
        "bfs_p95_cap": config["bfs_p95_cap"],
        "bfs_p95_cap_source": config["bfs_p95_cap_source"],
        "supported_algorithms": [
            algorithm
            for algorithm in config["supported_algorithms"]
            if enabled_algorithms is None or algorithm in enabled_algorithms
        ],
        "cache": {
            "cache_suffix": config["cache_suffix"],
            "node_universe": config["node_universe"],
        },
    }


def format_graph_size(nodes: int, edges: int) -> str:
    return f"{nodes:,} nodes · {edges:,} edges"


def get_available_demo_graphs() -> tuple[str, ...]:
    return tuple(
        graph_name
        for graph_name in DEMO_GRAPH_CHOICES
        if resolve_code_path(GRAPH_CONFIGS[graph_name]["path"]).exists()
    )


def validate_demo_graph_name(graph_name: str) -> str:
    graph_name = canonical_graph_name(graph_name)
    if graph_name not in DEMO_GRAPH_CHOICES:
        choices = ", ".join(DEMO_GRAPH_CHOICES)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Graph '{graph_name}' is not enabled for this web demo. "
                f"Enabled graphs: {choices}."
            ),
        )
    return graph_name


def resolve_endpoint_nodes(
    bundle: GraphBundle,
    source_value: str,
    target_value: str,
) -> tuple[str, str]:
    source = canonical_node(bundle, source_value)
    target = canonical_node(bundle, target_value)

    if source is None:
        raise HTTPException(
            status_code=400,
            detail=missing_node_message(bundle, source_value, "start"),
        )
    if target is None:
        raise HTTPException(
            status_code=400,
            detail=missing_node_message(bundle, target_value, "target"),
        )

    return source, target


def get_default_bfs_cap(graph_name: str) -> dict[str, Any]:
    graph_name = canonical_graph_name(graph_name)
    candidate_graphs = [graph_name]
    if DEFAULT_P95_CONFIG_SOURCE_GRAPH not in candidate_graphs:
        candidate_graphs.append(DEFAULT_P95_CONFIG_SOURCE_GRAPH)

    for candidate_graph in candidate_graphs:
        p95_files = [
            EVALUATION_DIR
            / graph_dir
            / DEFAULT_P95_CONFIG_SOURCE_DATASET
            / DEFAULT_RUN_SUFFIX
            / "visited_nodes_analysis.json"
            for graph_dir in (candidate_graph, *graph_aliases_for(candidate_graph))
        ]
        p95_file = next((path for path in p95_files if path.exists()), p95_files[0])
        cap = read_p95_bfs_cap(p95_file)
        if cap is not None:
            source_suffix = ""
            if candidate_graph != graph_name:
                source_suffix = " via DEFAULT_P95_CONFIG_SOURCE_GRAPH"
            return {
                "value": cap,
                "source": (
                    f"{candidate_graph}/{DEFAULT_P95_CONFIG_SOURCE_DATASET}/"
                    f"{DEFAULT_RUN_SUFFIX} p95 successful BFS visits"
                    f"{source_suffix}"
                ),
            }

    cap = get_graph_bfs_p95_cap(graph_name)
    source = GRAPH_CONFIGS[graph_name].get("bfs_p95_cap_source")
    if cap is not None:
        return {
            "value": cap,
            "source": source or "central graph registry p95 BFS cap",
        }

    return {
        "value": -1,
        "source": "uncapped; no graph-specific p95 BFS cap configured",
    }


def read_p95_bfs_cap(p95_file: Path) -> int | None:
    return read_p95_bfs_cap_index(p95_file)


@lru_cache(maxsize=16)
def read_p95_bfs_cap_index(p95_file: Path) -> int | None:
    if not p95_file.exists():
        return None

    try:
        with open(p95_file, encoding="utf-8") as file:
            entries = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    for model_name in (BFS_UNCAPPED_BASELINE_MODEL, BFS_CAPPED_BASELINE_MODEL):
        for entry in entries:
            analysis = entry.get("analysis", {})
            if entry.get("model") != model_name or analysis.get("strategy") != "BFS":
                continue
            p95_value = analysis.get("p95_visited_successful_only")
            if p95_value is not None:
                return int(math.ceil(p95_value))

    return None


def validate_search_cap(cap: int, label: str) -> None:
    if cap < -1:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be -1 for uncapped or a non-negative integer.",
        )


def bfs_cap_mode(graph_name: str, cap: int) -> str:
    if cap == -1:
        return "uncapped"
    if cap == get_default_bfs_cap(graph_name)["value"]:
        return "default_p95"
    return "custom"


def bfs_result_label(cap: int, cap_mode: str) -> str:
    if cap_mode == "uncapped":
        return "BFS (uncapped)"
    if cap_mode == "default_p95":
        return "BFS (p95 cap)"
    return "BFS (custom cap)"


def termination_reason(
    algorithm: str,
    found: bool,
    visited_nodes: int,
    cap: int | None,
) -> str:
    if found:
        return "path_found"
    if cap is not None and cap >= 0 and visited_nodes >= cap:
        return "cap_reached"
    if algorithm == "rl":
        return "rl_policy_terminated"
    return "frontier_exhausted"


def unsupported_rl_graph_message(graph_name: str, policy_config_id: str) -> str:
    policy = get_rl_policy_config(policy_config_id)
    supported = ", ".join(get_graph_label(graph) for graph in policy.supported_graphs)
    return (
        f"RL policy '{policy.label}' is not supported for "
        f"{get_graph_label(graph_name)}. Supported graphs: {supported}."
    )


def preload_inference_modules() -> None:
    started = time.perf_counter()

    # Import the traversal package during startup so the first UI click does not
    # pay Python import/module-initialization cost inside the timed endpoint.
    try:
        import traverse_strategies  # noqa: F401
    except ModuleNotFoundError as exc:
        _model_preload_status["inference_modules"] = {
            "loaded": False,
            "error": str(exc),
        }
        print(
            "Could not preload graph inference modules: "
            f"{exc}. Install the full requirements before running A* or RL.",
            flush=True,
        )
        return

    elapsed = time.perf_counter() - started
    _model_preload_status["inference_modules"] = {
        "loaded": True,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(
        "Preloaded graph inference modules in "
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
        "Preloading web-demo graphs and visual previews: "
        f"{', '.join(available_graphs)}",
        flush=True,
    )

    for graph_name in available_graphs:
        graph_started = time.perf_counter()
        try:
            bundle = get_graph_bundle(graph_name)
            preview = get_overview_visual_graph(bundle, SUBGRAPH_LIMIT)
        except Exception as exc:
            _preload_status[graph_name] = {
                "loaded": False,
                "path": str(resolve_code_path(GRAPH_CONFIGS[graph_name]["path"])),
                "nodes": GRAPH_CONFIGS[graph_name]["nodes"],
                "edges": GRAPH_CONFIGS[graph_name]["edges"],
                "error": exception_message(exc),
                "elapsed_seconds": round(time.perf_counter() - graph_started, 2),
            }
            print(
                "Web-demo graph preload failed: "
                f"{graph_name}: {exception_message(exc)}",
                flush=True,
            )
            raise RuntimeError(
                f"Could not preload web-demo graph '{graph_name}'."
            ) from exc

        _preload_status[graph_name].update(
            {
                "loaded": True,
                "preview_loaded": True,
                "preview_nodes": len(preview["nodes"]),
                "preview_edges": len(preview["links"]),
                "elapsed_seconds": round(time.perf_counter() - graph_started, 2),
            }
        )
        print(
            "Finished web-demo graph preload: "
            f"{graph_name} in {_preload_status[graph_name]['elapsed_seconds']:.2f}s.",
            flush=True,
        )

    _preload_complete = True
    print(
        "Finished web-demo graph and preview preload in "
        f"{time.perf_counter() - started:.2f}s.",
        flush=True,
    )


def preload_demo_rl_runtime() -> None:
    """Load the RL embedder and all enabled RL graph views before serving."""
    if not LOAD_ALL_MODELS:
        _model_preload_status["rl"] = {
            "loaded": False,
            "skipped": True,
            "reason": "RL is disabled in limited web-demo mode",
        }
        return

    policy = get_rl_policy_config(DEFAULT_RL_POLICY_ID)
    started = time.perf_counter()
    status: dict[str, Any] = {
        "loaded": False,
        "policy": policy.id,
        "graphs": [],
    }
    _model_preload_status["rl"] = status

    try:
        runtime = get_rl_runtime(policy.id)
        for graph_name in get_available_demo_graphs():
            if not graph_supports_rl(graph_name, policy.id):
                continue
            bundle = get_loaded_graph_bundle(graph_name)
            get_rl_graph(runtime, bundle)
            status["graphs"].append(graph_name)
    except Exception as exc:
        status.update(
            {
                "error": exception_message(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(
            "RL runtime preload failed: "
            f"{exception_message(exc)}",
            flush=True,
        )
        raise RuntimeError("Could not preload the RL runtime.") from exc

    status.update(
        {
            "loaded": True,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    )
    print(
        "Finished RL runtime preload in "
        f"{status['elapsed_seconds']:.2f}s.",
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

    models = get_model_options_for_preload(get_demo_models(discover_models()))
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
    runtime_groups = group_graphs_by_embedding_universe(graph_names)
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
            "runtime_groups": [],
        }
        _model_preload_status["models"].append(model_status)

        print(
            "Preloading A* model "
            f"{model_index}/{len(models)}: {model_option['label']} "
            f"(full dim {dim}; warms dims {model_option['dims']}).",
            flush=True,
        )

        model_failed = False
        indexed_graphs = set()
        for cache_suffix, node_universe, grouped_graph_names in runtime_groups:
            group_status = {
                "cache_suffix": cache_suffix,
                "node_universe": node_universe,
                "graphs": list(grouped_graph_names),
                "loaded": False,
            }
            model_status["runtime_groups"].append(group_status)

            try:
                runtime = get_model_runtime(
                    model_option["id"],
                    dim,
                    cache_suffix,
                    node_universe,
                )
                prepare_runtime_indexes(runtime, grouped_graph_names)
                warm_runtime_dimensions(runtime, model_option["dims"])
                warm_runtime_traversals(
                    runtime,
                    grouped_graph_names,
                    model_option["dims"],
                )
            except Exception as exc:
                model_failed = True
                group_status["error"] = exception_message(exc)
                model_status.update(
                    {
                        "loaded": False,
                        "error": exception_message(exc),
                        "elapsed_seconds": round(
                            time.perf_counter() - model_started,
                            2,
                        ),
                    }
                )
                print(
                    "A* model warmup failed: "
                    f"{model_option['label']}: {exception_message(exc)}",
                    flush=True,
                )
                break

            group_status["loaded"] = True
            indexed_graphs.update(runtime.indexed_graphs)

        if model_failed:
            failed_models.append(model_status)
            continue

        model_status.update(
            {
                "loaded": True,
                "indexed_graphs": sorted(indexed_graphs),
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
    if failed_models:
        failed_labels = ", ".join(model["label"] for model in failed_models)
        raise RuntimeError(
            "Could not preload every A* runtime: "
            f"{failed_labels}."
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


def group_graphs_by_embedding_universe(
    graph_names: tuple[str, ...],
) -> tuple[tuple[str | None, str, tuple[str, ...]], ...]:
    """Group graphs that can share an A* model runtime and embedding cache."""
    groups: dict[tuple[str | None, str], list[str]] = {}
    for graph_name in graph_names:
        key = (
            get_graph_cache_suffix(graph_name),
            get_graph_node_universe(graph_name),
        )
        groups.setdefault(key, []).append(graph_name)

    return tuple(
        (cache_suffix, node_universe, tuple(names))
        for (cache_suffix, node_universe), names in groups.items()
    )


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
            if is_ignorable_graph_node(node):
                continue
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
        warmup_query = get_graph_warmup_query(bundle)
        if warmup_query is None:
            continue

        source, target = warmup_query
        indexed_graph = runtime.indexed_graphs.get(graph_name)

        for dim in dims:
            runtime.embedder.set_matryoshka_dim(dim)
            max_visits = get_default_astar_max_visits(
                graph_name,
                runtime.model_path,
                dim,
            )["value"]
            config = {
                "embedding_index_min_successors": (
                    EMBEDDING_INDEX_MIN_SUCCESSORS
                ),
                "astar_max_visits": max_visits,
            }
            if indexed_graph is not None:
                config["_indexed_graph"] = indexed_graph

            synchronize_embedding_device(runtime.embedder)
            started = time.perf_counter()
            path, visited_nodes = traverse_graph(
                bundle.graph,
                source,
                target,
                runtime.embedder,
                ts.astar_traverse,
                config,
            )
            synchronize_embedding_device(runtime.embedder)
            print(
                "Finished dynamic A* startup warmup: "
                f"{graph_name}, {runtime.model_path}, d={dim}, "
                f"visited={visited_nodes:,}, found={bool(path)}, "
                f"elapsed={time.perf_counter() - started:.2f}s.",
                flush=True,
            )

    runtime.embedder.set_matryoshka_dim(dims[0])


def get_graph_warmup_query(bundle: GraphBundle) -> tuple[str, str] | None:
    """Select a representative two-hop query from graph topology alone."""
    cached = _graph_warmup_queries.get(bundle.name)
    if cached is not None:
        return cached

    adjacency = bundle.graph._succ
    fallback_edge = None
    best_query = None
    best_branching_distance = None

    for source in bundle.nodes:
        successors = adjacency.get(source, {})
        out_degree = len(successors)
        if not out_degree:
            continue

        if fallback_edge is None:
            fallback_edge = (source, next(iter(successors)))
        if out_degree < EMBEDDING_INDEX_MIN_SUCCESSORS:
            continue

        direct_successors = set(successors)
        target = next(
            (
                candidate
                for intermediary in successors
                for candidate in adjacency.get(intermediary, {})
                if candidate != source and candidate not in direct_successors
            ),
            None,
        )
        if target is None:
            continue

        branching_distance = out_degree - EMBEDDING_INDEX_MIN_SUCCESSORS
        if (
            best_branching_distance is None
            or branching_distance < best_branching_distance
        ):
            best_query = (source, target)
            best_branching_distance = branching_distance
            if branching_distance == 0:
                break

    selected_query = best_query or fallback_edge
    if selected_query is not None:
        _graph_warmup_queries[bundle.name] = selected_query

    return selected_query


def synchronize_embedding_device(embedder) -> None:
    if not str(getattr(embedder, "device", "")).startswith("cuda"):
        return

    import torch

    torch.cuda.synchronize()


def get_loaded_graph_bundle(graph_name: str) -> GraphBundle:
    graph_name = validate_demo_graph_name(graph_name)
    cached = _graph_cache.get(graph_name)
    return cached if cached is not None else get_graph_bundle(graph_name)


def get_graph_bundle(graph_name: str) -> GraphBundle:
    graph_name = validate_demo_graph_name(graph_name)

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
        remove_ignorable_graph_nodes(graph, graph_name)
        nodes = sorted(
            str(node)
            for node in graph.nodes
            if not is_ignorable_graph_node(node)
        )
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
        _preload_status[graph_name] = {
            "loaded": True,
            "path": str(bundle.path),
            "nodes": bundle.graph.number_of_nodes(),
            "edges": bundle.graph.number_of_edges(),
        }
        return bundle


def is_ignorable_graph_node(node: Any) -> bool:
    """Exclude blank and punctuation-only concepts such as ``# #``.

    CauseNet contains a few placeholder concepts that are not present in the
    embedding node universes. A node remains valid when it has at least one
    alphanumeric character, so concepts such as ``C#`` are retained.
    """
    value = str(node).strip()
    return not value or not any(character.isalnum() for character in value)


def remove_ignorable_graph_nodes(graph, graph_name: str) -> int:
    ignored_nodes = [
        node
        for node in list(graph.nodes)
        if is_ignorable_graph_node(node)
    ]
    if not ignored_nodes:
        return 0

    graph.remove_nodes_from(ignored_nodes)
    print(
        "Ignored blank or punctuation-only graph node(s) for web-demo graph "
        f"{graph_name}: removed {len(ignored_nodes):,}.",
        flush=True,
    )
    return len(ignored_nodes)


def remove_ignorable_rl_nodes(graph, graph_name: str) -> int:
    """Apply the same concept filter to the separate RL adjacency graph."""
    ignored_nodes = {
        node
        for node in graph.nodes
        if is_ignorable_graph_node(node)
    }
    if not ignored_nodes:
        return 0

    for node in ignored_nodes:
        graph.adjacency.pop(node, None)
    for source, successors in graph.adjacency.items():
        graph.adjacency[source] = [
            target
            for target in successors
            if target not in ignored_nodes
        ]
    graph.edge_sources = {
        (source, target): sentence
        for (source, target), sentence in graph.edge_sources.items()
        if source not in ignored_nodes and target not in ignored_nodes
    }
    graph.nodes.difference_update(ignored_nodes)
    print(
        "Ignored blank or punctuation-only RL graph node(s) for web-demo graph "
        f"{graph_name}: removed {len(ignored_nodes):,}.",
        flush=True,
    )
    return len(ignored_nodes)


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
    graph_name = canonical_graph_name(graph_name)
    return GRAPH_CACHE_DIR / f"{graph_name}.pickle"


def graph_cache_paths_for_read(graph_name: str) -> list[Path]:
    graph_name = canonical_graph_name(graph_name)
    return [
        GRAPH_CACHE_DIR / f"{name}.pickle"
        for name in (graph_name, *graph_aliases_for(graph_name))
    ]


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

    cache_path = next(
        (path for path in graph_cache_paths_for_read(graph_name) if path.exists()),
        graph_cache_path(graph_name),
    )
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
    graph_name = canonical_graph_name(graph_name)
    return VISUAL_CACHE_DIR / f"{graph_name}_preview_v{VISUAL_CACHE_VERSION}_{limit}.json"


def overview_visual_cache_paths_for_read(graph_name: str, limit: int) -> list[Path]:
    graph_name = canonical_graph_name(graph_name)
    return [
        VISUAL_CACHE_DIR / f"{name}_preview_v{VISUAL_CACHE_VERSION}_{limit}.json"
        for name in (graph_name, *graph_aliases_for(graph_name))
    ]


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
    graph_name = canonical_graph_name(graph_name)
    cache_path = next(
        (
            path
            for path in overview_visual_cache_paths_for_read(graph_name, limit)
            if path.exists()
        ),
        overview_visual_cache_path(graph_name, limit),
    )
    if not cache_path.exists() or not graph_path.exists():
        return None

    try:
        with open(cache_path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable visual cache {cache_path}: {exc}", flush=True)
        return None

    expected_signatures = [
        visual_cache_signature(name, graph_path, limit)
        for name in (graph_name, *graph_aliases_for(graph_name))
    ]
    if payload.get("signature") not in expected_signatures:
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


def get_overview_visual_graph(bundle: GraphBundle, limit: int) -> dict[str, Any]:
    """Return the prebuilt overview graph without repeating disk or graph work."""
    key = (bundle.name, limit)
    cached = _overview_visual_cache.get(key)
    if cached is not None:
        return cached

    cached = load_overview_visual_cache(bundle.name, bundle.path, limit)
    if cached is not None:
        _overview_visual_cache[key] = cached
        return cached

    selected = sample_overview_nodes(bundle.graph, limit)
    visual_graph = build_visual_graph(
        bundle.graph,
        selected,
        edge_limit=preview_edge_budget(bundle.graph, len(selected)),
        meta=visual_graph_meta(bundle, "memory", limit),
    )
    save_overview_visual_cache(bundle.name, bundle.path, limit, visual_graph)
    _overview_visual_cache[key] = visual_graph
    return visual_graph


def normalize_model_path(model_path: str) -> str:
    normalized = str(model_path).replace("\\", "/")
    if "://" in normalized:
        return normalized

    local_path = Path(normalized)
    if local_path.is_absolute() or resolve_code_path(local_path).exists():
        resolved = local_path if local_path.is_absolute() else resolve_code_path(local_path)
        try:
            return resolved.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    return normalized


def read_training_metadata(model_path: str) -> dict[str, Any]:
    metadata_path = resolve_code_path(model_path) / "training_metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def discover_models() -> tuple[dict[str, Any], ...]:
    models: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_model_types: set[tuple[str, str, str | None, str | None, bool | None]] = set()

    def add_model(model_path: str, is_finetuned: bool = False):
        normalized_path = normalize_model_path(model_path)
        if normalized_path in seen_paths:
            return

        metadata = read_training_metadata(normalized_path)
        dim = infer_model_dim(normalized_path, metadata)
        if dim is None:
            return

        if not is_finetuned and not embedding_cache_exists(normalized_path):
            return

        distance_metric = get_model_distance_metric(normalized_path)
        parsed_config = parse_model_config(
            normalized_path,
            metadata=metadata,
            is_finetuned=is_finetuned,
        )
        model_type = (
            parsed_config["model_key"] or normalized_path,
            parsed_config["variant"],
            parsed_config["activation"],
            parsed_config["distance"],
            parsed_config["normalize"],
        )
        # A search-method choice represents one model configuration. Matryoshka
        # dimensions are selected separately in the UI, and duplicate training
        # directories for the same configuration must not create extra choices.
        if model_type in seen_model_types:
            return

        dims = get_matryoshka_dims(dim)
        cache_name = Path(normalized_path).name
        label = format_model_display_label(
            normalized_path,
            is_finetuned=is_finetuned,
            metadata=metadata,
        )
        base_label = format_model_display_label(
            normalized_path,
            variant="base",
            is_finetuned=False,
            metadata=metadata,
        ).removesuffix(" Base")

        models.append(
            {
                "id": normalized_path,
                "label": label,
                "base_label": base_label,
                "config_label": label,
                "model_dim": dim,
                "dims": dims,
                "distance": distance_config_token(parsed_config["distance"])
                or distance_metric.name.lower(),
                "distance_label": distance_metric.name.title(),
                "is_finetuned": is_finetuned,
                "cache_name": cache_name,
                "variant": parsed_config["variant"],
                "activation": parsed_config["activation"],
                "model_key": parsed_config["model_key"],
                "normalize": parsed_config["normalize"],
                "metadata": metadata,
            }
        )
        seen_paths.add(normalized_path)
        seen_model_types.add(model_type)

    for model_path in BASE_MODELS:
        add_model(model_path)

    for model_path in get_fine_tuned_models(DEFAULT_RUN_SUFFIX):
        add_model(model_path, is_finetuned=True)

    # The Granite reference plus its three activation/distance variants are
    # stored under the ablation run suffix, not the normal fine-tuning suffix.
    # Include them explicitly before the general directory scan.
    for model_path in get_ablation_fine_tuned_models(DEFAULT_RUN_SUFFIX):
        add_model(model_path, is_finetuned=True)

    if LIGHTNING_MODELS_DIR.exists():
        for model_dir in sorted(LIGHTNING_MODELS_DIR.iterdir()):
            if model_dir.is_dir() and model_dir.name != "old":
                add_model(
                    str(model_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                    is_finetuned=True,
                )

    return tuple(models)


def get_demo_models(
    models: tuple[dict[str, Any], ...],
    *,
    load_all: bool | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the A* configurations exposed and preloaded by this demo run."""
    if load_all is None:
        load_all = LOAD_ALL_MODELS
    if load_all:
        return models

    best_models = [
        model
        for model in models
        if model["model_key"] == BEST_MODEL_KEY
        and model["variant"] == BEST_MODEL_VARIANT
        and model["activation"] == BEST_MODEL_ACTIVATION
        and model["distance"] == BEST_MODEL_DISTANCE
    ]
    if len(best_models) != 1:
        raise RuntimeError(
            "Limited web-demo mode requires exactly one Granite FT "
            "ReLU+Euclidean model."
        )

    best_model = dict(best_models[0])
    if BEST_MODEL_DIMENSION not in best_model["dims"]:
        raise RuntimeError(
            "Limited web-demo mode requires dimension "
            f"{BEST_MODEL_DIMENSION} for {best_model['label']}."
        )
    best_model["dims"] = [BEST_MODEL_DIMENSION]
    return (best_model,)


def discover_search_methods(
    models: tuple[dict[str, Any], ...] | None = None,
    *,
    load_all: bool | None = None,
) -> tuple[dict[str, Any], ...]:
    if load_all is None:
        load_all = LOAD_ALL_MODELS
    if models is None:
        models = get_demo_models(discover_models(), load_all=load_all)

    methods: dict[str, dict[str, Any]] = {}

    def add_method(method: dict[str, Any]) -> None:
        methods.setdefault(method["id"], method)

    if load_all:
        graph_choices = get_demo_graph_choices(load_all=True)
        add_method(
            {
                "id": stable_config_identity(algorithm="bfs"),
                "algorithm": "bfs",
                "label": "BFS",
                "supported_graphs": [
                    graph
                    for graph in graph_choices
                    if graph_supports_algorithm(graph, "bfs")
                ],
                "config": {},
            }
        )

        policy = get_rl_policy_config(DEFAULT_RL_POLICY_ID)
        add_method(
            {
                "id": stable_config_identity(
                    algorithm="rl",
                    policy_config_id=policy.id,
                    checkpoint_id=str(policy.checkpoint_path),
                ),
                "algorithm": "rl",
                "label": policy.label,
                "description": policy.description,
                "supported_graphs": [
                    graph
                    for graph in graph_choices
                    if graph in policy.supported_graphs
                ],
                "config": policy.public_config(),
            }
        )

    for model in models:
        add_method(
            astar_method_payload(
                model,
                graph_choices=get_demo_graph_choices(load_all=load_all),
            )
        )

    return tuple(sorted(methods.values(), key=method_sort_key))


def astar_method_payload(
    model: dict[str, Any],
    *,
    graph_choices: tuple[str, ...] = SUPPORTED_INFERENCE_GRAPHS,
) -> dict[str, Any]:
    label = format_model_display_label(
        model["id"],
        is_finetuned=model["is_finetuned"],
        metadata=model.get("metadata"),
    )
    config = {
        "model_id": model["id"],
        "model_key": model["model_key"],
        "variant": model["variant"],
        "activation": model["activation"],
        "distance": model["distance"],
        "dimensions": model["dims"],
        "default_dimension": model["dims"][0],
        "model_dim": model["model_dim"],
        "checkpoint_id": model["id"],
        "normalize": model["normalize"],
        "cache_name": model["cache_name"],
    }
    embedding_model = get_embedding_model(model["id"])
    if embedding_model is not None:
        config.update(
            {
                "model_label": embedding_model.label,
                "model_identifier": embedding_model.identifier,
                "parameters": embedding_model.parameters,
                "full_dimension": embedding_model.full_dimension,
            }
        )

    config_id = stable_config_identity(
        algorithm="astar",
        model_id=model["id"],
        variant=model["variant"],
        activation=model["activation"],
        distance=model["distance"],
        checkpoint_id=model["id"],
        normalize=model["normalize"],
    )
    return {
        "id": config_id,
        "algorithm": "astar",
        "label": label,
        "supported_graphs": [
            graph
            for graph in graph_choices
            if graph_supports_algorithm(graph, "astar")
        ],
        "config": config,
    }


def get_search_method(method_id: str) -> dict[str, Any] | None:
    return next(
        (
            method
            for method in discover_search_methods()
            if method["id"] == method_id
        ),
        None,
    )


def get_model_option(model_id: str) -> dict[str, Any] | None:
    normalized_id = normalize_model_path(model_id)
    return next(
        (
            model
            for model in get_demo_models(discover_models())
            if model["id"] == normalized_id or model["id"] == model_id
        ),
        None,
    )


def resolve_astar_model_option(
    method_id: str | None,
    model_id: str | None,
    dim: int | None,
) -> dict[str, Any]:
    if method_id:
        method = get_search_method(method_id)
        if method is None or method.get("algorithm") != "astar":
            raise HTTPException(
                status_code=400,
                detail=f"Unknown A* model configuration '{method_id}'.",
            )
        config = method["config"]
        model_id = config["model_id"]
        dim = dim if dim is not None else config["default_dimension"]

    if model_id is None or dim is None:
        raise HTTPException(
            status_code=400,
            detail="A* requests require model_config_id or model_id and dimension.",
        )

    model_option = get_model_option(model_id)
    if model_option is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or unavailable model '{model_id}'.",
        )

    if dim not in model_option["dims"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dimension {dim} is not available for "
                f"{model_option['label']}."
            ),
        )

    payload = dict(model_option)
    payload["selected_dim"] = dim
    payload["selected_label"] = format_model_display_label(
        model_option["id"],
        dimension=dim,
        include_dimension=len(model_option["dims"]) > 1,
        is_finetuned=model_option["is_finetuned"],
        metadata=model_option.get("metadata"),
    )
    payload["config_id"] = stable_config_identity(
        algorithm="astar",
        model_id=model_option["id"],
        variant=model_option["variant"],
        activation=model_option["activation"],
        distance=model_option["distance"],
        dimension=dim,
        checkpoint_id=model_option["id"],
        normalize=model_option["normalize"],
    )
    return payload


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


def get_rl_runtime(policy_config_id: str) -> RLRuntime:
    policy = get_rl_policy_config(policy_config_id)

    with _rl_runtime_lock:
        runtime = _rl_runtime_cache.get(policy.id)
        if runtime is not None:
            return runtime

        try:
            from core.embeddings import DistanceMetric, GloveEmbeder

            embedder = GloveEmbeder(
                policy.glove_path,
                DistanceMetric.COSINE,
                device=EMBEDDING_DEVICE,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load RL GloVe embedder: {exc}",
            ) from exc

        runtime = RLRuntime(policy.id, embedder, {})
        _rl_runtime_cache[policy.id] = runtime
        return runtime


def get_rl_graph(runtime: RLRuntime, bundle: GraphBundle):
    cached = runtime.graphs.get(bundle.name)
    if cached is not None:
        return cached

    graph_path = bundle.path
    start_time = time.perf_counter()
    try:
        graph = load_rl_graph(
            graph_path,
            use_inverse=False,
            progress_every=1_000_000,
            progress_label=f"{bundle.name} RL graph",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load RL graph for {bundle.label}: {exc}",
        ) from exc

    remove_ignorable_rl_nodes(graph, bundle.name)
    runtime.graphs[bundle.name] = graph
    print(
        "Loaded RL graph for inference: "
        f"{bundle.name} with {len(graph.nodes):,} nodes in "
        f"{time.perf_counter() - start_time:.2f}s.",
        flush=True,
    )
    return graph


def build_astar_runtime_config(
    graph_name: str,
    model_option: dict[str, Any],
    request: InferenceAStarConfig,
) -> dict[str, int]:
    config = {
        "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
    }

    if request.astar_max_visits is None:
        max_visits = get_default_astar_max_visits(
            graph_name,
            model_option["id"],
            model_option["selected_dim"],
        )["value"]
    else:
        max_visits = request.astar_max_visits

    validate_search_cap(max_visits, "A* max visits")
    config["astar_max_visits"] = max_visits

    threshold = request.embedding_index_min_successors
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
    graph_name = canonical_graph_name(graph_name)
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
        p95_files = [
            EVALUATION_DIR
            / graph_dir
            / "msmarco_train"
            / DEFAULT_RUN_SUFFIX
            / "visited_nodes_analysis.json"
            for graph_dir in (candidate_graph, *graph_aliases_for(candidate_graph))
        ]
        p95_file = next((path for path in p95_files if path.exists()), p95_files[0])
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
        get_graph_node_universe(DEFAULT_GRAPH_NAME),
    )
    return (
        node_file.exists()
        and (EMBEDDINGS_DIR / f"{cache_name}_vectors.npy").exists()
    )


def infer_model_dim(
    model_path: str,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    metadata = metadata or {}
    metadata_model_path = metadata.get("model_path")
    if metadata_model_path in MODEL_DIM_HINTS:
        return MODEL_DIM_HINTS[metadata_model_path]

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

    # GraphBundle.nodes is sorted during startup. Jump directly to the prefix
    # range so async datalist requests do not scan millions of nodes per key.
    prefix_matches = []
    start = bisect_left(bundle.nodes, query)
    for node in bundle.nodes[start:]:
        lowered = node.lower()
        if lowered.startswith(query):
            prefix_matches.append(node)
            if len(prefix_matches) >= limit:
                return prefix_matches
        elif prefix_matches or lowered > query:
            break

    if prefix_matches:
        return prefix_matches

    contains_matches = [
        node
        for node in bundle.nodes
        if query in node.lower()
    ]
    return contains_matches[:limit]


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
