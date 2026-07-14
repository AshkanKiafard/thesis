from core.constants import (
    CAUSENET_FULL_NODE_UNIVERSE,
    CAUSALBANK_FULL_GRAPH_PATH,
    CAUSALBANK_GRAPH_PATH,
    CAUSENET_FULL_GRAPH_PATH,
    CAUSENET_GRAPH_PATH,
    MERGED_NODE_UNIVERSE,
)

DEFAULT_GRAPH_NAME = "causenet"
SUPPORTED_INFERENCE_GRAPHS = ("causenet", "causenet_full", "causalbank")
DEFAULT_INFERENCE_GRAPH = DEFAULT_GRAPH_NAME

GRAPH_CONFIGS = {
    "causenet": {
        "id": "causenet",
        "label": "CauseNet Precision",
        "path": CAUSENET_GRAPH_PATH,
        "nodes": 80_214,
        "edges": 197_376,
        "bfs_p95_cap": 12_170,
        "bfs_p95_cap_source": (
            "causenet/msmarco_train/v3 p95 successful BFS visits"
        ),
        "supported_algorithms": ("bfs", "rl", "astar"),
        "cache_suffix": None,
        "node_universe": MERGED_NODE_UNIVERSE,
        "web_demo": True,
    },
    "causenet_full": {
        "id": "causenet_full",
        "label": "CauseNet Full",
        "path": CAUSENET_FULL_GRAPH_PATH,
        "nodes": 12_185_920,
        "edges": 11_606_975,
        "bfs_p95_cap": 12_170,
        "bfs_p95_cap_source": (
            "causenet/msmarco_train/v3 p95 successful BFS visits "
            "via DEFAULT_P95_CONFIG_SOURCE_GRAPH"
        ),
        "supported_algorithms": ("bfs", "rl", "astar"),
        "cache_suffix": "causenet_full",
        "node_universe": CAUSENET_FULL_NODE_UNIVERSE,
        "web_demo": True,
    },
    "causalbank": {
        "id": "causalbank",
        "label": "CausalBank Filtered",
        "path": CAUSALBANK_GRAPH_PATH,
        "nodes": 77_264,
        "edges": 21_507_177,
        "bfs_p95_cap": 12_170,
        "bfs_p95_cap_source": (
            "causenet/msmarco_train/v3 p95 successful BFS visits "
            "via DEFAULT_P95_CONFIG_SOURCE_GRAPH"
        ),
        "supported_algorithms": ("bfs", "rl", "astar"),
        "cache_suffix": None,
        "node_universe": MERGED_NODE_UNIVERSE,
        "web_demo": True,
    },
    "causalbank_full": {
        "id": "causalbank_full",
        "label": "CausalBank Full",
        "path": CAUSALBANK_FULL_GRAPH_PATH,
        "nodes": 79_865,
        "edges": 92_270_736,
        "bfs_p95_cap": None,
        "bfs_p95_cap_source": None,
        "supported_algorithms": ("bfs", "astar"),
        "cache_suffix": "causalbank_full",
        "node_universe": MERGED_NODE_UNIVERSE,
        "web_demo": False,
    },
}


def graph_choices():
    return tuple(GRAPH_CONFIGS.keys())


def inference_graph_choices():
    return SUPPORTED_INFERENCE_GRAPHS


def get_graph_config(graph_name):
    try:
        return GRAPH_CONFIGS[graph_name]
    except KeyError as exc:
        choices = ", ".join(graph_choices())
        raise ValueError(f"Unknown graph '{graph_name}'. Choices: {choices}") from exc


def get_graph_label(graph_name):
    return get_graph_config(graph_name)["label"]


def get_graph_path(graph_name):
    return get_graph_config(graph_name)["path"]


def get_graph_node_count(graph_name):
    return get_graph_config(graph_name)["nodes"]


def get_graph_edge_count(graph_name):
    return get_graph_config(graph_name)["edges"]


def get_graph_bfs_p95_cap(graph_name):
    return get_graph_config(graph_name)["bfs_p95_cap"]


def get_graph_supported_algorithms(graph_name):
    return tuple(get_graph_config(graph_name)["supported_algorithms"])


def get_graph_cache_suffix(graph_name):
    return get_graph_config(graph_name)["cache_suffix"]


def get_graph_node_universe(graph_name):
    return get_graph_config(graph_name)["node_universe"]


def graph_supports_algorithm(graph_name, algorithm):
    return algorithm in get_graph_supported_algorithms(graph_name)
