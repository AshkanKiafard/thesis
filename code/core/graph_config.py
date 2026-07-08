from core.constants import (
    CAUSALBANK_FULL_GRAPH_PATH,
    CAUSALBANK_GRAPH_PATH,
    CAUSENET_FULL_GRAPH_PATH,
    CAUSENET_GRAPH_PATH,
)

DEFAULT_GRAPH_NAME = "causenet"

GRAPH_CONFIGS = {
    "causenet": {
        "label": "CauseNet",
        "path": CAUSENET_GRAPH_PATH,
    },
    "causenet_full": {
        "label": "CauseNet full",
        "path": CAUSENET_FULL_GRAPH_PATH,
    },
    "causalbank": {
        "label": "CausalBank",
        "path": CAUSALBANK_GRAPH_PATH,
    },
    "causalbank_full": {
        "label": "CausalBank full",
        "path": CAUSALBANK_FULL_GRAPH_PATH,
    },
}


def graph_choices():
    return tuple(GRAPH_CONFIGS.keys())


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
