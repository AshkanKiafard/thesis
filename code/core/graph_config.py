from pathlib import Path

DEFAULT_GRAPH_NAME = "causenet"

GRAPH_CONFIGS = {
    "causenet": {
        "label": "CauseNet",
        "path": Path("data/graphs/causenet-precision.jsonl"),
    },
    "causenet_full": {
        "label": "CauseNet full",
        "path": Path("data/graphs/causenet-full.jsonl"),
    },
    "causalbank": {
        "label": "CausalBank",
        "path": Path("data/graphs/Lexical_Cause_Effect_Graph.filtered.txt"),
    },
    "causalbank_full": {
        "label": "CausalBank full",
        "path": Path("data/graphs/Lexical_Cause_Effect_Graph.txt"),
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
