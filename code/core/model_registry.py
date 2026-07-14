from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    label: str
    identifier: str
    parameters: str
    full_dimension: int
    aliases: tuple[str, ...]


MODEL_ORDER = ("mpnet", "bge", "granite", "mxbai", "qwen3_0_6b")

EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "mpnet": EmbeddingModel(
        key="mpnet",
        label="MPNet",
        identifier="sentence-transformers/all-mpnet-base-v2",
        parameters="109M",
        full_dimension=768,
        aliases=("all-mpnet-base-v2",),
    ),
    "bge": EmbeddingModel(
        key="bge",
        label="BGE",
        identifier="BAAI/bge-large-en-v1.5",
        parameters="335M",
        full_dimension=1024,
        aliases=("bge-large-en-v1.5",),
    ),
    "granite": EmbeddingModel(
        key="granite",
        label="Granite",
        identifier="ibm-granite/granite-embedding-english-r2",
        parameters="149M",
        full_dimension=768,
        aliases=("granite-embedding-english-r2",),
    ),
    "mxbai": EmbeddingModel(
        key="mxbai",
        label="mxbai",
        identifier="mixedbread-ai/mxbai-embed-large-v1",
        parameters="335M",
        full_dimension=1024,
        aliases=("mxbai-embed-large-v1",),
    ),
    "qwen3_0_6b": EmbeddingModel(
        key="qwen3_0_6b",
        label="Qwen3-0.6B",
        identifier="Qwen/Qwen3-Embedding-0.6B",
        parameters="595M",
        full_dimension=1024,
        aliases=("Qwen3-Embedding-0.6B",),
    ),
}

MODEL_DIMENSIONS = {
    model.identifier: model.full_dimension
    for model in EMBEDDING_MODELS.values()
}

_MODEL_ALIASES: dict[str, str] = {}
for model in EMBEDDING_MODELS.values():
    values = (model.identifier, model.key, model.label, *model.aliases)
    for value in values:
        _MODEL_ALIASES[value.lower()] = model.key
        _MODEL_ALIASES[Path(value).name.lower()] = model.key


MODEL_NAME_STOP_TOKENS = {
    "relu",
    "gelu",
    "cos",
    "cosine",
    "cosine_distance",
    "euclid",
    "euclidean",
    "l2",
    "norm",
    "nonorm",
    "matryoshka",
    "single",
    "best",
}

ACTIVATION_LABELS = {
    "relu": "ReLU",
    "gelu": "GELU",
}

DISTANCE_LABELS = {
    "cosine": "Cosine",
    "euclidean": "Euclidean",
}

VARIANT_ORDER = {
    "base": 0,
    "finetuned": 1,
    "ablation": 2,
}


def model_name(value: str | Path) -> str:
    return Path(str(value).replace("\\", "/")).name


def get_model_base_name(value: str | Path) -> str:
    name = model_name(value)
    name = name.removesuffix("_finetuned")
    name = name.removesuffix("_best")
    parts = name.split("_")

    base_parts = []
    for part in parts:
        if part.lower() in MODEL_NAME_STOP_TOKENS:
            break
        base_parts.append(part)

    return "_".join(base_parts)


def canonical_model_key(value: str | Path | None) -> str | None:
    if value is None:
        return None

    raw = str(value).replace("\\", "/")
    candidates = [
        raw,
        model_name(raw),
        get_model_base_name(raw),
    ]
    for candidate in candidates:
        key = _MODEL_ALIASES.get(candidate.lower())
        if key is not None:
            return key

    return None


def get_embedding_model(value: str | Path) -> EmbeddingModel | None:
    key = canonical_model_key(value)
    return EMBEDDING_MODELS.get(key) if key is not None else None


def canonical_model_label(value: str | Path) -> str:
    model = get_embedding_model(value)
    if model is not None:
        return model.label
    return get_model_base_name(value) or model_name(value)


def normalize_activation(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"relu", "gelu"}:
        return normalized

    raise ValueError(f"Unsupported activation function: {value}")


def normalize_distance(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"cos", "cosine", "cosine_distance"}:
        return "cosine"
    if normalized in {"euclid", "euclidean", "l2"}:
        return "euclidean"

    raise ValueError(f"Unsupported distance function: {value}")


def activation_label(value: str | None) -> str | None:
    normalized = normalize_activation(value)
    return ACTIVATION_LABELS[normalized] if normalized is not None else None


def distance_label(value: str | None) -> str | None:
    normalized = normalize_distance(value)
    return DISTANCE_LABELS[normalized] if normalized is not None else None


def distance_config_token(value: str | None) -> str | None:
    normalized = normalize_distance(value)
    if normalized == "euclidean":
        return "euclid"
    return normalized


def infer_variant(model_id: str, is_finetuned: bool | None = None) -> str:
    name = model_name(model_id)
    lowered = name.lower()
    if "_ablation_" in lowered:
        return "ablation"
    if is_finetuned is True or lowered.endswith("_finetuned"):
        return "finetuned"
    if any(token in lowered.split("_") for token in MODEL_NAME_STOP_TOKENS):
        return "finetuned"
    return "base"


def _first_token(tokens: list[str], allowed: set[str]) -> str | None:
    for token in tokens:
        normalized = token.lower()
        if normalized in allowed:
            return normalized
    return None


def parse_model_config(
    model_id: str,
    metadata: dict[str, Any] | None = None,
    is_finetuned: bool | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    tokens = model_name(model_id).removesuffix("_finetuned").split("_")
    base_model_id = metadata.get("model_path") or get_model_base_name(model_id)
    activation = metadata.get("activation") or _first_token(tokens, {"relu", "gelu"})
    distance = metadata.get("distance") or _first_token(
        tokens,
        {"cos", "cosine", "cosine_distance", "euclid", "euclidean", "l2"},
    )
    normalize = metadata.get("normalize")
    if normalize is None:
        if "norm" in tokens:
            normalize = True
        elif "nonorm" in tokens:
            normalize = False

    variant = infer_variant(model_id, is_finetuned=is_finetuned)
    if variant == "base":
        activation = None
        distance = None

    return {
        "model_key": canonical_model_key(base_model_id) or canonical_model_key(model_id),
        "base_model_id": get_embedding_model(base_model_id).identifier
        if get_embedding_model(base_model_id)
        else base_model_id,
        "variant": variant,
        "activation": normalize_activation(activation) if activation else None,
        "distance": normalize_distance(distance) if distance else None,
        "normalize": normalize,
    }


def format_model_display_label(
    model_id: str,
    *,
    variant: str | None = None,
    activation: str | None = None,
    distance: str | None = None,
    dimension: int | None = None,
    include_dimension: bool = False,
    is_finetuned: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    config = parse_model_config(
        model_id,
        metadata=metadata,
        is_finetuned=is_finetuned,
    )
    if variant is None:
        variant = config["variant"]
    if activation is None:
        activation = config["activation"]
    if distance is None:
        distance = config["distance"]

    base_label = canonical_model_label(config["base_model_id"] or model_id)

    if variant == "base":
        label = f"{base_label} Base"
    else:
        if activation is None or distance is None:
            label = f"{base_label} {'AB' if variant == 'ablation' else 'FT'}"
        else:
            variant_label = "AB" if variant == "ablation" else "FT"
            label = (
                f"{base_label} {variant_label} "
                f"{activation_label(activation)}+{distance_label(distance)}"
            )

    if include_dimension and dimension is not None:
        label = f"{label} (d={dimension})"

    return label


def stable_config_identity(
    *,
    algorithm: str,
    model_id: str | None = None,
    variant: str | None = None,
    activation: str | None = None,
    distance: str | None = None,
    dimension: int | None = None,
    checkpoint_id: str | None = None,
    policy_config_id: str | None = None,
    normalize: bool | None = None,
) -> str:
    algorithm = algorithm.lower()
    if algorithm == "bfs":
        return "algorithm=bfs"

    if algorithm == "rl":
        return "|".join(
            [
                "algorithm=rl",
                f"policy={policy_config_id or ''}",
                f"checkpoint={checkpoint_id or ''}",
            ]
        )

    if algorithm != "astar":
        raise ValueError(f"Unsupported algorithm for identity: {algorithm}")

    config = parse_model_config(
        model_id or "",
        is_finetuned=(variant in {"finetuned", "ablation"})
        if variant is not None
        else None,
    )
    resolved_variant = variant or config["variant"]
    resolved_activation = activation or config["activation"]
    resolved_distance = distance or config["distance"]

    return "|".join(
        [
            "algorithm=astar",
            f"model={canonical_model_key(model_id) or model_id or ''}",
            f"variant={resolved_variant or ''}",
            f"activation={normalize_activation(resolved_activation) if resolved_activation else ''}",
            f"distance={normalize_distance(resolved_distance) if resolved_distance else ''}",
            f"dimension={dimension or ''}",
            f"checkpoint={checkpoint_id or model_id or ''}",
            f"normalize={normalize if normalize is not None else config.get('normalize')}",
        ]
    )


def method_sort_key(method: dict[str, Any]) -> tuple[Any, ...]:
    algorithm = method.get("algorithm")
    if algorithm == "bfs":
        return (0, 0, "", "", 0)
    if algorithm == "rl":
        return (1, 0, "", "", 0)

    config = method.get("config", {})
    model_key = config.get("model_key") or canonical_model_key(config.get("model_id"))
    model_index = MODEL_ORDER.index(model_key) if model_key in MODEL_ORDER else 999
    variant = config.get("variant", "base")
    activation = config.get("activation") or ""
    distance = config.get("distance") or ""
    dimension = config.get("dimension") or 0
    return (
        2 + VARIANT_ORDER.get(variant, 99),
        model_index,
        activation,
        distance,
        dimension,
    )
