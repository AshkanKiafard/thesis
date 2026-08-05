"""Evaluate validation-selected A* models over a shared visit-budget grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import model_registry
from core.config import (
    BUDGET_TRADEOFF_VISIT_BUDGETS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_RUN_SUFFIX,
    DEFAULT_VALIDATION_DATASET,
    EMBEDDING_INDEX_MIN_SUCCESSORS,
    VALIDATION_SELECTED_FINETUNED_MODELS,
)
from core.constants import (
    CAUSENET_GRAPH_PATH,
    EMBEDDINGS_DIR,
    EVALUATION_DIR,
    LIGHTNING_MODELS_DIR,
    PLOTS_DIR,
)
from core.utils import (
    get_dimension_embedding_cache_path,
    get_embedding_cache_path,
    get_embedding_cache_vectors_path,
)


EXPERIMENT_NAME = "budget_tradeoff"
RUN_SUFFIX = DEFAULT_RUN_SUFFIX
DATASET_NAME = "MS MARCO validation"
DATASET_ID = "msmarco_valid"
DATASET_PATH = Path(DEFAULT_VALIDATION_DATASET)
GRAPH_NAME = "CauseNet Precision"
GRAPH_ID = "causenet"
GRAPH_PATH = CAUSENET_GRAPH_PATH
NODE_UNIVERSE_PATH = EMBEDDINGS_DIR / "merged_causenet_ceg_nodes.jsonl"
VALIDATION_RESULTS_PATH = (
    EVALUATION_DIR
    / GRAPH_ID
    / DATASET_ID
    / RUN_SUFFIX
    / "evaluation_results.json"
)

BUDGETS = BUDGET_TRADEOFF_VISIT_BUDGETS
RESULTS_DIR = EVALUATION_DIR / EXPERIMENT_NAME
PLOT_DIR = PLOTS_DIR / EXPERIMENT_NAME
RESULTS_CSV_PATH = RESULTS_DIR / "budget_tradeoff_results.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "budget_tradeoff_results.json"
PLOT_PDF_PATH = PLOT_DIR / "budget_tradeoff.pdf"
PLOT_PNG_PATH = PLOT_DIR / "budget_tradeoff.png"

CSV_FIELDS = (
    "model",
    "budget",
    "evaluated_examples",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "average_visited_nodes",
    "average_runtime_ms",
)
INTEGER_RESULT_FIELDS = (
    "budget",
    "evaluated_examples",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
)
FLOAT_RESULT_FIELDS = (
    "precision",
    "recall",
    "f1",
    "accuracy",
    "average_visited_nodes",
    "average_runtime_ms",
)


@dataclass(frozen=True)
class ModelConfig:
    """One validation-selected fine-tuned embedding configuration."""

    model: str
    checkpoint_name: str
    embedding_dimension: int
    activation_function: str
    distance_metric: str
    existing_validation_budget: int

    @property
    def checkpoint_path(self) -> Path:
        return LIGHTNING_MODELS_DIR / self.checkpoint_name

    @property
    def embedding_path(self) -> Path:
        registered_model = model_registry.get_embedding_model(
            self.checkpoint_name
        )
        if registered_model is None:
            raise ValueError(
                f"Unknown embedding model checkpoint: {self.checkpoint_name}"
            )
        if self.embedding_dimension > registered_model.full_dimension:
            raise ValueError(
                f"Configured dimension {self.embedding_dimension} exceeds "
                f"{self.model}'s registered dimension "
                f"{registered_model.full_dimension}"
            )

        cache_file = get_embedding_cache_path(
            EMBEDDINGS_DIR,
            self.checkpoint_path,
        )
        if self.embedding_dimension < registered_model.full_dimension:
            cache_file = get_dimension_embedding_cache_path(
                cache_file,
                self.embedding_dimension,
            )
        return get_embedding_cache_vectors_path(cache_file)

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        values["checkpoint_path"] = _relative(self.checkpoint_path)
        values["embedding_path"] = _relative(self.embedding_path)
        return values


MODEL_CONFIGS = tuple(
    ModelConfig(**model_config)
    for model_config in VALIDATION_SELECTED_FINETUNED_MODELS
)
MODEL_NAMES = tuple(config.model for config in MODEL_CONFIGS)
EXPECTED_COMBINATIONS = len(MODEL_CONFIGS) * len(BUDGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate five validation-selected fine-tuned A* embedding models "
            "over the shared budget trade-off grid."
        )
    )
    parser.add_argument(
        "--embedding-device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Torch device used by the existing embedding evaluation pipeline.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Batch size used while loading the existing embedding index.",
    )
    args = parser.parse_args()
    if args.embedding_batch_size <= 0:
        parser.error("--embedding-batch-size must be greater than 0")
    return args


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def evaluation_command() -> str:
    return "python -m evaluation.run_budget_tradeoff"


def validate_static_config() -> None:
    """Reject accidental duplicate or malformed experiment configuration."""

    if len(BUDGETS) != len(set(BUDGETS)):
        raise ValueError(f"Duplicate visit budgets in configuration: {BUDGETS}")
    if tuple(sorted(BUDGETS)) != BUDGETS or any(budget <= 0 for budget in BUDGETS):
        raise ValueError(f"Budgets must be unique, positive, and sorted: {BUDGETS}")
    if len(MODEL_NAMES) != len(set(MODEL_NAMES)):
        raise ValueError(f"Duplicate stable model names: {MODEL_NAMES}")
    checkpoint_names = tuple(config.checkpoint_name for config in MODEL_CONFIGS)
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError(f"Duplicate model checkpoints: {checkpoint_names}")


def _coerce_result_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in CSV_FIELDS if field not in raw_row]
    if missing_fields:
        raise ValueError(f"Incomplete result row; missing fields: {missing_fields}")

    row = {field: raw_row[field] for field in CSV_FIELDS}
    row["model"] = str(row["model"])

    try:
        for field in INTEGER_RESULT_FIELDS:
            value = float(row[field])
            if not value.is_integer():
                raise ValueError(f"{field} is not an integer: {row[field]!r}")
            row[field] = int(value)
        for field in FLOAT_RESULT_FIELDS:
            row[field] = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value in result row {raw_row!r}: {exc}"
        ) from exc

    non_finite = [
        field for field in FLOAT_RESULT_FIELDS if not math.isfinite(row[field])
    ]
    if non_finite:
        raise ValueError(f"Non-finite metrics {non_finite} for {row['model']}")

    return row


def validate_aggregated_results(
    raw_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and canonically order the complete model-budget matrix."""

    validate_static_config()
    rows = [_coerce_result_row(row) for row in raw_rows]
    expected_models = set(MODEL_NAMES)
    expected_budgets = set(BUDGETS)
    found_models = {row["model"] for row in rows}
    found_budgets = {row["budget"] for row in rows}

    unexpected_models = sorted(found_models - expected_models)
    unexpected_budgets = sorted(found_budgets - expected_budgets)
    if unexpected_models:
        raise ValueError(f"Unexpected additional models: {unexpected_models}")
    if unexpected_budgets:
        raise ValueError(f"Unexpected additional budgets: {unexpected_budgets}")

    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["model"], row["budget"])
        if key in seen:
            raise ValueError(f"Duplicate model-budget combination: {key}")
        seen.add(key)

        confusion_total = sum(
            row[field]
            for field in (
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            )
        )
        if row["evaluated_examples"] <= 0:
            raise ValueError(f"No evaluated examples for {key}")
        if confusion_total != row["evaluated_examples"]:
            raise ValueError(
                f"Incomplete confusion counts for {key}: {confusion_total} != "
                f"{row['evaluated_examples']}"
            )
        if any(row[field] < 0 for field in INTEGER_RESULT_FIELDS):
            raise ValueError(f"Negative count or budget for {key}")
        if any(
            not 0.0 <= row[field] <= 1.0
            for field in ("precision", "recall", "f1", "accuracy")
        ):
            raise ValueError(f"Metric outside [0, 1] for {key}")
        if row["average_visited_nodes"] < 0 or row["average_runtime_ms"] < 0:
            raise ValueError(f"Negative efficiency metric for {key}")

    missing_models = sorted(expected_models - found_models)
    if missing_models:
        raise ValueError(f"Missing expected models: {missing_models}")

    missing_by_model = {
        model: sorted(
            expected_budgets
            - {row["budget"] for row in rows if row["model"] == model}
        )
        for model in MODEL_NAMES
    }
    missing_by_model = {
        model: budgets for model, budgets in missing_by_model.items() if budgets
    }
    if missing_by_model:
        raise ValueError(f"Models do not have all selected budgets: {missing_by_model}")

    expected_keys = {
        (model, budget) for model in MODEL_NAMES for budget in BUDGETS
    }
    missing_keys = sorted(expected_keys - seen)
    if missing_keys or len(rows) != EXPECTED_COMBINATIONS:
        raise ValueError(
            "Incomplete budget trade-off result matrix: expected "
            f"{EXPECTED_COMBINATIONS} combinations, found {len(rows)}; "
            f"missing={missing_keys}"
        )

    model_order = {model: index for index, model in enumerate(MODEL_NAMES)}
    return sorted(rows, key=lambda row: (model_order[row["model"]], row["budget"]))


def aggregated_results_agree(
    left_rows: Iterable[dict[str, Any]],
    right_rows: Iterable[dict[str, Any]],
) -> bool:
    """Compare serialized CSV and JSON aggregate rows with float tolerance."""

    left = validate_aggregated_results(left_rows)
    right = validate_aggregated_results(right_rows)
    if len(left) != len(right):
        return False

    for left_row, right_row in zip(left, right):
        for field in CSV_FIELDS:
            if field in FLOAT_RESULT_FIELDS:
                if not math.isclose(
                    left_row[field],
                    right_row[field],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    return False
            elif left_row[field] != right_row[field]:
                return False
    return True


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _validate_checkpoint(model_config) -> None:
    checkpoint_path = model_config.checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Missing model checkpoint for {model_config.model}: {checkpoint_path}"
        )

    weight_files = tuple(checkpoint_path.glob("*.safetensors")) + tuple(
        checkpoint_path.glob("*.bin")
    )
    if not weight_files:
        raise FileNotFoundError(
            f"Missing model weight file for {model_config.model}: {checkpoint_path}"
        )

    metadata_path = checkpoint_path / "training_metadata.json"
    _require_file(metadata_path, f"training metadata for {model_config.model}")
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    expected_activation = model_config.activation_function.lower()
    expected_distance = model_config.distance_metric.lower()
    actual_activation = str(metadata.get("activation", "")).lower()
    actual_distance = str(metadata.get("distance", "")).lower()
    if actual_distance == "euclid":
        actual_distance = "euclidean"
    if actual_activation != expected_activation or actual_distance != expected_distance:
        raise ValueError(
            f"Checkpoint configuration mismatch for {model_config.model}: "
            f"expected {expected_activation}/{expected_distance}, found "
            f"{actual_activation}/{actual_distance} in {metadata_path}"
        )


def validate_input_files() -> None:
    """Fail before model loading if any required experiment input is absent."""

    validate_static_config()
    _require_file(DATASET_PATH, "MS MARCO validation dataset")
    _require_file(GRAPH_PATH, "CauseNet Precision graph")
    _require_file(NODE_UNIVERSE_PATH, "embedding node-universe file")

    for model_config in MODEL_CONFIGS:
        _validate_checkpoint(model_config)
        _require_file(
            model_config.embedding_path,
            f"embedding file for {model_config.model} at dimension "
            f"{model_config.embedding_dimension}",
        )


def make_aggregated_row(model: str, budget: int, metrics: dict) -> dict:
    return {
        "model": model,
        "budget": budget,
        "evaluated_examples": metrics["num_examples"],
        "true_positives": metrics["tp"],
        "false_positives": metrics["fp"],
        "true_negatives": metrics["tn"],
        "false_negatives": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "average_visited_nodes": metrics["avg_nodes_visited"],
        "average_runtime_ms": metrics["avg_time_ms"],
    }


def run_experiment(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Run only existing evaluation helpers; no traversal logic lives here."""

    import traverse_strategies as ts
    from core.embedding_preload import preload_graph_embeddings
    from core.embeddings import STEmbedder
    from core.utils import get_model_distance_metric, load_causal_graph
    from evaluation.evaluation import (
        cleanup_cuda_cache,
        run_evaluation_loop,
        run_warmup_traversal,
    )

    with open(DATASET_PATH, encoding="utf-8") as file:
        validation_data = json.load(file)

    print(f"Loading {GRAPH_NAME} from: {GRAPH_PATH}")
    graph = load_causal_graph(
        GRAPH_PATH,
        use_inverse=False,
        progress_every=1_000_000,
        progress_label=f"{GRAPH_ID} NetworkX graph",
    )

    aggregated_rows = []
    detailed_results = []
    evaluation_errors = []

    for model_config in MODEL_CONFIGS:
        print(f"\nEvaluating budget curve: {model_config.model}")
        embedder = None
        try:
            checkpoint_path = str(model_config.checkpoint_path)
            embedder = STEmbedder(
                model_path=checkpoint_path,
                distance_metric=get_model_distance_metric(checkpoint_path),
                device=args.embedding_device,
                cache_suffix=None,
                node_universe="merged_causenet_ceg",
            )
            model_dimension = embedder.get_model_dim()
            if model_config.embedding_dimension > model_dimension:
                raise ValueError(
                    f"Configured dimension {model_config.embedding_dimension} exceeds "
                    f"{model_config.model}'s dimension {model_dimension}"
                )
            embedder.set_matryoshka_dim(model_config.embedding_dimension)

            runtime_embedding_path = embedder.get_active_cache_vectors_file()
            if (
                runtime_embedding_path.resolve()
                != model_config.embedding_path.resolve()
            ):
                raise RuntimeError(
                    f"Embedding cache path mismatch for {model_config.model}: "
                    f"validation checked {model_config.embedding_path}, but "
                    f"preload would use {runtime_embedding_path}"
                )

            indexed_graph = preload_graph_embeddings(
                embedder,
                graph,
                batch_size=args.embedding_batch_size,
                save_cache=False,
            )
            if indexed_graph is None or not embedder.has_embedding_index():
                raise RuntimeError(
                    f"Failed to load embedding index for {model_config.model}"
                )

            for budget in BUDGETS:
                runtime_config = {
                    "astar_max_visits": budget,
                    "embedding_index_min_successors": EMBEDDING_INDEX_MIN_SUCCESSORS,
                    "_indexed_graph": indexed_graph,
                }
                run_warmup_traversal(
                    validation_data,
                    graph,
                    embedder,
                    ts.astar_traverse,
                    "A*",
                    config=runtime_config,
                )
                summary = run_evaluation_loop(
                    validation_data,
                    graph,
                    embedder,
                    {"A*": ts.astar_traverse},
                    f"{model_config.model} | budget {budget} | {RUN_SUFFIX}",
                    config=runtime_config,
                )["A*"]

                aggregated_rows.append(
                    make_aggregated_row(model_config.model, budget, summary["metrics"])
                )
                detailed_results.append(
                    {
                        "model": model_config.model,
                        "budget": budget,
                        "aggregated": aggregated_rows[-1],
                        "per_query_results": summary["per_example"],
                    }
                )
        except Exception as exc:
            evaluation_errors.append(f"{model_config.model}: {exc}")
            print(f"Evaluation failed for {model_config.model}: {exc}")
        finally:
            if embedder is not None:
                del embedder
            gc.collect()
            cleanup_cuda_cache()

    if evaluation_errors:
        raise RuntimeError(
            "Budget trade-off evaluation failed; no result files were written:\n- "
            + "\n- ".join(evaluation_errors)
        )

    return validate_aggregated_results(aggregated_rows), detailed_results


def build_json_document(
    args: argparse.Namespace,
    aggregated_rows: list[dict],
    detailed_results: list[dict],
) -> dict:
    return {
        "experiment": EXPERIMENT_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": DATASET_NAME,
            "id": DATASET_ID,
            "split": "validation",
            "path": _relative(DATASET_PATH),
        },
        "graph": {
            "name": GRAPH_NAME,
            "id": GRAPH_ID,
            "path": _relative(GRAPH_PATH),
        },
        "selected_budgets": list(BUDGETS),
        "expected_combinations": EXPECTED_COMBINATIONS,
        "run_suffix": RUN_SUFFIX,
        "selection_source": _relative(VALIDATION_RESULTS_PATH),
        "model_configurations": [config.metadata() for config in MODEL_CONFIGS],
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "evaluation_command": evaluation_command(),
        "aggregated_results": aggregated_rows,
        "detailed_results": detailed_results,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter=";"))


def write_results_safely(document: dict) -> None:
    """Validate temp serializations, then replace prior files without appending."""

    aggregated_rows = validate_aggregated_results(document["aggregated_results"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_fd, csv_temp_name = tempfile.mkstemp(
        prefix="budget_tradeoff_", suffix=".csv.tmp", dir=RESULTS_DIR
    )
    json_fd, json_temp_name = tempfile.mkstemp(
        prefix="budget_tradeoff_", suffix=".json.tmp", dir=RESULTS_DIR
    )
    os.close(csv_fd)
    os.close(json_fd)
    csv_temp = Path(csv_temp_name)
    json_temp = Path(json_temp_name)

    try:
        _write_csv(csv_temp, aggregated_rows)
        with open(json_temp, "w", encoding="utf-8") as file:
            json.dump(document, file, indent=2, ensure_ascii=False)
            file.write("\n")

        with open(json_temp, encoding="utf-8") as file:
            serialized_document = json.load(file)
        csv_rows = _load_csv(csv_temp)
        json_rows = serialized_document.get("aggregated_results", [])
        if not aggregated_results_agree(csv_rows, json_rows):
            raise ValueError(
                "CSV and JSON aggregated results disagree; existing files were not replaced"
            )

        os.replace(csv_temp, RESULTS_CSV_PATH)
        os.replace(json_temp, RESULTS_JSON_PATH)
    finally:
        csv_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    validate_input_files()
    aggregated_rows, detailed_results = run_experiment(args)
    if len(aggregated_rows) != EXPECTED_COMBINATIONS:
        raise RuntimeError(
            f"Incomplete evaluation: expected {EXPECTED_COMBINATIONS} rows, "
            f"found {len(aggregated_rows)}"
        )
    document = build_json_document(args, aggregated_rows, detailed_results)
    write_results_safely(document)
    print(f"Results written to {RESULTS_CSV_PATH} and {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
