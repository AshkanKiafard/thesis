"""Generate the final-training hyperparameter and best-epoch report.

Lightning's current_epoch and checkpoint {epoch} field are zero-indexed. The
best epoch comes from strict improvements in the monitored TensorBoard metric,
then the best checkpoint, and only as a last resort stopped_epoch - patience.
"""

import argparse
import csv
import json
import math
import re
import sqlite3
from pathlib import Path

from core.constants import (
    CHECKPOINTS_DIR,
    FINAL_TRAINING_LOGS_DIR,
    HPARAM_SEARCH_STUDIES_DIR,
    LIGHTNING_MODELS_DIR,
    REPORTS_DIR,
)
from reports.common import (
    display_path,
    latex_escape,
    report_paths,
    resolve_repo_path,
    write_json,
    write_latex,
)

MONITORED_METRIC = "val/astar_cost"
CHECKPOINT_EPOCH_PATTERN = re.compile(r"(?:^|/)best-(\d+)-")
DISPLAY_NAMES = {
    "sentence-transformers/all-mpnet-base-v2": "MPNet",
    "BAAI/bge-large-en-v1.5": "BGE",
    "mixedbread-ai/mxbai-embed-large-v1": "MXBAI Embed",
    "ibm-granite/granite-embedding-english-r2": "Granite Embed",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen 0.6B",
}


def _parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_hparams(path):
    """Load Lightning's flat hparams YAML without requiring PyYAML."""
    values = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = _parse_scalar(value)
    return values


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def canonical_activation(value):
    return str(value).strip().lower()


def canonical_distance(value):
    value = str(value).strip().lower()
    return "euclid" if value in {"euclid", "euclidean"} else value


def display_activation(value):
    return {"relu": "ReLU", "gelu": "GELU"}.get(value, value)


def display_distance(value):
    return {"cosine": "Cosine", "euclid": "Euclidean"}.get(value, value)


def decode_optuna_param(value, distribution_json):
    distribution = json.loads(distribution_json)
    if distribution.get("name") == "CategoricalDistribution":
        choices = distribution["attributes"]["choices"]
        return choices[int(value)]
    return value


def read_optuna_best_trial(path, expected_study_name=None):
    """Read the best completed single-objective trial from Optuna SQLite."""
    with sqlite3.connect(path) as connection:
        if expected_study_name:
            study = connection.execute(
                "SELECT study_id, study_name FROM studies WHERE study_name = ?",
                (expected_study_name,),
            ).fetchone()
        else:
            study = connection.execute(
                "SELECT study_id, study_name FROM studies ORDER BY study_id DESC LIMIT 1"
            ).fetchone()
        if study is None:
            raise ValueError(f"No matching study in {path}")

        study_id, study_name = study
        direction = connection.execute(
            (
                "SELECT direction FROM study_directions "
                "WHERE study_id = ? AND objective = 0"
            ),
            (study_id,),
        ).fetchone()[0]
        order = "DESC" if direction == "MAXIMIZE" else "ASC"
        best_trial = connection.execute(
            (
                "SELECT t.trial_id, t.number, v.value "
                "FROM trials AS t "
                "JOIN trial_values AS v ON v.trial_id = t.trial_id "
                "WHERE t.study_id = ? AND t.state = 'COMPLETE' "
                f"ORDER BY v.value {order} LIMIT 1"
            ),
            (study_id,),
        ).fetchone()
        if best_trial is None:
            raise ValueError(f"No completed trial in {path}")

        trial_id, trial_number, objective_value = best_trial
        params = {}
        for name, value, distribution_json in connection.execute(
            (
                "SELECT param_name, param_value, distribution_json "
                "FROM trial_params WHERE trial_id = ? ORDER BY param_name"
            ),
            (trial_id,),
        ):
            params[name] = decode_optuna_param(value, distribution_json)

    return {
        "study_name": study_name,
        "storage_path": display_path(path),
        "direction": direction.lower(),
        "best_trial_number": trial_number,
        "best_objective_value": objective_value,
        "best_params": params,
    }


def read_tensorboard_history(log_dir):
    """Return all monitored validation events with their Lightning epochs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {
            "available": False,
            "error": (
                "tensorboard is not installed; install repository requirements "
                "to extract validation history"
            ),
            "events": [],
        }

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    if MONITORED_METRIC not in scalar_tags:
        return {
            "available": False,
            "error": f"missing TensorBoard scalar {MONITORED_METRIC}",
            "events": [],
        }

    epoch_by_step = {}
    if "epoch" in scalar_tags:
        epoch_by_step = {
            event.step: int(round(event.value))
            for event in accumulator.Scalars("epoch")
        }
    events = [
        {
            "epoch": epoch_by_step.get(event.step),
            "validation_ordinal": ordinal,
            "global_step": event.step,
            "value": event.value,
            "wall_time": event.wall_time,
        }
        for ordinal, event in enumerate(accumulator.Scalars(MONITORED_METRIC))
    ]
    return {
        "available": bool(events),
        "error": None,
        "events": events,
        "epoch_scalar_available": bool(epoch_by_step),
    }


def best_improvement_event(history, mode="min"):
    """Return the final event that strictly improved the running best."""
    best = None
    for event in history:
        if event["epoch"] is None:
            continue
        improved = (
            best is None
            or (mode == "min" and event["value"] < best["value"])
            or (mode == "max" and event["value"] > best["value"])
        )
        if improved:
            best = event
    return best


def epoch_from_checkpoint_name(path):
    if not path:
        return None
    match = CHECKPOINT_EPOCH_PATTERN.search(str(path).replace("\\", "/"))
    return int(match.group(1)) if match else None


def _to_json_scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_checkpoint_evidence(run_name, metadata, checkpoints_dir):
    """Inspect local Lightning checkpoints, then the recorded best filename."""
    metadata_path = metadata.get("best_checkpoint_path")
    checkpoint_dir = checkpoints_dir / run_name
    local_files = sorted(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
    metadata_name = Path(metadata_path).name if metadata_path else None
    preferred = next(
        (path for path in local_files if path.name == metadata_name),
        None,
    )
    best_file = preferred or next(
        (path for path in local_files if epoch_from_checkpoint_name(path) is not None),
        None,
    )
    last_file = checkpoint_dir / "last.ckpt"
    if not last_file.exists():
        last_file = (
            max(local_files, key=lambda path: path.stat().st_mtime)
            if local_files
            else None
        )
    loaded_epoch = None
    stopped_epoch = None
    callback_states = {}
    load_error = None

    if best_file is not None or last_file is not None:
        try:
            import torch

            if best_file is not None:
                best_checkpoint = torch.load(
                    best_file,
                    map_location="cpu",
                    weights_only=False,
                )
                loaded_epoch = best_checkpoint.get("epoch")

            stopping_checkpoint = (
                best_checkpoint
                if best_file is not None and best_file == last_file
                else torch.load(
                    last_file,
                    map_location="cpu",
                    weights_only=False,
                )
            )
            for callback_name, state in stopping_checkpoint.get("callbacks", {}).items():
                serializable_state = {
                    key: _to_json_scalar(value)
                    for key, value in state.items()
                    if key in {
                        "best_model_score",
                        "best_score",
                        "stopped_epoch",
                        "wait_count",
                    }
                }
                if serializable_state:
                    callback_states[str(callback_name)] = serializable_state
                    stopped_epoch = max(
                        stopped_epoch or 0,
                        int(serializable_state.get("stopped_epoch") or 0),
                    )
        except Exception as error:
            load_error = f"{type(error).__name__}: {error}"

    return {
        "metadata_best_checkpoint_path": metadata_path,
        "metadata_filename_epoch": epoch_from_checkpoint_name(metadata_path),
        "local_checkpoint_dir": display_path(checkpoint_dir),
        "local_checkpoint_files": [display_path(path) for path in local_files],
        "loaded_checkpoint_path": display_path(best_file) if best_file else None,
        "loaded_checkpoint_epoch": loaded_epoch,
        "stopping_checkpoint_path": (
            display_path(last_file) if last_file else None
        ),
        "stopped_epoch": stopped_epoch,
        "callback_states": callback_states,
        "load_error": load_error,
    }


def choose_best_epoch(tensorboard, checkpoint, patience, max_epochs):
    """Choose direct evidence first and EarlyStopping arithmetic last."""
    history = tensorboard["events"]
    best_event = best_improvement_event(history)
    stopped_epoch = max(
        (event["epoch"] for event in history if event["epoch"] is not None),
        default=checkpoint.get("stopped_epoch"),
    )
    checkpoint_epoch = (
        checkpoint.get("loaded_checkpoint_epoch")
        if checkpoint.get("loaded_checkpoint_epoch") is not None
        else checkpoint.get("metadata_filename_epoch")
    )

    if best_event is not None:
        best_epoch = best_event["epoch"]
        source = "tensorboard_validation_history"
    elif checkpoint_epoch is not None:
        best_epoch = int(checkpoint_epoch)
        source = "best_checkpoint_epoch"
    elif stopped_epoch is not None and patience is not None:
        best_epoch = int(stopped_epoch) - int(patience)
        source = "early_stopping_fallback"
    else:
        best_epoch = None
        source = "unavailable"

    stopped_early = (
        stopped_epoch is not None
        and max_epochs is not None
        and int(stopped_epoch) + 1 < int(max_epochs)
    )
    fallback_epoch = None
    if stopped_epoch is not None and patience is not None and stopped_early:
        fallback_epoch = int(stopped_epoch) - int(patience)

    return {
        "value": best_epoch,
        "indexing": "zero-based",
        "source": source,
        "monitored_metric": MONITORED_METRIC,
        "best_metric_value": best_event["value"] if best_event else None,
        "best_metric_global_step": best_event["global_step"] if best_event else None,
        "tensorboard_epoch": best_event["epoch"] if best_event else None,
        "checkpoint_epoch": checkpoint_epoch,
        "stopped_epoch": stopped_epoch,
        "patience": patience,
        "configured_max_epochs": max_epochs,
        "stopped_early": stopped_early,
        "fallback_epoch_if_needed": fallback_epoch,
        "direct_sources_agree": (
            best_event is None
            or checkpoint_epoch is None
            or best_event["epoch"] == checkpoint_epoch
        ),
    }


def find_metadata(models_dir, run_name):
    expected_path = models_dir / f"{run_name}_finetuned" / "training_metadata.json"
    if expected_path.exists():
        return expected_path
    for path in models_dir.glob("*/training_metadata.json"):
        if load_json(path).get("run_model_str") == run_name:
            return path
    return None


def find_study_path(studies_dir, metadata, model_name):
    study_name = metadata.get("source_optuna_study")
    if study_name:
        path = studies_dir / f"{study_name}.sqlite3"
        if path.exists():
            return path, study_name
    candidates = sorted(studies_dir.glob(f"{Path(model_name).name}_*.sqlite3"))
    if len(candidates) == 1:
        return candidates[0], None
    return None, study_name


def configurations_agree(hparams, metadata, optuna):
    checks = {}
    if optuna:
        params = optuna["best_params"]
        checks["activation"] = (
            canonical_activation(hparams["cls_activation_func"])
            == canonical_activation(params["activation"])
            == canonical_activation(metadata["activation"])
        )
        checks["distance"] = (
            canonical_distance(hparams["cls_distance_metric"])
            == canonical_distance(params["distance"])
            == canonical_distance(metadata["distance"])
        )
        checks["learning_rate"] = math.isclose(
            float(hparams["lr"]),
            float(params["lr"]),
            rel_tol=1e-12,
        ) and math.isclose(
            float(hparams["lr"]),
            float(metadata["lr"]),
            rel_tol=1e-12,
        )
    return checks


def extract_model_report(log_dir, models_dir, studies_dir, checkpoints_dir):
    hparams_path = log_dir / "hparams.yaml"
    hparams = load_hparams(hparams_path)
    run_name = log_dir.name
    model_name = hparams["model_name"]
    metadata_path = find_metadata(models_dir, run_name)
    metadata = load_json(metadata_path) if metadata_path else {}
    study_path, expected_study_name = find_study_path(
        studies_dir,
        metadata,
        model_name,
    )
    optuna = (
        read_optuna_best_trial(study_path, expected_study_name)
        if study_path
        else None
    )
    tensorboard = read_tensorboard_history(log_dir)
    checkpoint = read_checkpoint_evidence(
        run_name,
        metadata,
        checkpoints_dir,
    )
    best_epoch = choose_best_epoch(
        tensorboard=tensorboard,
        checkpoint=checkpoint,
        patience=metadata.get("patience"),
        max_epochs=metadata.get("epochs"),
    )

    activation = canonical_activation(
        hparams.get("cls_activation_func", metadata.get("activation"))
    )
    distance = canonical_distance(
        hparams.get("cls_distance_metric", metadata.get("distance"))
    )
    learning_rate = float(hparams.get("lr", metadata.get("lr")))
    verification = configurations_agree(hparams, metadata, optuna)
    verification["best_epoch"] = best_epoch["direct_sources_agree"]

    return {
        "model": {
            "display_name": DISPLAY_NAMES.get(model_name, Path(model_name).name),
            "model_name": model_name,
            "run_name": run_name,
        },
        "selected_hyperparameters": {
            "activation": activation,
            "activation_display": display_activation(activation),
            "distance_or_similarity": distance,
            "distance_or_similarity_display": display_distance(distance),
            "learning_rate": learning_rate,
            "learning_rate_times_1e5": learning_rate * 1e5,
            "batch_size": hparams.get("batch_size", metadata.get("batch_size")),
            "normalize": hparams.get("cls_normalize", metadata.get("normalize")),
            "matryoshka": hparams.get(
                "cls_use_matryoshka",
                metadata.get("matryoshka"),
            ),
        },
        "best_epoch": best_epoch,
        "validation_history": tensorboard["events"],
        "optuna": optuna,
        "verification": verification,
        "sources": {
            "lightning_log_dir": display_path(log_dir),
            "hparams_path": display_path(hparams_path),
            "hparams": hparams,
            "training_metadata_path": (
                display_path(metadata_path) if metadata_path else None
            ),
            "training_metadata": metadata,
            "tensorboard": {
                key: value
                for key, value in tensorboard.items()
                if key != "events"
            },
            "checkpoint": checkpoint,
        },
    }


def discover_final_runs(logs_dir):
    return [
        path
        for path in logs_dir.iterdir()
        if path.is_dir()
        and not path.name.endswith("_ablation")
        and (path / "hparams.yaml").exists()
    ]


def build_report(logs_dir, models_dir, studies_dir, checkpoints_dir):
    models = [
        extract_model_report(
            log_dir,
            models_dir=models_dir,
            studies_dir=studies_dir,
            checkpoints_dir=checkpoints_dir,
        )
        for log_dir in discover_final_runs(logs_dir)
    ]
    model_order = {
        model_name: index
        for index, model_name in enumerate(DISPLAY_NAMES)
    }
    models.sort(
        key=lambda row: (
            model_order.get(row["model"]["model_name"], len(model_order)),
            row["model"]["display_name"],
        )
    )
    return {
        "methodology": {
            "monitored_metric": MONITORED_METRIC,
            "monitor_mode": "min",
            "epoch_indexing": "zero-based",
            "best_epoch_definition": (
                "The final epoch that strictly improved the monitored validation "
                "metric, not the epoch at which training stopped."
            ),
            "source_precedence": [
                "TensorBoard validation history with Lightning epoch scalar",
                "best checkpoint epoch",
                "stopped_epoch - patience fallback",
            ],
            "fallback_assumptions": (
                "Validation occurs once per epoch and early stopping checks at "
                "validation end."
            ),
        },
        "models": models,
    }


def write_csv(path, models):
    fieldnames = [
        "model",
        "model_name",
        "run_name",
        "activation",
        "distance_or_similarity",
        "learning_rate",
        "learning_rate_times_1e5",
        "best_epoch_zero_based",
        "best_epoch_source",
        "best_validation_value",
        "stopped_epoch_zero_based",
        "patience",
        "fallback_epoch_if_needed",
        "optuna_study",
        "optuna_best_trial",
        "optuna_best_objective_value",
        "configuration_verified",
        "best_epoch_verified",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in models:
            model = row["model"]
            hparams = row["selected_hyperparameters"]
            best_epoch = row["best_epoch"]
            optuna = row["optuna"] or {}
            verification = row["verification"]
            writer.writerow(
                {
                    "model": model["display_name"],
                    "model_name": model["model_name"],
                    "run_name": model["run_name"],
                    "activation": hparams["activation_display"],
                    "distance_or_similarity": hparams[
                        "distance_or_similarity_display"
                    ],
                    "learning_rate": hparams["learning_rate"],
                    "learning_rate_times_1e5": hparams[
                        "learning_rate_times_1e5"
                    ],
                    "best_epoch_zero_based": best_epoch["value"],
                    "best_epoch_source": best_epoch["source"],
                    "best_validation_value": best_epoch["best_metric_value"],
                    "stopped_epoch_zero_based": best_epoch["stopped_epoch"],
                    "patience": best_epoch["patience"],
                    "fallback_epoch_if_needed": best_epoch[
                        "fallback_epoch_if_needed"
                    ],
                    "optuna_study": optuna.get("study_name"),
                    "optuna_best_trial": optuna.get("best_trial_number"),
                    "optuna_best_objective_value": optuna.get(
                        "best_objective_value"
                    ),
                    "configuration_verified": all(
                        verification.get(key, False)
                        for key in ("activation", "distance", "learning_rate")
                    ),
                    "best_epoch_verified": verification["best_epoch"],
                }
            )


def build_latex(models):
    hyperparameter_rows = []
    epoch_rows = []
    for row in models:
        model = latex_escape(row["model"]["display_name"])
        hparams = row["selected_hyperparameters"]
        lr_scaled = f"{hparams['learning_rate_times_1e5']:.3f}"
        hyperparameter_rows.append(
            "      "
            + " & ".join(
                (
                    model,
                    latex_escape(hparams["activation_display"]),
                    latex_escape(hparams["distance_or_similarity_display"]),
                    "$" + lr_scaled + "$",
                )
            )
            + r" \\"
        )
        epoch = row["best_epoch"]["value"]
        epoch_rows.append(
            f"      {model} & {epoch if epoch is not None else '--'}"
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \begin{minipage}[t]{0.70\textwidth}",
            r"    \centering",
            r"    \textbf{(a) Selected hyperparameters}\\[2pt]",
            r"    \begin{tabular}{@{}lllr@{}}",
            r"      \toprule",
            (
                r"      \textbf{Model} & \textbf{Activation} & \textbf{Metric} & "
                r"\textbf{lr [$10^{-5}$]} \\"
            ),
            r"      \midrule",
            *hyperparameter_rows,
            r"      \bottomrule",
            r"    \end{tabular}",
            r"  \end{minipage}\hfill%",
            r"  \begin{minipage}[t]{0.27\textwidth}",
            r"    \centering",
            r"    \textbf{(b) Final improvement}\\[2pt]",
            r"    \begin{tabular}{@{}lr@{}}",
            r"      \toprule",
            r"      \textbf{Model} & \textbf{Epoch} \\",
            r"      \midrule",
            *epoch_rows,
            r"      \bottomrule",
            r"    \end{tabular}",
            r"  \end{minipage}",
            (
                r"  \caption{Selected final-training hyperparameters and the "
                r"zero-indexed epoch after which no further improvement in "
                r"validation A* cost was observed.}"
            ),
            r"  \label{tab:training-hyperparameters}",
            r"\end{table}",
        ]
    )


def write_reports(report, output_dir):
    json_path, csv_path, tex_path = report_paths(
        "training_hyperparameters",
        output_dir,
    )
    write_json(json_path, report)
    write_csv(csv_path, report["models"])
    write_latex(tex_path, build_latex(report["models"]))
    return json_path, csv_path, tex_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected final-training hyperparameters and the last "
            "validation-improvement epoch from Lightning and Optuna artifacts."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=FINAL_TRAINING_LOGS_DIR,
        help="Directory containing final-training TensorBoard runs.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=LIGHTNING_MODELS_DIR,
        help="Directory containing fine-tuned model metadata.",
    )
    parser.add_argument(
        "--studies-dir",
        type=Path,
        default=HPARAM_SEARCH_STUDIES_DIR,
        help="Directory containing Optuna SQLite studies.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Directory containing Lightning checkpoints, when retained.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Report root; files are written below training_hyperparameters/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logs_dir = resolve_repo_path(args.logs_dir)
    models_dir = resolve_repo_path(args.models_dir)
    studies_dir = resolve_repo_path(args.studies_dir)
    checkpoints_dir = resolve_repo_path(args.checkpoints_dir)
    output_dir = resolve_repo_path(args.output_dir)

    for required_dir in (logs_dir, models_dir, studies_dir):
        if not required_dir.exists():
            raise FileNotFoundError(required_dir)

    report = build_report(
        logs_dir=logs_dir,
        models_dir=models_dir,
        studies_dir=studies_dir,
        checkpoints_dir=checkpoints_dir,
    )
    if not report["models"]:
        raise RuntimeError(f"No final-training runs found in {logs_dir}")

    json_path, csv_path, tex_path = write_reports(report, output_dir)
    for row in report["models"]:
        model = row["model"]["display_name"]
        best_epoch = row["best_epoch"]
        print(
            f"{model}: best epoch {best_epoch['value']} "
            f"({best_epoch['source']})"
        )
    print(f"\nWrote JSON report: {json_path}")
    print(f"Wrote CSV report:  {csv_path}")
    print(f"Wrote LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
