import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_ablation.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    TARGET_EFFECTIVE_BATCH_SIZE,
    load_or_create_datasets,
    load_hparams,
    str_to_bool,
    validate_training_args,
)
from finetune.finetune_best import train
from core.utils import (
    canonical_activation,
    canonical_distance,
    load_causal_graph,
    parse_activation_func,
    parse_distance_metric,
)

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")


ALL_ABLATION_COMBOS = (
    ("relu", "euclid"),
    ("relu", "cosine"),
    ("gelu", "euclid"),
    ("gelu", "cosine"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train activation/distance ablations for one model using the best "
            "Optuna learning rate from the latest hparam search study."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path, e.g. Qwen/Qwen3-Embedding-0.6B or sentence-transformers/all-mpnet-base-v2",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Physical batch size. Will use gradient accumulation to reach effective batch size of 128.",
    )

    parser.add_argument(
        "--normalize",
        type=str_to_bool,
        default=False,
        help="Whether to L2-normalize embeddings before distance computation (default: false)",
    )

    parser.add_argument(
        "--matryoshka",
        type=str_to_bool,
        default=True,
        help="Whether to use Matryoshka slicing during training (default: true)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of ablation training epochs",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of validation epochs without improvement before early stopping",
    )

    parser.add_argument(
        "--run-suffix",
        type=str,
        default="v3",
        help="Suffix for checkpoint/log/model directory names",
    )

    return parser.parse_args()


def validate_ablation_space():
    for activation, distance in ALL_ABLATION_COMBOS:
        parse_activation_func(activation)
        parse_distance_metric(distance)


def build_ablation_run_suffix(run_suffix):
    if run_suffix.endswith("_ablation"):
        return run_suffix

    return f"{run_suffix}_ablation"


def build_run_model_str(
    curr_model_name,
    activation,
    distance,
    normalize,
    use_matryoshka,
    run_suffix,
):
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    return (
        f"{curr_model_name}_{activation}_{distance}_"
        f"{normalize_str}_{mrl_str}_{run_suffix}"
    )


def get_ablation_combos(base_activation, base_distance):
    base_combo = (base_activation, base_distance)

    if base_combo not in ALL_ABLATION_COMBOS:
        expected = ", ".join(
            f"{activation}+{distance}"
            for activation, distance in ALL_ABLATION_COMBOS
        )
        raise ValueError(
            "Optuna selected unsupported activation/distance combination: "
            f"{base_activation}+{base_distance}. Expected one of: {expected}"
        )

    return [
        combo for combo in ALL_ABLATION_COMBOS
        if combo != base_combo
    ]


def augment_ablation_metadata(
    final_model_dir,
    base_activation,
    base_distance,
    activation,
    distance,
    lr,
    source_study_name,
    source_study_path,
):
    metadata_path = final_model_dir / "training_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing training metadata after export: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    metadata.update(
        {
            "ablation": True,
            "ablation_base_activation": base_activation,
            "ablation_base_distance": base_distance,
            "activation": activation,
            "distance": distance,
            "lr": lr,
            "source_optuna_study": source_study_name,
            "source_optuna_study_path": str(source_study_path),
        }
    )

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def main():
    args = parse_args()

    model_path = args.model
    batch_size = args.batch_size
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    epochs = args.epochs
    patience = args.patience
    run_suffix = args.run_suffix

    validate_training_args(batch_size, epochs, patience)
    validate_ablation_space()

    # Keep the effective batch size fixed across models for fair comparison.
    accumulate_grad_batches = TARGET_EFFECTIVE_BATCH_SIZE // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"Ablation training epochs: {epochs}")
    print(f"Ablation training patience: {patience}")
    print(f"Run suffix: {run_suffix}")
    print(f"SLURM job ID: {SLURM_JOB_ID}")

    causal_graph = load_causal_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as valid_file:
        valid_data = json.load(valid_file)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    optuna_hparam_search_dir = DATA_DIR / "optuna_studies" / "hparam_search"

    best_params, latest_study_path, source_study_name = load_hparams(
        optuna_hparam_search_dir=optuna_hparam_search_dir,
        curr_model_name=curr_model_name,
        normalize_str=normalize_str,
        mrl_str=mrl_str,
        run_suffix=run_suffix,
    )

    base_activation = canonical_activation(best_params["activation"])
    base_distance = canonical_distance(best_params["distance"])
    lr = float(best_params["lr"])

    if lr <= 0:
        raise ValueError(f"Optuna-selected lr must be > 0, got: {lr}")

    ablation_combos = get_ablation_combos(base_activation, base_distance)
    ablation_run_suffix = build_ablation_run_suffix(run_suffix)

    print("=" * 80)
    print("ABLATION SETUP")
    print("=" * 80)
    print(f"Selected Optuna combo: {base_activation} + {base_distance}")
    print(f"Fixed LR: {lr}")
    print(f"Source Optuna study: {source_study_name}")
    print(f"Source Optuna study file: {latest_study_path}")
    print("Ablation combos to train:")
    for activation, distance in ablation_combos:
        print(f"  - {activation} + {distance}")
    print(f"Ablation run suffix: {ablation_run_suffix}")
    print("=" * 80)

    datasets_by_distance = {}
    for distance in sorted({distance for _, distance in ablation_combos}):
        print("=" * 80)
        print(f"Preparing datasets for ablation distance: {distance}")
        print("=" * 80)
        train_dataset, valid_dataset = load_or_create_datasets(
            model_path=model_path,
            curr_model_name=curr_model_name,
            distance_metric_str=distance,
            train_data=train_data,
            valid_data=valid_data,
            causal_graph=causal_graph,
            datasets_dir=DATA_DIR / "datasets",
        )
        datasets_by_distance[distance] = {
            "train": train_dataset,
            "valid": valid_dataset,
        }

    trained_model_dirs = []

    try:
        for combo_index, (activation, distance) in enumerate(ablation_combos, start=1):
            run_model_str = build_run_model_str(
                curr_model_name=curr_model_name,
                activation=activation,
                distance=distance,
                normalize=normalize,
                use_matryoshka=use_matryoshka,
                run_suffix=ablation_run_suffix,
            )
            checkpoint_dir = DATA_DIR / "checkpoints" / run_model_str
            final_model_dir = DATA_DIR / "models" / "lightning" / f"{run_model_str}_finetuned"

            print("=" * 80)
            print(
                f"START ABLATION {combo_index}/{len(ablation_combos)}: "
                f"{activation} + {distance}"
            )
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Export directory: {final_model_dir}")
            print("=" * 80)

            train_dataset = datasets_by_distance[distance]["train"]
            valid_dataset = datasets_by_distance[distance]["valid"]

            train(
                f_model_path=model_path,
                f_curr_model_name=curr_model_name,
                f_train_dataset=train_dataset,
                f_valid_dataset=valid_dataset,
                f_batch_size=batch_size,
                f_accumulate_grad_batches=accumulate_grad_batches,
                f_normalize=normalize,
                f_use_matryoshka=use_matryoshka,
                f_max_epochs=epochs,
                f_patience=patience,
                f_run_suffix=ablation_run_suffix,
                f_causal_graph=causal_graph,
                f_activation_func_str=activation,
                f_distance_metric_str=distance,
                f_lr=lr,
                f_source_study_name=source_study_name,
            )

            augment_ablation_metadata(
                final_model_dir=final_model_dir,
                base_activation=base_activation,
                base_distance=base_distance,
                activation=activation,
                distance=distance,
                lr=lr,
                source_study_name=source_study_name,
                source_study_path=latest_study_path,
            )

            trained_model_dirs.append(final_model_dir)

            print("=" * 80)
            print(f"END ABLATION {combo_index}/{len(ablation_combos)}: {activation} + {distance}")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Export directory: {final_model_dir}")
            print("=" * 80)

            del train_dataset, valid_dataset
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del datasets_by_distance
        gc.collect()
        torch.cuda.empty_cache()

    print("=" * 80)
    print("ABLATION TRAINING SUMMARY")
    print("=" * 80)
    for model_dir in trained_model_dirs:
        print(f"- {model_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
