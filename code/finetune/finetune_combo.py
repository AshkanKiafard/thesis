import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_combo.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    ActivationFunc,
    LitAStar,
    create_dataset,
    str_to_bool,
)
from core.embeddings import STEmbedder, DistanceMetric
from core.utils import load_graph

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ReLU/GELU x cosine/euclidean combinations for one embedding model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path, e.g. Qwen/Qwen3-Embedding-0.6B or sentence-transformers/all-mpnet-base-v2"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Physical batch size. Will use gradient accumulation to reach effective batch size of 128."
    )

    parser.add_argument(
        "--normalize",
        type=str_to_bool,
        default=False,
        help="Whether to L2-normalize embeddings before distance computation (default: false)"
    )

    parser.add_argument(
        "--matryoshka",
        type=str_to_bool,
        default=True,
        help="Whether to use Matryoshka slicing during training (default: true)"
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Number of epochs per combination"
    )

    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True
    )


def run_combo(model_name, model_str, train_loader, val_loader,
              activation_func, distance_metric, normalize, use_matryoshka,
              graph, accumulate_grad_batches, max_epochs):
    lr = 3e-5

    model = LitAStar(
        model_name=model_name,
        cls_activation_func=activation_func,
        cls_distance_metric=distance_metric,
        cls_normalize=normalize,
        lr=lr,
        cls_use_matryoshka=use_matryoshka,
        graph=graph
    )

    early_stop = EarlyStopping(
        monitor="val/astar_cost",
        patience=3,
        mode="min"
    )

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=str(DATA_DIR / "lightning_logs" / "combo_search" / model_str / SLURM_JOB_ID),
        enable_checkpointing=False,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop],
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    score = trainer.callback_metrics["val/astar_cost"].item()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return score


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model
    batch_size = args.batch_size
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    max_epochs = args.max_epochs

    if batch_size > 128:
        raise ValueError("batch_size must be <= 128")
    if 128 % batch_size != 0:
        raise ValueError("batch_size must divide 128 (e.g. 128, 64, 32, 16, 8)")

    accumulate_grad_batches = 128 // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"Fixed learning rate: 3e-5")

    causal_graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as f:
        valid_data = json.load(f)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    datasets_dir = DATA_DIR / "datasets"
    combo_dir = DATA_DIR / "combo_search"
    combo_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        (ActivationFunc.RELU, DistanceMetric.COSINE, "relu", "cosine"),
        (ActivationFunc.RELU, DistanceMetric.EUCLIDEAN, "relu", "euclid"),
        (ActivationFunc.GELU, DistanceMetric.COSINE, "gelu", "cosine"),
        (ActivationFunc.GELU, DistanceMetric.EUCLIDEAN, "gelu", "euclid"),
    ]

    results = []

    for activation_func, distance_metric, activation_str, distance_str in combos:
        print("=" * 80)
        print(f"Testing combo: {activation_str} + {distance_str}")
        print("=" * 80)

        dataset_suffix = f"{curr_model_name.replace('/', '_')}_{distance_str}"
        train_ds_path = datasets_dir / f"train_{dataset_suffix}"
        valid_ds_path = datasets_dir / f"valid_{dataset_suffix}"

        train_exists = train_ds_path.exists()
        valid_exists = valid_ds_path.exists()

        main_embedder = None
        if not train_exists or not valid_exists:
            print(f"Initializing Embedder for {curr_model_name} with {distance_str} distance ...")
            main_embedder = STEmbedder(model_path, distance_metric)

        if train_exists:
            print(f"Loading cached TRAIN dataset: {train_ds_path}")
            train_dataset = load_from_disk(str(train_ds_path))
        else:
            print(f"Creating TRAIN dataset: {train_ds_path}")
            train_dataset = create_dataset(train_data, causal_graph, main_embedder)
            train_dataset.save_to_disk(str(train_ds_path))
            print(f"TRAIN Dataset saved to: {train_ds_path}")

        if valid_exists:
            print(f"Loading cached VAL dataset: {valid_ds_path}")
            valid_dataset = load_from_disk(str(valid_ds_path))
        else:
            print(f"Creating VAL dataset: {valid_ds_path}")
            valid_dataset = create_dataset(valid_data, causal_graph, main_embedder)
            valid_dataset.save_to_disk(str(valid_ds_path))
            print(f"VAL Dataset saved to: {valid_ds_path}")

        if main_embedder:
            del main_embedder
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Total Train examples: {len(train_dataset)}")
        print(f"Total Val examples: {len(valid_dataset)}")

        train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(valid_dataset, batch_size=batch_size, shuffle=False)

        combo_model_str = (
            f"{curr_model_name}_{activation_str}_{distance_str}_{normalize_str}_{mrl_str}_combo"
        )

        score = run_combo(
            model_name=model_path,
            model_str=combo_model_str,
            train_loader=train_loader,
            val_loader=val_loader,
            activation_func=activation_func,
            distance_metric=distance_metric,
            normalize=normalize,
            use_matryoshka=use_matryoshka,
            graph=causal_graph,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs
        )

        result = {
            "model": curr_model_name,
            "activation": activation_str,
            "distance": distance_str,
            "normalize": normalize,
            "matryoshka": use_matryoshka,
            "lr": 3e-5,
            "batch_size": batch_size,
            "accumulate_grad_batches": accumulate_grad_batches,
            "effective_batch_size": batch_size * accumulate_grad_batches,
            "max_epochs": max_epochs,
            "val_astar_cost": score,
        }

        results.append(result)

        result_path = combo_dir / f"{curr_model_name}_{normalize_str}_{mrl_str}_combo_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"Combo result: {result}")
        print(f"Saved partial results to: {result_path}")

        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    best = min(results, key=lambda x: x["val_astar_cost"])

    print("=" * 80)
    print("BEST COMBO")
    print("=" * 80)
    print(json.dumps(best, indent=2))