import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_best.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    LitAStar,
    create_dataset,
    load_hparams,
    str_to_bool,
)
from core.embeddings import STEmbedder
from core.utils import load_causal_graph, parse_activation_func, parse_distance_metric

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train final model using the best hyperparameters from the latest Optuna hparam search study."
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
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of final training epochs"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of validation epochs without improvement before early stopping"
    )

    parser.add_argument(
        "--run-suffix",
        type=str,
        default="v3",
        help="Suffix for checkpoint/log/model directory names"
    )

    return parser.parse_args()


def train(f_model_path, f_curr_model_name, f_train_dataset, f_valid_dataset,
          f_batch_size, f_accumulate_grad_batches, f_normalize,
          f_use_matryoshka, f_max_epochs, f_patience, f_run_suffix,
          f_causal_graph, f_activation_func_str, f_distance_metric_str, f_lr,
          f_source_study_name):
    f_activation_func = parse_activation_func(f_activation_func_str)
    f_distance_metric = parse_distance_metric(f_distance_metric_str)

    train_loader = DataLoader(
        f_train_dataset,
        batch_size=f_batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    valid_loader = DataLoader(
        f_valid_dataset,
        batch_size=f_batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    f_normalize_str = "norm" if f_normalize else "nonorm"
    f_mrl_str = "matryoshka" if f_use_matryoshka else "single"

    f_run_model_str = (
        f"{f_curr_model_name}_{f_activation_func_str}_{f_distance_metric_str}_"
        f"{f_normalize_str}_{f_mrl_str}_{f_run_suffix}"
    )

    checkpoint_dir = DATA_DIR / "checkpoints" / f_run_model_str
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = checkpoint_dir / "last.ckpt"
    ckpt_path = str(last_checkpoint_path) if last_checkpoint_path.exists() else None

    print("=" * 80)
    print(f"Final training run: {f_run_model_str}")
    print(f"Run suffix: {f_run_suffix}")
    print(f"Source Optuna study: {f_source_study_name}")
    print(f"Activation: {f_activation_func_str}")
    print(f"Distance: {f_distance_metric_str}")
    print(f"LR: {f_lr}")
    print(f"Batch size: {f_batch_size}")
    print(f"Accumulate grad batches: {f_accumulate_grad_batches}")
    print(f"Effective batch size: {f_batch_size * f_accumulate_grad_batches}")
    print(f"Max epochs: {f_max_epochs}")
    print(f"Patience: {f_patience}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found. Starting from scratch.")

    print("=" * 80)

    model = LitAStar(
        model_name=f_model_path,
        cls_activation_func=f_activation_func,
        cls_distance_metric=f_distance_metric,
        cls_normalize=f_normalize,
        lr=f_lr,
        cls_use_matryoshka=f_use_matryoshka,
        graph=f_causal_graph,
        batch_size=f_batch_size,
    )

    early_stop = EarlyStopping(
        monitor="val/astar_cost",
        patience=f_patience,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val/astar_cost:.4f}",
        monitor="val/astar_cost",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    logger = TensorBoardLogger(
        save_dir=str(DATA_DIR / "lightning_logs"),
        name="final_training",
        version=f_run_model_str,
    )

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        max_epochs=f_max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, checkpoint_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=f_accumulate_grad_batches,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=ckpt_path,
    )

    final_model_dir = DATA_DIR / "models" / "lightning" / f"{f_run_model_str}_finetuned"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)
    print(f"Final model directory: {final_model_dir}")

    best_checkpoint_path = checkpoint_callback.best_model_path

    if not best_checkpoint_path:
        best_checkpoints = sorted(
            checkpoint_dir.glob("best-*.ckpt"),
            key=lambda checkpoint_file: checkpoint_file.stat().st_mtime,
            reverse=True,
        )

        if best_checkpoints:
            best_checkpoint_path = str(best_checkpoints[0])
        elif last_checkpoint_path.exists():
            best_checkpoint_path = str(last_checkpoint_path)
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading best checkpoint for export: {best_checkpoint_path}")

    best_model = LitAStar.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model_name=f_model_path,
        cls_activation_func=f_activation_func,
        cls_distance_metric=f_distance_metric,
        cls_normalize=f_normalize,
        lr=f_lr,
        cls_use_matryoshka=f_use_matryoshka,
        graph=f_causal_graph,
    )

    best_model.embedding_model.save(str(final_model_dir))

    metadata = {
        "model_path": f_model_path,
        "run_model_str": f_run_model_str,
        "run_suffix": f_run_suffix,
        "source_optuna_study": f_source_study_name,
        "activation": f_activation_func_str,
        "distance": f_distance_metric_str,
        "lr": f_lr,
        "normalize": f_normalize,
        "matryoshka": f_use_matryoshka,
        "epochs": f_max_epochs,
        "patience": f_patience,
        "batch_size": f_batch_size,
        "accumulate_grad_batches": f_accumulate_grad_batches,
        "effective_batch_size": f_batch_size * f_accumulate_grad_batches,
        "checkpoint_dir": str(checkpoint_dir),
        "resumed_from_checkpoint": ckpt_path,
        "best_checkpoint_path": best_checkpoint_path,
        "best_val_astar_cost": checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score is not None else None,
    }

    with open(final_model_dir / "training_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best val/astar_cost: {metadata['best_val_astar_cost']}")
    print("=" * 80)

    del model, best_model, trainer, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model
    batch_size = args.batch_size
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    epochs = args.epochs
    patience = args.patience
    run_suffix = args.run_suffix

    if batch_size > 128:
        raise ValueError("batch_size must be <= 128")
    if 128 % batch_size != 0:
        raise ValueError("batch_size must divide 128 (e.g. 128, 64, 32, 16, 8)")
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if patience >= epochs:
        print("Warning: patience >= epochs, so early stopping will probably not trigger.")

    # Keep the effective batch size fixed across models for fair comparison.
    accumulate_grad_batches = 128 // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"Final training epochs: {epochs}")
    print(f"Final training patience: {patience}")
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

    activation_func_str = best_params["activation"]
    distance_metric_str = best_params["distance"]
    lr = best_params["lr"]

    distance_metric = parse_distance_metric(distance_metric_str)

    datasets_dir = DATA_DIR / "datasets"
    dataset_suffix = f"{curr_model_name.replace('/', '_')}_{distance_metric_str}"

    train_ds_path = datasets_dir / f"train_{dataset_suffix}"
    valid_ds_path = datasets_dir / f"valid_{dataset_suffix}"

    train_exists = train_ds_path.exists()
    valid_exists = valid_ds_path.exists()

    main_embedder = None
    if not train_exists or not valid_exists:
        print(f"Initializing Embedder for {curr_model_name} with {distance_metric_str} distance ...")
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
        f_run_suffix=run_suffix,
        f_causal_graph=causal_graph,
        f_activation_func_str=activation_func_str,
        f_distance_metric_str=distance_metric_str,
        f_lr=lr,
        f_source_study_name=source_study_name,
    )
