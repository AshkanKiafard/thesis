import argparse
import gc
import json
import os
import sys
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_lr.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    ActivationFunc,
    LitAStar,
    cleanup_zombie_trials,
    create_dataset,
    parse_activation_func,
    parse_distance_metric,
    str_to_bool,
)
from core.embeddings import STEmbedder, DistanceMetric
from core.utils import load_graph

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune one embedding model for A* heuristic learning."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path, e.g. Qwen/Qwen3-Embedding-0.6B or sentence-transformers/all-mpnet-base-v2"
    )

    parser.add_argument(
        "--activation",
        type=str,
        required=True,
        choices=["relu", "gelu"],
        help="Activation function used inside the ranking loss"
    )

    parser.add_argument(
        "--distance",
        type=str,
        required=True,
        choices=["cosine", "euclid", "euclidean"],
        help="Distance metric used for search and training"
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
        "--target-trials",
        type=int,
        default=30,
        help="Total number of Optuna LR trials to target"
    )

    parser.add_argument(
        "--lr-epochs",
        type=int,
        default=5,
        help="Number of epochs per Optuna LR-search trial"
    )

    parser.add_argument(
        "--final-epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for the final training run"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Physical batch size. Will use gradient accumulation to reach effective batch size of 128."
    )

    return parser.parse_args()


def objective(trial, model_name, model_str, train_loader, val_loader,
              f_activation_func, f_distance_metric, f_normalize, f_use_matryoshka,
              f_accumulate_grad_batches, f_lr_epochs, graph):
    # The search range is intentionally narrow because previous runs already showed
    # that useful learning rates are in this area.
    lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)

    model = LitAStar(
        model_name=model_name,
        cls_activation_func=f_activation_func,
        cls_distance_metric=f_distance_metric,
        cls_normalize=f_normalize,
        lr=lr,
        cls_use_matryoshka=f_use_matryoshka,
        graph=graph
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/astar_cost")
    early_stop = EarlyStopping(monitor="val/astar_cost", patience=3, mode="min")

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=str(DATA_DIR / "lightning_logs" / "lr_search" / model_str / SLURM_JOB_ID),
        enable_checkpointing=False,
        max_epochs=f_lr_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, pruning_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=f_accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    score = trainer.callback_metrics["val/astar_cost"].item()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return score


if __name__ == "__main__":
    args = parse_args()

    target_trials = args.target_trials
    lr_epochs = args.lr_epochs
    final_epochs = args.final_epochs
    distance_metric = parse_distance_metric(args.distance)
    activation_func = parse_activation_func(args.activation)
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    model_path = args.model
    batch_size = args.batch_size

    if batch_size > 128:
        raise ValueError("batch_size must be <= 128")
    if 128 % batch_size != 0:
        raise ValueError("batch_size must divide 128 (e.g. 128, 64, 32, 16, 8)")

    # Keep the effective batch size fixed across models for fair comparison.
    accumulate_grad_batches = 128 // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"Target LR trials: {target_trials}")
    print(f"LR search epochs: {lr_epochs}")
    print(f"Final training epochs: {final_epochs}")

    causal_graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as f:
        valid_data = json.load(f)

    curr_model_name = model_path.split("/")[-1]
    activation_func_str = "relu" if activation_func == ActivationFunc.RELU else "gelu"
    distance_metric_str = "cosine" if distance_metric == DistanceMetric.COSINE else "euclid"
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    trained_model_str = f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_{normalize_str}_{mrl_str}"

    save_path = DATA_DIR / "models" / "lightning" / f"{trained_model_str}_finetuned"
    ckpt_dir = DATA_DIR / "checkpoints" / trained_model_str

    datasets_dir = DATA_DIR / "datasets"
    models_dir = DATA_DIR / "models" / "lightning"
    checkpoints_dir = DATA_DIR / "checkpoints"
    optuna_root_dir = DATA_DIR / "optuna_studies"
    optuna_lr_dir = optuna_root_dir / "lr"

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    optuna_lr_dir.mkdir(parents=True, exist_ok=True)

    # Skip everything if the final exported SentenceTransformer already exists.
    if save_path.exists():
        print(f"Model already exists at: {save_path}")
        raise SystemExit(0)

    print(f"Optimization starting for: {trained_model_str} - Slurm Job ID: {SLURM_JOB_ID}")

    dataset_suffix = f"{curr_model_name.replace('/', '_')}_{distance_metric_str}"
    train_ds_path = datasets_dir / f"train_{dataset_suffix}"
    valid_ds_path = datasets_dir / f"valid_{dataset_suffix}"

    train_exists = train_ds_path.exists()
    valid_exists = valid_ds_path.exists()

    main_embedder = None
    if not train_exists or not valid_exists:
        print(f"Initializing Embedder for {curr_model_name} ...")
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

    main_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    main_valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    pruner = optuna.pruners.MedianPruner()

    # The LR study name includes the number of LR trials and LR epochs.
    # This prevents reusing an old LR study that used different search settings.
    study_name = (
        f"{trained_model_str}_"
        f"lr{target_trials}trials_{lr_epochs}epochs_optimization"
    )

    optuna_db_path = optuna_lr_dir / (
        f"{trained_model_str}_"
        f"lr{target_trials}trials_{lr_epochs}epochs.sqlite3"
    )

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )

    cleanup_zombie_trials(study, "LR")

    # Count both COMPLETE and PRUNED as already-attempted useful trials.
    valid_trials = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    current_valid_count = len(valid_trials)
    trials_to_run = target_trials - current_valid_count

    if trials_to_run > 0:
        print(f"Running study for {trained_model_str} for {trials_to_run} trials ...")
        study.optimize(
            lambda l_trial: objective(
                l_trial,
                model_path,
                trained_model_str,
                main_train_loader,
                main_valid_loader,
                activation_func,
                distance_metric,
                normalize,
                use_matryoshka,
                accumulate_grad_batches,
                lr_epochs,
                causal_graph
            ),
            n_trials=trials_to_run,
            gc_after_trial=True
        )
        print(f"Finished study for {trained_model_str} for {trials_to_run} trials.")

        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Study is already complete.")

    best_lr = study.best_params["lr"]
    print(f"Training {trained_model_str} with LR={best_lr} ...")

    final_model = LitAStar(
        model_path,
        activation_func,
        distance_metric,
        normalize,
        best_lr,
        use_matryoshka,
        causal_graph
    )

    logger = TensorBoardLogger(
        str(DATA_DIR / "tb_logs"),
        name=trained_model_str,
        version=(
            f"{SLURM_JOB_ID}_{activation_func_str}_{distance_metric_str}_"
            f"{normalize_str}_lr{target_trials}trials_{lr_epochs}epochs_final{final_epochs}epochs"
        )
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val/astar_cost:.4f}",
        monitor="val/astar_cost",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    # Final training gets a larger patience than the Optuna search stage.
    early_stop_callback = EarlyStopping(monitor="val/astar_cost", patience=10, mode="min")

    main_trainer = pl.Trainer(
        max_epochs=final_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        default_root_dir=str(DATA_DIR / "lightning_logs" / "final_training" / trained_model_str / SLURM_JOB_ID),
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    ckpt_path = None
    if ckpt_dir.exists():
        checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

        if checkpoint_files:
            # Resume from the most recently modified checkpoint if one exists.
            checkpoint_files.sort(
                key=lambda x: os.path.getmtime(ckpt_dir / x),
                reverse=True
            )
            ckpt_path = str(ckpt_dir / checkpoint_files[0])
            print(f"Found checkpoint! Resuming from: {ckpt_path}")
        else:
            print("No .ckpt files found in directory. Training from scratch.")
    else:
        print("Checkpoint directory not found. Training from scratch.")

    # Needed so Lightning/Torch can safely deserialize these custom enum values.
    torch.serialization.add_safe_globals([ActivationFunc, DistanceMetric])

    main_trainer.fit(final_model, main_train_loader, main_valid_loader, ckpt_path=ckpt_path)

    print(f"Loading best model from checkpoint: {checkpoint_callback.best_model_path}")
    best_model = LitAStar.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        graph=causal_graph
    )

    print(f"Saving best SentenceTransformer model to: {save_path}")
    best_model.embedding_model.save(str(save_path))

    # Try to free GPU/CPU memory
    del final_model, best_model, main_trainer, study
    gc.collect()
    torch.cuda.empty_cache()