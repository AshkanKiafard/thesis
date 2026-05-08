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
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_hparam_search.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    LitAStar,
    cleanup_zombie_trials,
    create_dataset,
    find_latest_hparam_study,
    parse_activation_func,
    parse_distance_metric,
    str_to_bool,
)
from core.embeddings import STEmbedder
from core.utils import load_graph

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize activation function, distance metric, and learning rate with Optuna."
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
        default=10,
        help="Number of epochs per Optuna trial"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of validation epochs without improvement before early stopping"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials for activation, distance metric, and learning rate search"
    )

    return parser.parse_args()


def run_training_trial(f_trial, f_model_path, f_curr_model_name, f_train_dataset, f_valid_dataset,
                       f_batch_size, f_accumulate_grad_batches, f_normalize,
                       f_use_matryoshka, f_max_epochs, f_patience, f_causal_graph,
                       f_activation_func_str, f_distance_metric_str, f_lr,
                       f_log_group, f_run_suffix):
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

    print("=" * 80)
    print(f"Run: {f_run_model_str}")
    print(f"LR: {f_lr}")
    print(f"Batch size: {f_batch_size}")
    print(f"Accumulate grad batches: {f_accumulate_grad_batches}")
    print(f"Max epochs: {f_max_epochs}")
    print(f"Patience: {f_patience}")
    print("=" * 80)

    model = LitAStar(
        model_name=f_model_path,
        cls_activation_func=f_activation_func,
        cls_distance_metric=f_distance_metric,
        cls_normalize=f_normalize,
        lr=f_lr,
        cls_use_matryoshka=f_use_matryoshka,
        graph=f_causal_graph
    )

    early_stop = EarlyStopping(
        monitor="val/astar_cost",
        patience=f_patience,
        mode="min"
    )

    # Optuna pruning stops clearly bad trials early based on intermediate
    # validation performance. This reduces wasted compute during the search.
    pruning_callback = PyTorchLightningPruningCallback(
        f_trial,
        monitor="val/astar_cost",
    )

    logger = TensorBoardLogger(
        save_dir=str(DATA_DIR / "lightning_logs"),
        name=f_log_group,
        version=f_run_model_str,
    )

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        max_epochs=f_max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, pruning_callback],
        num_sanity_val_steps=0,
        accumulate_grad_batches=f_accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    pruning_callback.check_pruned()

    score = trainer.callback_metrics["val/astar_cost"].item()

    del model, trainer, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

    return score


def objective(trial, f_model_path, f_curr_model_name, f_datasets_by_distance,
              f_batch_size, f_accumulate_grad_batches, f_normalize,
              f_use_matryoshka, f_epochs, f_patience, f_causal_graph):
    # Activation, distance metric, and learning rate are optimized jointly.
    # This is the standard Optuna setup: all relevant hyperparameters are part
    # of the same search space instead of being optimized in separate stages.
    f_activation_func_str = trial.suggest_categorical("activation", ["relu", "gelu"])
    f_distance_metric_str = trial.suggest_categorical("distance", ["cosine", "euclid"])

    # The LR range is intentionally narrow because previous runs already showed
    # that useful learning rates are in this area.
    lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)

    # Datasets depend only on model and distance metric, not on activation or LR.
    # Therefore, they are precomputed once before the Optuna objective and reused here.
    f_train_dataset = f_datasets_by_distance[f_distance_metric_str]["train"]
    f_valid_dataset = f_datasets_by_distance[f_distance_metric_str]["valid"]

    print("=" * 80)
    print(f"Trial {trial.number}")
    print(f"Activation: {f_activation_func_str}")
    print(f"Distance: {f_distance_metric_str}")
    print(f"LR: {lr}")
    print("=" * 80)

    return run_training_trial(
        f_trial=trial,
        f_model_path=f_model_path,
        f_curr_model_name=f_curr_model_name,
        f_train_dataset=f_train_dataset,
        f_valid_dataset=f_valid_dataset,
        f_batch_size=f_batch_size,
        f_accumulate_grad_batches=f_accumulate_grad_batches,
        f_normalize=f_normalize,
        f_use_matryoshka=f_use_matryoshka,
        f_max_epochs=f_epochs,
        f_patience=f_patience,
        f_causal_graph=f_causal_graph,
        f_activation_func_str=f_activation_func_str,
        f_distance_metric_str=f_distance_metric_str,
        f_lr=lr,
        f_log_group="hparam_search",
        f_run_suffix=f"trial_{trial.number}",
    )


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model
    batch_size = args.batch_size
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    epochs = args.epochs
    patience = args.patience
    target_trials = args.trials

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
    print(f"Search epochs per trial: {epochs}")
    print(f"Search patience: {patience}")
    print(f"Optuna trials: {target_trials}")

    causal_graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as valid_file:
        valid_data = json.load(valid_file)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    datasets_dir = DATA_DIR / "datasets"

    optuna_root_dir = DATA_DIR / "optuna_studies"
    optuna_hparam_search_dir = optuna_root_dir / "hparam_search"
    optuna_hparam_search_dir.mkdir(parents=True, exist_ok=True)

    # Dataset creation is done before the objective because both distance datasets
    # are needed anyway for the search.
    datasets_by_distance = {}

    for curr_distance_metric_str in ["cosine", "euclid"]:
        curr_distance_metric = parse_distance_metric(curr_distance_metric_str)

        dataset_suffix = f"{curr_model_name.replace('/', '_')}_{curr_distance_metric_str}"
        train_ds_path = datasets_dir / f"train_{dataset_suffix}"
        valid_ds_path = datasets_dir / f"valid_{dataset_suffix}"

        train_exists = train_ds_path.exists()
        valid_exists = valid_ds_path.exists()

        main_embedder = None
        if not train_exists or not valid_exists:
            print(f"Initializing Embedder for {curr_model_name} with {curr_distance_metric_str} distance ...")
            main_embedder = STEmbedder(model_path, curr_distance_metric)

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

        print(f"Total Train examples for {curr_distance_metric_str}: {len(train_dataset)}")
        print(f"Total Val examples for {curr_distance_metric_str}: {len(valid_dataset)}")

        datasets_by_distance[curr_distance_metric_str] = {
            "train": train_dataset,
            "valid": valid_dataset,
        }

    optuna.logging.set_verbosity(optuna.logging.INFO)

    sampler = TPESampler(seed=42)

    # MedianPruner compares the current trial against the median performance
    # of previous completed trials and stops clearly underperforming trials early.
    # A few startup trials are allowed to finish completely before pruning begins.
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
    )

    # Reuse the latest matching hparam search study if one already exists.
    # This allows continuing older studies even if the naming scheme changes.
    latest_study = find_latest_hparam_study(
        optuna_hparam_search_dir=optuna_hparam_search_dir,
        curr_model_name=curr_model_name,
        normalize_str=normalize_str,
        mrl_str=mrl_str,
    )

    if latest_study is not None:
        optuna_db_path, study_name = latest_study

        print("=" * 80)
        print("REUSING EXISTING HPARAM SEARCH STUDY")
        print("=" * 80)
        print(f"Study file: {optuna_db_path}")
        print(f"Study name: {study_name}")
        print("=" * 80)
    else:
        # Create a new Optuna hparam search study only if no compatible
        # previous study exists for this model/configuration.
        # The study name still includes the search settings for readability
        # and easier manual inspection of study files.
        study_name = (
            f"{curr_model_name}_{normalize_str}_{mrl_str}_"
            f"{target_trials}trials_{epochs}epochs_search_{patience}patience"
        )

        optuna_db_path = optuna_hparam_search_dir / f"{study_name}.sqlite3"

        print("=" * 80)
        print("CREATING NEW HPARAM SEARCH STUDY")
        print("=" * 80)
        print(f"Study file: {optuna_db_path}")
        print(f"Study name: {study_name}")
        print("=" * 80)

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    cleanup_zombie_trials(study, "hparam_search")

    completed_trials = [
        t for t in study.trials
        if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
    ]

    trials_to_run = target_trials - len(completed_trials)

    if trials_to_run > 0:
        print(f"Running hparam search study for {curr_model_name} for {trials_to_run} trials ...")
        study.optimize(
            lambda l_trial: objective(
                l_trial,
                model_path,
                curr_model_name,
                datasets_by_distance,
                batch_size,
                accumulate_grad_batches,
                normalize,
                use_matryoshka,
                epochs,
                patience,
                causal_graph,
            ),
            n_trials=trials_to_run,
            gc_after_trial=True
        )
        print(f"Finished hparam search study for {curr_model_name}.")
    else:
        print("Hparam search study is already complete.")

    print("=" * 80)
    print("BEST HPARAMS")
    print("=" * 80)
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")