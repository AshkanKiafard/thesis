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
from optuna.samplers import GridSampler
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# code/finetune/finetune_combo.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    LitAStar,
    create_dataset,
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
        description="Grid-search the best activation and distance combination for one embedding model."
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
        help="Number of epochs per activation-distance combination"
    )

    return parser.parse_args()


def objective(trial, model_path, curr_model_name, datasets_by_distance,
              batch_size, accumulate_grad_batches, normalize, use_matryoshka,
              max_epochs, causal_graph):
    # Combo search is intentionally discrete:
    # - Activation: ReLU or GELU
    # - Distance: cosine or Euclidean
    # The learning rate is fixed here, because LR is optimized later by finetune_lr.py.
    lr = 3e-5

    # GridSampler ensures that each combination is evaluated exactly once.
    activation_func_str = trial.suggest_categorical("activation", ["relu", "gelu"])
    distance_metric_str = trial.suggest_categorical("distance", ["cosine", "euclid"])

    activation_func = parse_activation_func(activation_func_str)
    distance_metric = parse_distance_metric(distance_metric_str)

    # Datasets depend only on model and distance metric, not on activation.
    # Therefore they are precomputed once before the Optuna objective and reused here.
    train_dataset = datasets_by_distance[distance_metric_str]["train"]
    valid_dataset = datasets_by_distance[distance_metric_str]["valid"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    combo_model_str = (
        f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_{normalize_str}_{mrl_str}_combo"
    )

    print("=" * 80)
    print(f"Trial {trial.number}: {combo_model_str}")
    print(f"Fixed LR: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print("=" * 80)

    model = LitAStar(
        model_name=model_path,
        cls_activation_func=activation_func,
        cls_distance_metric=distance_metric,
        cls_normalize=normalize,
        lr=lr,
        cls_use_matryoshka=use_matryoshka,
        graph=causal_graph
    )

    early_stop = EarlyStopping(
        monitor="val/astar_cost",
        patience=3,
        mode="min"
    )

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=str(DATA_DIR / "lightning_logs" / "combo_search" / combo_model_str / SLURM_JOB_ID),
        enable_checkpointing=False,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop],
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    score = trainer.callback_metrics["val/astar_cost"].item()

    del model, trainer, train_loader, valid_loader
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

    # Keep the effective batch size fixed across models for fair comparison.
    accumulate_grad_batches = 128 // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print("Fixed learning rate: 3e-5")

    causal_graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as f:
        valid_data = json.load(f)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    datasets_dir = DATA_DIR / "datasets"
    optuna_dir = DATA_DIR / "optuna_studies"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    # Dataset creation is done before the objective because both distance datasets
    # are needed anyway for the 2 x 2 grid search.
    datasets_by_distance = {}

    for distance_metric_str in ["cosine", "euclid"]:
        distance_metric = parse_distance_metric(distance_metric_str)

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

        print(f"Total Train examples for {distance_metric_str}: {len(train_dataset)}")
        print(f"Total Val examples for {distance_metric_str}: {len(valid_dataset)}")

        datasets_by_distance[distance_metric_str] = {
            "train": train_dataset,
            "valid": valid_dataset,
        }

    # Grid search is better than random sampling here because there are only four
    # possible combinations and each one should be evaluated exactly once.
    search_space = {
        "activation": ["relu", "gelu"],
        "distance": ["cosine", "euclid"],
    }

    sampler = GridSampler(search_space)

    study_name = f"{curr_model_name}_{normalize_str}_{mrl_str}_combo_search"
    optuna_db_path = optuna_dir / f"{curr_model_name}_{normalize_str}_{mrl_str}_combo.sqlite3"

    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    # When a run is interrupted, Optuna may leave trials in RUNNING state.
    # Mark them as failed so the study can resume cleanly.
    print("Cleaning zombie trials ...")
    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]

    if running_trials:
        print(f"Found {len(running_trials)} interrupted trials (Zombies). Cleaning them up ...")
        for r_trial in running_trials:
            try:
                study.tell(r_trial.number, state=optuna.trial.TrialState.FAIL)
                print(f"Marked interrupted Trial {r_trial.number} as FAILED.")
            except Exception as e:
                print(f"Warning: Could not update status for Trial {r_trial.number}: {e}")

    # GridSampler has exactly four combinations here.
    target_trials = 4

    valid_trials = [
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ]
    current_valid_count = len(valid_trials)
    trials_to_run = target_trials - current_valid_count

    if trials_to_run > 0:
        print(f"Running combo grid search for {curr_model_name} for {trials_to_run} trials ...")
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
                max_epochs,
                causal_graph,
            ),
            n_trials=trials_to_run,
            gc_after_trial=True
        )
        print(f"Finished combo grid search for {curr_model_name}.")
    else:
        print("Combo study is already complete.")

    print("=" * 80)
    print("BEST COMBO")
    print("=" * 80)
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")