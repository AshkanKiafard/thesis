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
DATA_DIR = REPO_ROOT / "code" / "data"

# Make code/ importable when this script is executed from code/finetune/.
sys.path.append(str(REPO_ROOT / "code"))

from finetune.astar_training_core import (
    LitAStar,
    cleanup_zombie_trials,
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
        description="Find the best LR for each activation-distance combination, then grid-search the best combination."
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
        "--lr-epochs",
        type=int,
        default=5,
        help="Number of epochs per LR-search trial"
    )

    parser.add_argument(
        "--combo-epochs",
        type=int,
        default=10,
        help="Number of epochs per activation-distance combination after the best LR was found"
    )

    parser.add_argument(
        "--lr-trials",
        type=int,
        default=10,
        help="Number of LR-search trials per activation-distance combination"
    )

    return parser.parse_args()


def run_training_trial(f_model_path, f_curr_model_name, f_train_dataset, f_valid_dataset,
                       f_batch_size, f_accumulate_grad_batches, f_normalize,
                       f_use_matryoshka, f_max_epochs, f_causal_graph,
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
        patience=3 if f_log_group == "lr_search" else 5,
        mode="min"
    )

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=str(DATA_DIR / "lightning_logs" / f_log_group / f_run_model_str / SLURM_JOB_ID),
        enable_checkpointing=False,
        max_epochs=f_max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop],
        num_sanity_val_steps=0,
        accumulate_grad_batches=f_accumulate_grad_batches,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    score = trainer.callback_metrics["val/astar_cost"].item()

    del model, trainer, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

    return score


def lr_objective(trial, f_model_path, f_curr_model_name, f_train_dataset, f_valid_dataset,
                 f_batch_size, f_accumulate_grad_batches, f_normalize,
                 f_use_matryoshka, f_lr_epochs, f_causal_graph,
                 f_activation_func_str, f_distance_metric_str):
    # The LR range is intentionally narrow because previous runs already showed
    # that useful learning rates are in this area.
    lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)

    print("=" * 80)
    print(f"LR Trial {trial.number}")
    print(f"Activation: {f_activation_func_str}")
    print(f"Distance: {f_distance_metric_str}")
    print(f"LR: {lr}")
    print("=" * 80)

    return run_training_trial(
        f_model_path=f_model_path,
        f_curr_model_name=f_curr_model_name,
        f_train_dataset=f_train_dataset,
        f_valid_dataset=f_valid_dataset,
        f_batch_size=f_batch_size,
        f_accumulate_grad_batches=f_accumulate_grad_batches,
        f_normalize=f_normalize,
        f_use_matryoshka=f_use_matryoshka,
        f_max_epochs=f_lr_epochs,
        f_causal_graph=f_causal_graph,
        f_activation_func_str=f_activation_func_str,
        f_distance_metric_str=f_distance_metric_str,
        f_lr=lr,
        f_log_group="lr_search",
        f_run_suffix=f"lr_trial_{trial.number}",
    )


def combo_objective(trial, f_model_path, f_curr_model_name, f_datasets_by_distance,
              f_batch_size, f_accumulate_grad_batches, f_normalize,
              f_use_matryoshka, f_combo_epochs, f_causal_graph, f_best_lrs):
    # Combo search is intentionally discrete:
    # - Activation: ReLU or GELU
    # - Distance: cosine or Euclidean
    # The learning rate is not fixed globally. It is optimized separately for each combo first.
    # GridSampler ensures that each combination is evaluated exactly once.
    f_activation_func_str = trial.suggest_categorical("activation", ["relu", "gelu"])
    f_distance_metric_str = trial.suggest_categorical("distance", ["cosine", "euclid"])

    lr = f_best_lrs[(f_activation_func_str, f_distance_metric_str)]
    trial.set_user_attr("lr", lr)

    # Datasets depend only on model and distance metric, not on activation.
    # Therefore, they are precomputed once before the Optuna objective and reused here.
    f_train_dataset = f_datasets_by_distance[f_distance_metric_str]["train"]
    f_valid_dataset = f_datasets_by_distance[f_distance_metric_str]["valid"]

    print("=" * 80)
    print(f"Combo Trial {trial.number}")
    print(f"Activation: {f_activation_func_str}")
    print(f"Distance: {f_distance_metric_str}")
    print(f"Best LR for this combo: {lr}")
    print("=" * 80)

    return run_training_trial(
        f_model_path=f_model_path,
        f_curr_model_name=f_curr_model_name,
        f_train_dataset=f_train_dataset,
        f_valid_dataset=f_valid_dataset,
        f_batch_size=f_batch_size,
        f_accumulate_grad_batches=f_accumulate_grad_batches,
        f_normalize=f_normalize,
        f_use_matryoshka=f_use_matryoshka,
        f_max_epochs=f_combo_epochs,
        f_causal_graph=f_causal_graph,
        f_activation_func_str=f_activation_func_str,
        f_distance_metric_str=f_distance_metric_str,
        f_lr=lr,
        f_log_group="combo_search",
        f_run_suffix="combo",
    )


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model
    batch_size = args.batch_size
    normalize = args.normalize
    use_matryoshka = args.matryoshka
    lr_epochs = args.lr_epochs
    combo_epochs = args.combo_epochs
    lr_trials = args.lr_trials

    if batch_size > 128:
        raise ValueError("batch_size must be <= 128")
    if 128 % batch_size != 0:
        raise ValueError("batch_size must divide 128 (e.g. 128, 64, 32, 16, 8)")

    # Keep the effective batch size fixed across models for fair comparison.
    accumulate_grad_batches = 128 // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"LR search epochs: {lr_epochs}")
    print(f"Combo search epochs: {combo_epochs}")
    print(f"LR trials per combo: {lr_trials}")

    causal_graph = load_graph(DATA_DIR / "graphs" / "causenet-precision.jsonl")

    with open(DATA_DIR / "datasets" / "msmarco_train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(DATA_DIR / "datasets" / "msmarco_valid.json", encoding="utf-8") as f:
        valid_data = json.load(f)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    datasets_dir = DATA_DIR / "datasets"

    optuna_root_dir = DATA_DIR / "optuna_studies"
    optuna_lr_dir = optuna_root_dir / "lr"
    optuna_combo_dir = optuna_root_dir / "combo"
    optuna_lr_dir.mkdir(parents=True, exist_ok=True)
    optuna_combo_dir.mkdir(parents=True, exist_ok=True)

    # Dataset creation is done before the objective because both distance datasets
    # are needed anyway for the 2 x 2 grid search.
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

    best_lrs = {}

    for activation_func_str in ["relu", "gelu"]:
        for distance_metric_str in ["cosine", "euclid"]:
            print("=" * 80)
            print(f"OPTIMIZING LR FOR COMBO: {activation_func_str} + {distance_metric_str}")
            print("=" * 80)

            train_dataset = datasets_by_distance[distance_metric_str]["train"]
            valid_dataset = datasets_by_distance[distance_metric_str]["valid"]

            # The LR study name includes the number of LR trials and LR epochs.
            # This prevents reusing an old LR study that used different search settings.
            lr_study_name = (
                f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_"
                f"{normalize_str}_{mrl_str}_"
                f"lr{lr_trials}trials_{lr_epochs}epochs_search"
            )

            lr_optuna_db_path = optuna_lr_dir / (
                f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_"
                f"{normalize_str}_{mrl_str}_"
                f"lr{lr_trials}trials_{lr_epochs}epochs.sqlite3"
            )

            lr_study = optuna.create_study(
                storage=f"sqlite:///{lr_optuna_db_path}",
                study_name=lr_study_name,
                load_if_exists=True,
                direction="minimize",
            )

            cleanup_zombie_trials(lr_study, "LR")

            completed_lr_trials = [
                t for t in lr_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            trials_to_run = lr_trials - len(completed_lr_trials)

            if trials_to_run > 0:
                print(
                    f"Running LR search for {curr_model_name}, "
                    f"{activation_func_str} + {distance_metric_str}, "
                    f"for {trials_to_run} trials ..."
                )
                lr_study.optimize(
                    lambda l_trial: lr_objective(
                        l_trial,
                        model_path,
                        curr_model_name,
                        train_dataset,
                        valid_dataset,
                        batch_size,
                        accumulate_grad_batches,
                        normalize,
                        use_matryoshka,
                        lr_epochs,
                        causal_graph,
                        activation_func_str,
                        distance_metric_str,
                    ),
                    n_trials=trials_to_run,
                    gc_after_trial=True
                )
                print(f"Finished LR search for {activation_func_str} + {distance_metric_str}.")
            else:
                print(f"LR study for {activation_func_str} + {distance_metric_str} is already complete.")

            if len(lr_study.trials) == 0 or lr_study.best_trial is None:
                raise RuntimeError(
                    f"No successful LR trials found for {activation_func_str} + {distance_metric_str}."
                )

            best_lr = lr_study.best_params["lr"]
            best_lrs[(activation_func_str, distance_metric_str)] = best_lr
            print(f"Best LR for {activation_func_str} + {distance_metric_str}: {best_lr}")

    # Grid search is better than random sampling here because there are only four
    # possible combinations and each one should be evaluated exactly once.
    search_space = {
        "activation": ["relu", "gelu"],
        "distance": ["cosine", "euclid"],
    }

    sampler = GridSampler(search_space)

    # The combo study name includes LR-search and combo-search settings.
    # This prevents old studies with different epoch/trial settings from being reused accidentally.
    study_name = (
        f"{curr_model_name}_{normalize_str}_{mrl_str}_"
        f"lr{lr_trials}trials_{lr_epochs}epochs_"
        f"combo{combo_epochs}epochs_search"
    )

    optuna_db_path = optuna_combo_dir / (
        f"{curr_model_name}_{normalize_str}_{mrl_str}_"
        f"lr{lr_trials}trials_{lr_epochs}epochs_"
        f"combo{combo_epochs}epochs.sqlite3"
    )

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    cleanup_zombie_trials(study, "combo")

    # GridSampler has exactly four combinations here:
    # 2 activations x 2 distance metrics = 4 trials.
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
            lambda l_trial: combo_objective(
                l_trial,
                model_path,
                curr_model_name,
                datasets_by_distance,
                batch_size,
                accumulate_grad_batches,
                normalize,
                use_matryoshka,
                combo_epochs,
                causal_graph,
                best_lrs,
            ),
            n_trials=trials_to_run,
            gc_after_trial=True
        )
        print(f"Finished combo grid search for {curr_model_name}.")
    else:
        print("Combo study is already complete.")

    print("=" * 80)
    print("BEST LRS")
    print("=" * 80)
    for (activation_func_str, distance_metric_str), best_lr in best_lrs.items():
        print(f"{activation_func_str} + {distance_metric_str}: {best_lr}")

    print("=" * 80)
    print("BEST COMBO")
    print("=" * 80)
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Best LR: {study.best_trial.user_attrs.get('lr')}")