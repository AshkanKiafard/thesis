import argparse
import gc
import json
import os
import sys
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# Make code/ importable when this script is executed from code/finetune/.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_ROOT))

from core.constants import (
    CAUSENET_GRAPH_PATH,
    DATASETS_DIR,
    LIGHTNING_LOGS_DIR,
    OPTUNA_STUDIES_DIR,
)
from finetune.astar_training_core import (
    LitAStar,
    cleanup_zombie_trials,
    find_latest_hparam_study,
    load_or_create_datasets,
    str_to_bool,
)
from core.utils import (
    canonical_activation,
    canonical_distance,
    load_causal_graph,
    parse_activation_func,
    parse_distance_metric,
)

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision("medium")

HPARAM_SEARCH_DEFAULT_BATCH_SIZE = 128
HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE = 128
HPARAM_SEARCH_MAX_BATCH_SIZE = 128


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
        default=HPARAM_SEARCH_DEFAULT_BATCH_SIZE,
        help=(
            "Physical batch size. Batch sizes up to the target effective batch size "
            f"will use gradient accumulation to reach {HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE}; "
            f"larger batch sizes run directly up to {HPARAM_SEARCH_MAX_BATCH_SIZE}."
        )
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

    parser.add_argument(
        "--run-suffix",
        type=str,
        default="v3",
        help="Suffix for Optuna study and hparam-search log names"
    )

    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["relu", "gelu"],
        help="Force every trial to use this activation instead of searching activation"
    )

    parser.add_argument(
        "--distance",
        type=str,
        default=None,
        choices=["cosine", "euclid", "euclidean"],
        help="Force every trial to use this distance metric instead of searching distance"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Force every trial to use this learning rate instead of searching LR"
    )

    return parser.parse_args()


def format_lr_slug(value):
    return f"{value:.12g}".replace(".", "p").replace("+", "").replace("-", "m")


def build_search_space_slug(fixed_activation, fixed_distance, fixed_lr):
    activation_part = fixed_activation if fixed_activation is not None else "search"
    distance_part = fixed_distance if fixed_distance is not None else "search"
    lr_part = format_lr_slug(fixed_lr) if fixed_lr is not None else "search"

    return f"act-{activation_part}_dist-{distance_part}_lr-{lr_part}"


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
        graph=f_causal_graph,
        batch_size=f_batch_size,
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
        save_dir=str(LIGHTNING_LOGS_DIR),
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
              f_use_matryoshka, f_epochs, f_patience, f_causal_graph,
              f_fixed_activation, f_fixed_distance, f_fixed_lr,
              f_run_suffix):
    # Activation, distance metric, and learning rate are optimized jointly.
    # This is the standard Optuna setup: all relevant hyperparameters are part
    # of the same search space instead of being optimized in separate stages.
    activation_choices = [f_fixed_activation] if f_fixed_activation is not None else ["relu", "gelu"]
    distance_choices = [f_fixed_distance] if f_fixed_distance is not None else ["cosine", "euclid"]

    f_activation_func_str = trial.suggest_categorical("activation", activation_choices)
    f_distance_metric_str = trial.suggest_categorical("distance", distance_choices)

    # The LR range is intentionally narrow because previous runs already showed
    # that useful learning rates are in this area.
    if f_fixed_lr is None:
        lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)
    else:
        lr = trial.suggest_float("lr", f_fixed_lr, f_fixed_lr, log=True)

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
        f_run_suffix=f"{f_run_suffix}_trial_{trial.number}",
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
    run_suffix = args.run_suffix
    fixed_activation = canonical_activation(args.activation)
    fixed_distance = canonical_distance(args.distance)
    fixed_lr = args.lr

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if batch_size > HPARAM_SEARCH_MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be <= {HPARAM_SEARCH_MAX_BATCH_SIZE}")
    if (
        batch_size <= HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE
        and HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE % batch_size != 0
    ):
        raise ValueError(
            f"batch_size must divide {HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE} "
            f"when batch_size <= {HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE} "
            "(e.g. 128, 64, 32, 16, 8)"
        )
    if patience < 0:
        raise ValueError("patience must be >= 0")
    if patience >= epochs:
        print("Warning: patience >= epochs, so early stopping will probably not trigger.")
    if fixed_lr is not None and fixed_lr <= 0:
        raise ValueError("lr must be > 0")

    # Keep smaller physical batches at the target effective batch size with
    # accumulation. Larger explicitly requested batches run directly.
    if batch_size <= HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE:
        accumulate_grad_batches = HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE // batch_size
    else:
        accumulate_grad_batches = 1

    hparam_search_space_slug = build_search_space_slug(fixed_activation, fixed_distance, fixed_lr)
    full_search_space_slug = build_search_space_slug(None, None, None)

    print(f"Max hparam search batch size: {HPARAM_SEARCH_MAX_BATCH_SIZE}")
    print(f"Target effective batch size: {HPARAM_SEARCH_TARGET_EFFECTIVE_BATCH_SIZE}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulate grad batches: {accumulate_grad_batches}")
    print(f"Effective batch size: {batch_size * accumulate_grad_batches}")
    print(f"Search epochs per trial: {epochs}")
    print(f"Search patience: {patience}")
    print(f"Optuna trials: {target_trials}")
    print(f"Run suffix: {run_suffix}")
    print(f"Activation: {fixed_activation if fixed_activation is not None else 'search'}")
    print(f"Distance: {fixed_distance if fixed_distance is not None else 'search'}")
    print(f"LR: {fixed_lr if fixed_lr is not None else 'search'}")

    causal_graph = load_causal_graph(CAUSENET_GRAPH_PATH)

    with open(DATASETS_DIR / "msmarco_train.json", encoding="utf-8") as train_file:
        train_data = json.load(train_file)

    with open(DATASETS_DIR / "msmarco_valid.json", encoding="utf-8") as valid_file:
        valid_data = json.load(valid_file)

    curr_model_name = model_path.split("/")[-1]
    normalize_str = "norm" if normalize else "nonorm"
    mrl_str = "matryoshka" if use_matryoshka else "single"

    datasets_dir = DATASETS_DIR

    optuna_root_dir = OPTUNA_STUDIES_DIR
    optuna_hparam_search_dir = optuna_root_dir / "hparam_search"
    optuna_hparam_search_dir.mkdir(parents=True, exist_ok=True)

    # Dataset creation is done before the objective because datasets depend on
    # distance. If distance is fixed, only that dataset is needed.
    datasets_by_distance = {}
    distance_search_space = [fixed_distance] if fixed_distance is not None else ["cosine", "euclid"]

    for curr_distance_metric_str in distance_search_space:
        train_dataset, valid_dataset = load_or_create_datasets(
            model_path=model_path,
            curr_model_name=curr_model_name,
            distance_metric_str=curr_distance_metric_str,
            train_data=train_data,
            valid_data=valid_data,
            causal_graph=causal_graph,
            datasets_dir=datasets_dir,
        )

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
        run_suffix=run_suffix,
        hparam_search_space_slug=hparam_search_space_slug,
    )

    # Old full-search studies did not include an explicit search-space slug.
    # Keep them resumable, but do not let a full search resume a fixed-search study.
    if latest_study is None and hparam_search_space_slug == full_search_space_slug:
        latest_study = find_latest_hparam_study(
            optuna_hparam_search_dir=optuna_hparam_search_dir,
            curr_model_name=curr_model_name,
            normalize_str=normalize_str,
            mrl_str=mrl_str,
            run_suffix=run_suffix,
            hparam_search_space_slug=None,
            include_slugged_studies=False,
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
            f"{curr_model_name}_{normalize_str}_{mrl_str}_{run_suffix}_"
            f"{hparam_search_space_slug}_{target_trials}trials_{epochs}epochs_{patience}patience"
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
                fixed_activation,
                fixed_distance,
                fixed_lr,
                run_suffix,
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
