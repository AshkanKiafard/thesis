import gc
import json
import os
from enum import Enum

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

import traverse_strategies as ts
from embeddings import STEmbedder, DistanceMetric
from utils import get_concept, load_graph, traverse_graph, get_matryoshka_dims

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# "medium" is usually a decent trade-off here and can speed up training on newer GPUs.
torch.set_float32_matmul_precision('medium')


class ActivationFunc(Enum):
    RELU = 1
    GELU = 2


class MatryoshkaAStarLoss(nn.Module):
    def __init__(self, model, activation_func, distance_metric: DistanceMetric, normalize: bool,
                 matryoshka_dims: list[int] = None):
        super().__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.activation_func = nn.ReLU() if activation_func == ActivationFunc.RELU else nn.GELU()
        self.normalize = normalize
        # If no explicit Matryoshka dimensions are given, just use the full embedding size.
        if matryoshka_dims is None:
            self.matryoshka_dims = [self.model.get_sentence_embedding_dimension()]
        else:
            # Sorting from large to small makes the truncation logic easier to reason about.
            self.matryoshka_dims = sorted(set(matryoshka_dims), reverse=True)

    def distance(self, a, b):
        if self.distance_metric == DistanceMetric.COSINE:
            return 1 - torch.nn.functional.cosine_similarity(a, b)
        elif self.distance_metric == DistanceMetric.EUCLIDEAN:
            return torch.norm(a - b, dim=-1)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def get_raw_embeddings(self, texts):
        # We directly call the SentenceTransformer forward pass here instead of encode()
        # because this is used during training and needs gradients.
        features = self.model.tokenize(texts)
        features = {key: val.to(self.model.device) for key, val in features.items()}
        embeddings = self.model(features)["sentence_embedding"]
        return embeddings

    def compute_sub_loss(self, c_emb, e_emb, p_emb, n_emb):
        # Distances for the positive step on the path.
        d_cp = self.distance(c_emb, p_emb)
        d_pe = self.distance(p_emb, e_emb)

        # Distances for a negative alternative successor.
        d_cn = self.distance(c_emb, n_emb)
        d_ne = self.distance(n_emb, e_emb)

        # The idea is simple:
        # a good next node should make the overall route to the effect look better
        # than a bad successor.
        diff = (d_cp + d_pe) - (d_cn + d_ne)

        # If embeddings are not normalized explicitly, keep their norms roughly stable.
        embeddings_sum = (torch.linalg.vector_norm(c_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(e_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(p_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(n_emb, dim=-1) - 1) ** 2
        regularization = embeddings_sum.mean() if not self.normalize else 0.0

        return self.activation_func(diff).mean() + regularization

    def forward(self, sentence_features):
        # sentence_features contains:
        # [start_nodes, end_nodes, positives, negatives]
        c_raw = self.get_raw_embeddings(sentence_features[0])
        e_raw = self.get_raw_embeddings(sentence_features[1])
        p_raw = self.get_raw_embeddings(sentence_features[2])
        n_raw = self.get_raw_embeddings(sentence_features[3])

        total_loss = 0.0

        # Compute the loss for every Matryoshka slice and sum them up.
        for dim in self.matryoshka_dims:
            cf = c_raw[:, :dim]
            ef = e_raw[:, :dim]
            pf = p_raw[:, :dim]
            nf = n_raw[:, :dim]

            if self.normalize:
                cf = F.normalize(cf, p=2, dim=1)
                ef = F.normalize(ef, p=2, dim=1)
                pf = F.normalize(pf, p=2, dim=1)
                nf = F.normalize(nf, p=2, dim=1)

            total_loss += self.compute_sub_loss(cf, ef, pf, nf)

        return total_loss


class LitAStar(pl.LightningModule):
    def __init__(self, model_name, activation_func, distance_metric, normalize, lr, use_matryoshka, graph):
        super().__init__()

        # The graph object is large and not something we want in Lightning's saved hparams.
        self.save_hyperparameters(ignore=['graph'])

        self.lr = lr
        self.graph = graph

        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.train()

        model_dimension = self.embedding_model.get_sentence_embedding_dimension()
        matryoshka_dims = get_matryoshka_dims(model_dimension) if use_matryoshka else [model_dimension]

        self.loss_fn = MatryoshkaAStarLoss(
            self.embedding_model,
            activation_func,
            distance_metric,
            normalize,
            matryoshka_dims
        )

        # val_cache stores node -> embedding for the current validation epoch.
        # result_cache stores (start, end) -> visited nodes, so repeated pairs are cheap.
        self.val_cache = {}
        self.result_cache = {}

    def on_validation_epoch_start(self):
        self.val_cache.clear()
        self.result_cache.clear()

        all_nodes = list(self.graph.nodes())

        if not all_nodes:
            return

        # Validation repeatedly queries embeddings for graph nodes.
        # So we precompute all node embeddings once per validation epoch.
        self.embedding_model.eval()
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                all_nodes,
                batch_size=2048,
                convert_to_tensor=True,
                normalize_embeddings=self.loss_fn.normalize,
                show_progress_bar=False,
                device=self.device
            )

        self.val_cache = dict(zip(all_nodes, embeddings))

    def on_validation_epoch_end(self):
        # Free memory explicitly after each validation epoch.
        self.val_cache.clear()
        self.result_cache.clear()

    def embed(self, text: str):
        # traverse_graph expects the model object to expose an embed() method.
        return self.val_cache[text]

    def get_distance(self, emb_a, emb_b):
        # traverse_graph also expects a distance function on the model side.
        return self.loss_fn.distance(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn([
            batch["start_nodes"], batch["end_nodes"],
            batch["positives"], batch["negatives"]
        ])

        batch_size = len(batch["start_nodes"])
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        starts = batch["start_nodes"]
        ends = batch["end_nodes"]

        # Validation is based on actual A* search behavior, not only embedding loss.
        # We deduplicate pairs inside the batch so we do not run the same search twice.
        unique_pairs = list(set(zip(starts, ends)))
        total_visits = 0
        pairs_validated = 0

        for u, v in unique_pairs:
            if (u, v) in self.result_cache:
                visits = self.result_cache[(u, v)]
            else:
                _, visits = traverse_graph(self.graph, u, v, self, ts.astar_traverse, None)
                self.result_cache[(u, v)] = visits

            total_visits += visits
            pairs_validated += 1

        avg_visited = total_visits / max(pairs_validated, 1)
        self.log("val/astar_cost", avg_visited, prog_bar=True, batch_size=pairs_validated)

        # Still log the embedding loss as a secondary signal for debugging.
        val_loss = self.loss_fn([
            batch["start_nodes"], batch["end_nodes"],
            batch["positives"], batch["negatives"]
        ])
        self.log("val/embedding_loss", val_loss, prog_bar=True, batch_size=len(starts))

        return avg_visited

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # We reduce LR when the actual search cost stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/astar_cost",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Logging gradient norm helps spot exploding gradients or dead training.
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) > 0:
            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
            self.log("train/grad_norm", grad_norm, prog_bar=True)


def create_dataset(data, graph, embedder):
    astar_cache = {}
    start_nodes, end_nodes, positives, negatives = [], [], [], []

    for i, pair in enumerate(data):
        cause = get_concept(pair, 0)
        effect = get_concept(pair, 1)

        # Cache path generation because the same pair may appear multiple times.
        if (cause, effect) in astar_cache:
            path = astar_cache[(cause, effect)]
        else:
            path, _ = traverse_graph(graph, cause, effect, embedder, ts.astar_traverse, None)
            astar_cache[(cause, effect)] = path

        # We need at least one hop to define a positive next node.
        if len(path) < 2:
            continue

        # For each step on the A* path:
        # - next_node is the positive example
        # - every other successor becomes a negative example
        for j, node in enumerate(path[:-1]):
            next_node = path[j + 1]
            successors = list(graph.successors(node))

            if next_node in successors:
                successors.remove(next_node)

            for successor in successors:
                start_nodes.append(node)
                end_nodes.append(effect)
                positives.append(next_node)
                negatives.append(successor)

    return Dataset.from_dict({
        "start_nodes": start_nodes,
        "end_nodes": end_nodes,
        "positives": positives,
        "negatives": negatives
    })


def objective(trial, model_name, model_str, train_loader, val_loader, activation_func, distance_metric, use_matryoshka, graph):
    # The search range is intentionally narrow because previous runs already showed
    # that useful learning rates are in this area.
    lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)

    model = LitAStar(
        model_name=model_name,
        activation_func=activation_func,
        distance_metric=distance_metric,
        normalize=False,
        lr=lr,
        use_matryoshka=use_matryoshka,
        graph=graph
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/astar_cost")
    early_stop = EarlyStopping(monitor="val/astar_cost", patience=3, mode="min")

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=f"data/lightning_logs/{model_str}/{SLURM_JOB_ID}",
        enable_checkpointing=False,
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, pruning_callback],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics["val/astar_cost"].item()


if __name__ == "__main__":
    TOTAL_TARGET_TRIALS = 30
    CFG_DISTANCE_METRIC = DistanceMetric.EUCLIDEAN
    CFG_ACTIVATION_FUNC = ActivationFunc.RELU
    USE_MATRYOSHKA = True

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")

    with open("data/datasets/msmarco_train.json") as f:
        train_data = json.load(f)

    with open("data/datasets/msmarco_valid.json") as f:
        valid_data = json.load(f)

    # Can be extended later if multiple base encoders should be tuned in one run.
    model_list = ["Qwen/Qwen3-Embedding-0.6B"]

    for model_path in model_list:
        curr_model_name = model_path.split("/")[-1]
        activation_func_str = 'relu' if CFG_ACTIVATION_FUNC == ActivationFunc.RELU else 'gelu'
        distance_metric_str = 'cosine' if CFG_DISTANCE_METRIC == DistanceMetric.COSINE else 'euclid'
        mrl_str = "matryoshka" if USE_MATRYOSHKA else "single"
        trained_model_str = f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_{mrl_str}"
        save_path = f"data/models/lightning/{trained_model_str}_finetuned"
        ckpt_dir = f"data/checkpoints/{trained_model_str}"

        # Skip everything if the final exported SentenceTransformer already exists.
        if os.path.exists(save_path):
            print(f"Model already exists at: {save_path}")
            continue

        print(f"Optimization starting for: {trained_model_str} - Slurm Job ID: {SLURM_JOB_ID}")

        dataset_suffix = f"{curr_model_name.replace('/', '_')}_{distance_metric_str}"
        train_ds_path = f"data/datasets/train_{dataset_suffix}"
        valid_ds_path = f"data/datasets/valid_{dataset_suffix}"

        train_exists = os.path.exists(train_ds_path)
        valid_exists = os.path.exists(valid_ds_path)

        main_embedder = None
        if not train_exists or not valid_exists:
            print(f"Initializing Embedder for {curr_model_name}...")
            main_embedder = STEmbedder(model_path, CFG_DISTANCE_METRIC)

        if train_exists:
            print(f"Loading cached TRAIN dataset: {train_ds_path}")
            train_dataset = load_from_disk(train_ds_path)
        else:
            print(f"Creating TRAIN dataset: {train_ds_path}")
            train_dataset = create_dataset(train_data, causal_graph, main_embedder)
            train_dataset.save_to_disk(train_ds_path)
            print(f"TRAIN Dataset saved to: {train_ds_path}")

        if valid_exists:
            print(f"Loading cached VAL dataset: {valid_ds_path}")
            valid_dataset = load_from_disk(valid_ds_path)
        else:
            print(f"Creating VAL dataset: {valid_ds_path}")
            valid_dataset = create_dataset(valid_data, causal_graph, main_embedder)
            valid_dataset.save_to_disk(valid_ds_path)
            print(f"VAL Dataset saved to: {valid_ds_path}")

        if main_embedder:
            del main_embedder

        print(f"Total Train examples: {len(train_dataset)}")
        print(f"Total Val examples: {len(valid_dataset)}")

        main_train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

        main_valid_loader = DataLoader(
            valid_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

        optuna.logging.set_verbosity(optuna.logging.INFO)
        pruner = optuna.pruners.MedianPruner()

        study_name = f"{trained_model_str}_optimization_{SLURM_JOB_ID}"
        optuna_db_path = f"data/optuna_studies/{trained_model_str}_{SLURM_JOB_ID}.sqlite3"
        study = optuna.create_study(
            storage=f"sqlite:///{optuna_db_path}",
            study_name=study_name,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner
        )

        # When a run is interrupted, Optuna may leave trials in RUNNING state.
        # Mark them as failed so the study can resume cleanly.
        print("Cleaning zombie trials ...")
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]

        if running_trials:
            print(f"Found {len(running_trials)} interrupted trials (Zombies). Cleaning them up...")
            for r_trial in running_trials:
                try:
                    study.tell(r_trial.number, state=optuna.trial.TrialState.FAIL)
                    print(f"Marked interrupted Trial {r_trial.number} as FAILED.")
                except Exception as e:
                    print(f"Warning: Could not update status for Trial {r_trial.number}: {e}")

        # Count both COMPLETE and PRUNED as already-attempted useful trials.
        valid_trials = [
            t for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        ]
        current_valid_count = len(valid_trials)
        trials_to_run = TOTAL_TARGET_TRIALS - current_valid_count

        if trials_to_run > 0:
            print(f"Resuming study. Running {trials_to_run} more trials...")
            study.optimize(
                lambda l_trial: objective(
                    l_trial,
                    model_path,
                    trained_model_str,
                    main_train_loader,
                    main_valid_loader,
                    CFG_ACTIVATION_FUNC,
                    CFG_DISTANCE_METRIC,
                    USE_MATRYOSHKA,
                    causal_graph
                ),
                n_trials=trials_to_run,
                gc_after_trial=True
            )

            gc.collect()
            torch.cuda.empty_cache()

        best_lr = study.best_params["lr"]
        print(f"Training model with LR={best_lr} ...")

        final_model = LitAStar(
            model_path,
            CFG_ACTIVATION_FUNC,
            CFG_DISTANCE_METRIC,
            False,
            best_lr,
            USE_MATRYOSHKA,
            causal_graph
        )

        logger = TensorBoardLogger(
            "data/tb_logs",
            name=trained_model_str,
            version=SLURM_JOB_ID
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
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
            max_epochs=50,
            accelerator="gpu",
            devices=1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            default_root_dir=f"data/lightning_logs/{trained_model_str}/{SLURM_JOB_ID}",
            num_sanity_val_steps=0,
        )

        ckpt_path = None
        if os.path.exists(ckpt_dir):
            checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]

            if checkpoint_files:
                # Resume from the most recently modified checkpoint if one exists.
                checkpoint_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
                    reverse=True
                )
                ckpt_path = os.path.join(ckpt_dir, checkpoint_files[0])
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
        best_model.embedding_model.save(save_path)

        # Try to free GPU/CPU memory before moving to the next model.
        del final_model, best_model, main_trainer, study
        gc.collect()
        torch.cuda.empty_cache()
