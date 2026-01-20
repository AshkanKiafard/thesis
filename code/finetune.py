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
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

import traverse_strategies as ts
from embeddings import Embeder, DistanceMetric
from utils import get_concept, load_graph, traverse_graph

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
        if matryoshka_dims is None:
            self.matryoshka_dims = [self.model.get_sentence_embedding_dimension()]
        else:
            self.matryoshka_dims = sorted(matryoshka_dims, reverse=True)

    def distance(self, a, b):
        if self.distance_metric == DistanceMetric.COSINE:
            return 1 - torch.nn.functional.cosine_similarity(a, b)
        elif self.distance_metric == DistanceMetric.EUCLIDEAN:
            return torch.norm(a - b, dim=-1)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def get_raw_embeddings(self, texts):
        features = self.model.tokenize(texts)
        features = {key: val.to(self.model.device) for key, val in features.items()}
        embeddings = self.model(features)["sentence_embedding"]
        return embeddings

    def compute_sub_loss(self, c_emb, e_emb, p_emb, n_emb):
        d_cp = self.distance(c_emb, p_emb)
        d_pe = self.distance(p_emb, e_emb)
        d_cn = self.distance(c_emb, n_emb)
        d_ne = self.distance(n_emb, e_emb)

        diff = (d_cp + d_pe) - (d_cn + d_ne)

        embeddings_sum = (torch.linalg.vector_norm(c_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(e_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(p_emb, dim=-1) - 1) ** 2 + \
                         (torch.linalg.vector_norm(n_emb, dim=-1) - 1) ** 2
        regularization = embeddings_sum.mean() if not self.normalize else 0.0

        return self.activation_func(diff).mean() + regularization

    def forward(self, sentence_features):
        c_raw = self.get_raw_embeddings(sentence_features[0])
        e_raw = self.get_raw_embeddings(sentence_features[1])
        p_raw = self.get_raw_embeddings(sentence_features[2])
        n_raw = self.get_raw_embeddings(sentence_features[3])

        total_loss = 0.0

        for dim in self.matryoshka_dims:
            c = c_raw[:, :dim]
            e = e_raw[:, :dim]
            p = p_raw[:, :dim]
            n = n_raw[:, :dim]

            if self.normalize:
                c = F.normalize(c, p=2, dim=1)
                e = F.normalize(e, p=2, dim=1)
                p = F.normalize(p, p=2, dim=1)
                n = F.normalize(n, p=2, dim=1)

            total_loss += self.compute_sub_loss(c, e, p, n)

        return total_loss


class LitAStar(pl.LightningModule):
    def __init__(self, model_name, activation_func, distance_metric, normalize, lr, matryoshka_dims, graph):
        super().__init__()
        self.save_hyperparameters(ignore=['graph'])
        self.lr = lr
        self.graph = graph
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.train()
        self.loss_fn = MatryoshkaAStarLoss(self.embedding_model,
                                           activation_func,
                                           distance_metric,
                                           normalize,
                                           matryoshka_dims)
        self.val_cache = {}
        self.result_cache = {}

    def on_validation_epoch_start(self):
        self.val_cache.clear()
        self.result_cache.clear()

    def on_validation_epoch_end(self):
        self.val_cache.clear()
        self.result_cache.clear()

    def embed(self, text: str):
        if text in self.val_cache:
            return self.val_cache[text]
        emb = self.loss_fn.get_raw_embeddings([text])
        if self.loss_fn.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        self.val_cache[text] = emb
        return emb

    def get_distance(self, emb_a, emb_b):
        return self.loss_fn.distance(emb_a, emb_b).item()

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn([
            batch["start_nodes"], batch["end_nodes"],
            batch["positives"], batch["negatives"]
        ])
        batch_size = len(batch["start_nodes"])
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO optimize
        # starts = batch["start_nodes"]
        # ends = batch["end_nodes"]
        #
        # total_visits = 0
        # seen_in_batch = set()
        # check_limit = 10
        # for i, (u, v) in enumerate(zip(starts, ends)):
        #     if len(seen_in_batch) >= check_limit:
        #         break
        #
        #     if (u, v) in seen_in_batch:
        #         continue
        #
        #     if (u, v) in self.result_cache:
        #         visits = self.result_cache[(u, v)]
        #     else:
        #         print(f"Validating pair {i}: {u} -> {v}")
        #         _, visits = traverse_graph(self.graph, u, v, self, ts.astar_traverse)
        #         self.result_cache[(u, v)] = visits
        #
        #     seen_in_batch.add((u, v))
        #     total_visits += visits
        #
        # avg_visited = total_visits / max(len(seen_in_batch), 1)
        # self.log("val_loss", avg_visited, prog_bar=True, batch_size=len(seen_in_batch))
        # return avg_visited

        loss = self.loss_fn([
            batch["start_nodes"], batch["end_nodes"],
            batch["positives"], batch["negatives"]
        ])
        batch_size = len(batch["start_nodes"])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

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
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def create_dataset(data, graph, embeder):
    astar_cache = {}
    start_nodes, end_nodes, positives, negatives = [], [], [], []

    for i, pair in enumerate(data):
        cause = get_concept(pair, 0)
        effect = get_concept(pair, 1)
        if (cause, effect) in astar_cache:
            path = astar_cache[(cause, effect)]
        else:
            path, _ = traverse_graph(graph, cause, effect, embeder, ts.astar_traverse)
            astar_cache[(cause, effect)] = path

        if len(path) < 2: continue

        for j, node in enumerate(path[:-1]):
            next_node = path[j + 1]
            successors = list(graph.successors(node))
            if next_node in successors: successors.remove(next_node)
            for successor in successors:
                start_nodes.append(node)
                end_nodes.append(effect)
                positives.append(next_node)
                negatives.append(successor)

    return Dataset.from_dict({
        "start_nodes": start_nodes, "end_nodes": end_nodes,
        "positives": positives, "negatives": negatives
    })


def objective(trial, model_name, train_loader, val_loader, activation_func, distance_metric, matryoshka_dims, graph):
    # TODO hyperparameter matryoshka
    lr = trial.suggest_float("lr", 2.5e-5, 3e-5, log=True)

    model = LitAStar(
        model_name=model_name,
        activation_func=activation_func,
        distance_metric=distance_metric,
        normalize=False,
        lr=lr,
        matryoshka_dims=matryoshka_dims,
        graph=graph
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        logger=True,
        default_root_dir="data/lightning_logs",
        enable_checkpointing=False,
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop, pruning_callback],
        enable_progress_bar=True
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    VERSION = 2
    CFG_DISTANCE_METRIC = DistanceMetric.COSINE
    CFG_ACTIVATION_FUNC = ActivationFunc.RELU
    MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")

    with open("data/datasets/msmarco_train.json") as f:
        train_data = json.load(f)

    with open("data/datasets/msmarco_valid.json") as f:
        valid_data = json.load(f)

    model_list = ["all-mpnet-base-v2"]

    for model_path in model_list:
        curr_model_name = model_path.split("/")[-1]
        activation_func_str = 'relu' if CFG_ACTIVATION_FUNC == ActivationFunc.RELU else 'gelu'
        distance_metric_str = 'cosine' if CFG_DISTANCE_METRIC == DistanceMetric.COSINE else 'euclid'
        trained_model_str = f"{curr_model_name}_{activation_func_str}_{distance_metric_str}_v{VERSION}"
        save_path = f"data/models/lightning/{trained_model_str}_finetuned"
        if os.path.exists(save_path):
            print(f"Model already exists at: {save_path}")
            continue

        print(f"Optimization starting for: {trained_model_str}")

        train_ds_path = f"data/datasets/train_{curr_model_name.replace('/', '_')}"
        valid_ds_path = f"data/datasets/valid_{curr_model_name.replace('/', '_')}"

        train_exists = os.path.exists(train_ds_path)
        valid_exists = os.path.exists(valid_ds_path)

        main_embeder = None
        if not train_exists or not valid_exists:
            print(f"Initializing Embedder for {curr_model_name}...")
            main_embeder = Embeder(model_path, CFG_DISTANCE_METRIC)

        if train_exists:
            print(f"Loading cached TRAIN dataset: {train_ds_path}")
            train_dataset = load_from_disk(train_ds_path)
        else:
            print(f"Creating TRAIN dataset for {curr_model_name} ...")
            train_dataset = create_dataset(train_data, causal_graph, main_embeder)
            train_dataset.save_to_disk(train_ds_path)
            print(f"TRAIN Dataset saved to: {train_ds_path}")

        if valid_exists:
            print(f"Loading cached VAL dataset: {valid_ds_path}")
            valid_dataset = load_from_disk(valid_ds_path)
        else:
            print(f"Creating VAL dataset for {curr_model_name} ...")
            valid_dataset = create_dataset(valid_data, causal_graph, main_embeder)
            valid_dataset.save_to_disk(valid_ds_path)
            print(f"VAL Dataset saved to: {valid_ds_path}")

        if main_embeder:
            del main_embeder

        print(f"Total Train examples: {len(train_dataset)}")
        print(f"Total Val examples: {len(valid_dataset)}")

        main_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4,
                                       persistent_workers=True)

        main_valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4,
                                       persistent_workers=True)

        optuna.logging.set_verbosity(optuna.logging.INFO)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            storage="sqlite:///data/optuna_studies/db.sqlite3",
            study_name=f"{trained_model_str}_optimization",
            load_if_exists=True,
            direction="minimize",
            pruner=pruner
        )

        print("Finetuning parameters ...")
        study.optimize(
            lambda trial: objective(trial,
                                    model_path,
                                    main_train_loader,
                                    main_valid_loader,
                                    CFG_ACTIVATION_FUNC,
                                    CFG_DISTANCE_METRIC,
                                    MATRYOSHKA_DIMS,
                                    causal_graph),
            n_trials=30,
            gc_after_trial=True
        )

        best_lr = study.best_params["lr"]

        print(f"Training model with LR={best_lr} ...")

        final_model = LitAStar(model_path,
                               CFG_ACTIVATION_FUNC,
                               CFG_DISTANCE_METRIC,
                               False,
                               best_lr,
                               MATRYOSHKA_DIMS,
                               causal_graph)

        logger = TensorBoardLogger("data/tb_logs", name=save_path)

        main_trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
            logger=logger
        )

        main_trainer.fit(final_model, main_train_loader, main_valid_loader)
        final_model.embedding_model.save(save_path)

        del final_model, main_trainer, study
        gc.collect()
        torch.cuda.empty_cache()
