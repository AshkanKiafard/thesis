import gc
import json
import os
from enum import Enum

import pytorch_lightning as pl
import torch
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


class AStarLoss(nn.Module):
    def __init__(self, model, activation_func, distance_metric: DistanceMetric, margin: float):
        super().__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.margin = margin
        self.activation_func = activation_func
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def distance(self, a, b):
        if self.distance_metric == DistanceMetric.COSINE:
            return 1 - torch.nn.functional.cosine_similarity(a, b)
        elif self.distance_metric == DistanceMetric.EUCLIDEAN:
            return torch.norm(a - b, dim=-1)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def forward_text(self, texts):
        features = self.model.tokenize(texts)
        features = {key: val.to(self.model.device) for key, val in features.items()}
        return self.model(features)["sentence_embedding"]

    def forward(self, sentence_features):
        c_emb = self.forward_text(sentence_features[0])
        e_emb = self.forward_text(sentence_features[1])
        p_emb = self.forward_text(sentence_features[2])
        n_emb = self.forward_text(sentence_features[3])

        d_cp = self.distance(c_emb, p_emb)
        d_pe = self.distance(p_emb, e_emb)
        d_cn = self.distance(c_emb, n_emb)
        d_ne = self.distance(n_emb, e_emb)

        diff = (d_cp + d_pe) - (d_cn + d_ne)
        embeddings_sum = (torch.linalg.vector_norm(c_emb, dim=-1) - 1) ** 2 + (
                torch.linalg.vector_norm(e_emb, dim=-1) - 1) ** 2 + (
                                 torch.linalg.vector_norm(p_emb, dim=-1) - 1) ** 2 + (
                                 torch.linalg.vector_norm(n_emb, dim=-1) - 1) ** 2

        if self.activation_func == ActivationFunc.RELU:
            loss = self.relu(diff + embeddings_sum + self.margin).mean()
        else:
            loss = self.gelu(diff + embeddings_sum + self.margin).mean()

        return loss, diff.mean(), embeddings_sum.mean()


class LitAStar(pl.LightningModule):
    def __init__(self, model_name, activation_func, distance_metric, margin, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.embedding_model = SentenceTransformer(model_name)
        self.loss_fn = AStarLoss(self.embedding_model, activation_func, distance_metric, margin)

    def training_step(self, batch, batch_idx):
        loss, diff_val, emb_sum = self.loss_fn([
            batch["start_nodes"],
            batch["end_nodes"],
            batch["successors"],
            batch["negatives"]
        ])

        batch_len = len(batch["start_nodes"])
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_len)
        self.log("train_diff", diff_val, prog_bar=False, batch_size=batch_len)
        self.log("train_reg", emb_sum, prog_bar=False, batch_size=batch_len)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, diff_val, emb_sum = self.loss_fn([
            batch["start_nodes"],
            batch["end_nodes"],
            batch["successors"],
            batch["negatives"]
        ])

        batch_len = len(batch["start_nodes"])
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_len)
        self.log("val_diff", diff_val, prog_bar=False, batch_size=batch_len)
        self.log("val_reg", emb_sum, prog_bar=False, batch_size=batch_len)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def create_dataset(data, graph, embeder):
    astar_cache = {}

    start_nodes = []
    end_nodes = []
    positives = []
    negatives = []

    for i, pair in enumerate(data):
        if i % 100 == 0:
            print(f"{embeder.get_model_name()}: {i} / {len(data)}")

        cause = get_concept(pair, 0)
        effect = get_concept(pair, 1)

        if (cause, effect) in astar_cache:
            path = astar_cache[(cause, effect)]
        else:
            path, _ = traverse_graph(graph, cause, effect, embeder, ts.astar_traverse)
            astar_cache[(cause, effect)] = path

        if not path or len(path) < 2:
            continue

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
        "successors": positives,
        "negatives": negatives
    })


if __name__ == "__main__":
    CFG_DISTANCE_METRIC = DistanceMetric.COSINE
    CFG_ACTIVATION_FUNC = ActivationFunc.RELU
    CFG_MARGIN = 0.5
    CFG_LR = 2e-5

    causal_graph = load_graph("data/graphs/causenet-precision.jsonl")

    with open("data/datasets/msmarco_train.json") as f:
        train_data = json.load(f)

    model_list = [
        "sentence-transformers/all-mpnet-base-v2",
        # "sentence-transformers/all-MiniLM-L-12-v2",
        # "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    ]

    for model_path in model_list:
        curr_model_name = model_path.split("/")[-1]

        save_path = f"data/models/lightning/{curr_model_name}_fine-tuned_{'relu' if CFG_ACTIVATION_FUNC == ActivationFunc.RELU else 'gelu'}"

        if os.path.exists(save_path):
            print(f"Model already exists at: {save_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"STARTING PIPELINE FOR: {curr_model_name}")
        print(f"{'=' * 50}\n")

        main_embeder = Embeder(model_path, CFG_DISTANCE_METRIC)

        ds_path = f"data/datasets/train_{curr_model_name.replace('/', '_')}"

        if os.path.exists(ds_path):
            print(f"Loading cached dataset: {ds_path}")
            full_dataset = load_from_disk(ds_path)
        else:
            print(f"Creating dataset for {curr_model_name}...")
            full_dataset = create_dataset(train_data, causal_graph, main_embeder)
            full_dataset.save_to_disk(ds_path)
            print(f"Dataset saved to: {ds_path}")

        print(f"Total examples: {len(full_dataset)}")

        split = full_dataset.train_test_split(test_size=0.1)
        train_ds = split["train"]
        val_ds = split["test"]

        train_loader = DataLoader(
            train_ds,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )

        lit_model = LitAStar(model_path, CFG_ACTIVATION_FUNC, CFG_DISTANCE_METRIC, CFG_MARGIN, lr=CFG_LR)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )

        logger = TensorBoardLogger("tb_logs", name=save_path)
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=50,
            callbacks=[early_stop],
            enable_progress_bar=True,
            logger=logger
        )

        print(f"Training: {curr_model_name}")
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print(f"Saving model to {save_path}")
        lit_model.embedding_model.save(save_path)

        del lit_model, trainer, main_embeder
        gc.collect()
        torch.cuda.empty_cache()

    print("Pipeline finished.")
