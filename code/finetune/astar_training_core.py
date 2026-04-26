import argparse
from enum import Enum

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import nn

import traverse_strategies as ts
from core.embeddings import DistanceMetric
from core.utils import get_concept, traverse_graph, get_matryoshka_dims


class ActivationFunc(Enum):
    RELU = 1
    GELU = 2


def parse_activation_func(value: str) -> ActivationFunc:
    value = value.strip().lower()
    if value == "relu":
        return ActivationFunc.RELU
    elif value == "gelu":
        return ActivationFunc.GELU
    else:
        raise ValueError(f"Unsupported activation function: {value}")


def parse_distance_metric(value: str) -> DistanceMetric:
    value = value.strip().lower()
    if value == "cosine":
        return DistanceMetric.COSINE
    elif value in {"euclid", "euclidean"}:
        return DistanceMetric.EUCLIDEAN
    else:
        raise ValueError(f"Unsupported distance metric: {value}")


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    elif value in {"false", "0", "no", "n"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


class MatryoshkaAStarLoss(nn.Module):
    def __init__(
        self,
        model,
        cls_activation_func,
        cls_distance_metric: DistanceMetric,
        cls_normalize: bool,
        matryoshka_dims: list[int] = None
    ):
        super().__init__()
        self.model = model
        self.distance_metric = cls_distance_metric
        self.activation_func = nn.ReLU() if cls_activation_func == ActivationFunc.RELU else nn.GELU()
        self.normalize = cls_normalize

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
        embeddings_sum = (
            (torch.linalg.vector_norm(c_emb, dim=-1) - 1) ** 2
            + (torch.linalg.vector_norm(e_emb, dim=-1) - 1) ** 2
            + (torch.linalg.vector_norm(p_emb, dim=-1) - 1) ** 2
            + (torch.linalg.vector_norm(n_emb, dim=-1) - 1) ** 2
        )

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
    def __init__(
        self,
        model_name,
        cls_activation_func,
        cls_distance_metric,
        cls_normalize,
        lr,
        cls_use_matryoshka,
        graph
    ):
        super().__init__()

        # The graph object is large and not something we want in Lightning's saved hparams.
        self.save_hyperparameters(ignore=["graph"])

        self.lr = lr
        self.graph = graph

        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.train()

        model_dimension = self.embedding_model.get_sentence_embedding_dimension()
        matryoshka_dims = get_matryoshka_dims(model_dimension) if cls_use_matryoshka else [model_dimension]

        self.loss_fn = MatryoshkaAStarLoss(
            self.embedding_model,
            cls_activation_func,
            cls_distance_metric,
            cls_normalize,
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
            batch["start_nodes"],
            batch["end_nodes"],
            batch["positives"],
            batch["negatives"]
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
            batch["start_nodes"],
            batch["end_nodes"],
            batch["positives"],
            batch["negatives"]
        ])
        self.log("val/embedding_loss", val_loss, prog_bar=True, batch_size=len(starts))

        return avg_visited

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # We reduce LR when the actual search cost stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
            grad_norm = torch.norm(
                torch.stack([torch.norm(g.detach(), 2) for g in grads]),
                2
            )
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