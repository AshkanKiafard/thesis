import os
from enum import Enum
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


class DistanceMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


class STEmbeder:
    def __init__(self, model_path: str, distance_metric: DistanceMetric):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = os.path.basename(model_path.rstrip('/'))
        self.distance_metric = distance_metric
        self.matryoshka_dim = None
        self.cache = {}

        cache_dir = "data/embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{self.model_name}_embeddings.npy"

        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            self.cache = np.load(cache_file, allow_pickle=True).item()

        self.model = SentenceTransformer(model_path, device=self.device)

    def set_matryoshka_dim(self, dim: int):
        self.matryoshka_dim = dim

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]

        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self.cache[text] = emb

        return emb

    def get_distance(self, embed1, embed2):
        e1 = embed1.flatten()
        e2 = embed2.flatten()

        if self.matryoshka_dim is not None and self.matryoshka_dim < len(e1):
            e1 = e1[:self.matryoshka_dim]
            e2 = e2[:self.matryoshka_dim]

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            return 1 - np.dot(e1, e2) / (norm1 * norm2)

        return np.linalg.norm(e1 - e2)

    def get_model_name(self):
        return self.model_name


class GloveEmbeder:
    def __init__(self, glove_file_path: str, distance_metric: DistanceMetric = DistanceMetric.COSINE):
        self.distance_metric = distance_metric
        self.embeddings = {}
        self.default_dim = 300

        if not os.path.exists(glove_file_path):
            raise FileNotFoundError(f"Please unzip glove.6B.zip and place glove.6B.300d.txt at {glove_file_path}")

        print("Loading glove.6B embeddings...")
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings[word] = vector

    def embed(self, text: str) -> np.ndarray:
        text = text.lower()
        if text in self.embeddings:
            return self.embeddings[text]
        else:
            return np.zeros(self.default_dim, dtype="float32")

    def get_distance(self, embed1, embed2):
        e1 = embed1.flatten()
        e2 = embed2.flatten()

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            return 1 - np.dot(e1, e2) / (norm1 * norm2)

        return np.linalg.norm(e1 - e2)

