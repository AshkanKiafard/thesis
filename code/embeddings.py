import os
from enum import Enum
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


class DistanceMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


class Embeder:
    def __init__(self, model_path: str, distance_metric: DistanceMetric):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = os.path.basename(model_path.rstrip('/'))
        self.distance_metric = distance_metric
        self.cache = {}

        # Load cache
        cache_dir = "data/embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{self.model_name}_embeddings.npy"

        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            self.cache = np.load(cache_file, allow_pickle=True).item()

        self.model = SentenceTransformer(model_path, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]

        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self.cache[text] = emb

        return emb

    def get_distance(self, embed1, embed2):
        if self.distance_metric == DistanceMetric.COSINE:
            embed1, embed2 = embed1.flatten(), embed2.flatten()
            return 1 - np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

        return np.linalg.norm(embed1 - embed2)

    def get_model_name(self):
        return self.model_name
