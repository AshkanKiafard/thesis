import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embeder:
    def __init__(self, model_path: str, distance_metric: str = 'cosine'):
        self.model = SentenceTransformer(model_path)
        model_name = model_path.split("/")[-1]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distance_metric = distance_metric
        self.cache = {}
        cache_file = f"data/embeddings/{model_name}_embeddings.npy"
        if os.path.exists(cache_file):
            self.cache = np.load(cache_file, allow_pickle=True).item()

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        embeddings = self.model.encode(text, device=self.device, normalize_embeddings=True)
        self.cache[text] = embeddings
        return embeddings

    def get_distance(self, embed1: np.ndarray, embed2: np.ndarray) -> float:
        match self.distance_metric:
            case 'cosine':
                embed1 = embed1.flatten()
                embed2 = embed2.flatten()
                return 1 - np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
            case 'euclidean':
                return np.linalg.norm(embed1 - embed2)
            case _:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
