import os
from pathlib import Path

import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from core.constants import DistanceMetric

# code/core/embeddings.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"


class STEmbedder:
    def __init__(self, model_path: str, distance_metric: DistanceMetric):
        # Use GPU if available, otherwise fall back to CPU.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Store only the last path component as a readable model name.
        self.model_name = os.path.basename(model_path.rstrip('/'))

        self.distance_metric = distance_metric

        # If set, distances will be computed only on the first k embedding dimensions.
        self.matryoshka_dim = None

        # In-memory cache for already embedded texts.
        self.cache = {}

        # Store embedding caches inside the mounted project directory.
        # This avoids writing to ../data, which resolves outside /app inside the Slurm container.
        cache_dir = DATA_DIR / "embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = cache_dir / f"{self.model_name}_embeddings.npy"

        # Load precomputed embeddings if they exist.
        if os.path.exists(self.cache_file):
            self.cache = np.load(self.cache_file, allow_pickle=True).item()
            print(f"Loaded cached embeddings from {self.cache_file}")

        tokenizer_kwargs = {}
        # Fix known tokenizer regex issue for Mistral/Qwen-style tokenizers.
        if "Qwen" in self.model_name or "Mistral" in self.model_name:
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.model = SentenceTransformer(
            model_path,
            device=self.device,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    def set_matryoshka_dim(self, dim: int):
        self.matryoshka_dim = dim

    def get_model_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        # Return cached embedding if available.
        if text in self.cache:
            return self.cache[text]

        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self.cache[text] = emb
        return emb

    def preload(self, texts, batch_size: int = 64, save: bool = True) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        missing_texts = [
            text
            for text in dict.fromkeys(texts)
            if text not in self.cache
        ]

        if not missing_texts:
            return 0

        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start:start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            for text, emb in zip(batch, batch_embeddings):
                self.cache[text] = emb

        if save:
            self.save_cache()

        return len(missing_texts)

    def save_cache(self):
        np.save(self.cache_file, self.cache)

    def get_distance(self, embed1, embed2):
        e1 = embed1.flatten()
        e2 = embed2.flatten()

        # Optionally truncate embeddings to a smaller Matryoshka dimension.
        if self.matryoshka_dim is not None and self.matryoshka_dim < len(e1):
            e1 = e1[:self.matryoshka_dim]
            e2 = e2[:self.matryoshka_dim]

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)

            # If one vector is zero, cosine similarity is undefined.
            # Returning 1.0 here means "max distance".
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

        # Match original causal-qa-rl preprocessing.
        self.stop_words = set(stopwords.words("english"))

        if not os.path.exists(glove_file_path):
            raise FileNotFoundError(
                f"Please unzip glove.6B.zip and place glove.6B.300d.txt at {glove_file_path}"
            )

        print("Loading glove.6B embeddings...")
        with open(glove_file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                self.embeddings[word] = vector

    def _remove_stop_words(self, context: str):
        # Original graph_utils.remove_stop_words:
        # tokens = word_tokenize(context)
        # return [t for t in tokens if t not in STOP_WORDS]
        tokens = word_tokenize(context)
        return [t for t in tokens if t not in self.stop_words]

    def _mean_embedding(self, parts):
        # Original GloveEmbeddingProvider._get_embedding:
        # part_embeddings = [np.array(self.embeddings[part]) for part in parts if part in self.embeddings]
        # emb = np.mean(part_embeddings, axis=0) if len(part_embeddings) > 0 else np.ones(self.num_dimensions)
        part_embeddings = [
            np.asarray(self.embeddings[part], dtype="float32")
            for part in parts
            if part in self.embeddings
        ]

        if len(part_embeddings) == 0:
            return np.ones(self.default_dim, dtype="float32")

        return np.mean(part_embeddings, axis=0).astype("float32")

    def embed_entity(self, text: str) -> np.ndarray:
        # Original entity embedding:
        # entity.split(" ")
        #
        # Important: no custom tokenizer and no stopword removal here.
        return self._mean_embedding(text.split(" "))

    def embed_question(self, text: str) -> np.ndarray:
        # Original question embedding:
        # graph_utils.remove_stop_words(question)
        return self._mean_embedding(self._remove_stop_words(text))

    def embed_relation(self, text: str) -> np.ndarray:
        # Original relation/source embedding:
        # if relation is a string, relation.split(" ") happens in original relation_embeddings()
        # for CauseNet sources, graph_sources already stores remove_stop_words(source).
        #
        # In our port, we receive the raw source sentence, so we apply remove_stop_words here.
        return self._mean_embedding(self._remove_stop_words(text))

    def embed(self, text: str) -> np.ndarray:
        # Keep old API for path-cost computation.
        return self.embed_entity(text)

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
