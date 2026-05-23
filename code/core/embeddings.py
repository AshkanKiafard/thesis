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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Store only the last path component as a readable model name.
        self.model_name = os.path.basename(model_path.rstrip('/'))

        self.distance_metric = distance_metric

        # If set, distances will be computed only on the first k embedding dimensions.
        self.matryoshka_dim = None

        # CPU cache is persisted as NumPy. Tensor cache keeps hot embeddings on
        # the active torch device for traversal distance computations.
        self.cache = {}
        self.tensor_cache = {}

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

    def _as_tensor(self, embedding) -> torch.Tensor:
        if isinstance(embedding, torch.Tensor):
            return embedding.to(device=self.device, dtype=torch.float32).flatten()

        return torch.as_tensor(
            embedding,
            device=self.device,
            dtype=torch.float32,
        ).flatten()

    @staticmethod
    def _as_numpy(embedding) -> np.ndarray:
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().numpy().astype("float32", copy=False)

        return np.asarray(embedding, dtype="float32")

    def set_matryoshka_dim(self, dim: int):
        self.matryoshka_dim = dim

    def get_model_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> torch.Tensor:
        if text in self.tensor_cache:
            return self.tensor_cache[text]

        if text in self.cache:
            tensor = self._as_tensor(self.cache[text])
            self.tensor_cache[text] = tensor
            return tensor

        with torch.no_grad():
            emb = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            ).detach()

        tensor = self._as_tensor(emb)
        self.tensor_cache[text] = tensor
        self.cache[text] = self._as_numpy(tensor)
        return tensor

    def embed_numpy(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.embed(text)

        return self._as_numpy(self.cache[text])

    def preload(self, texts, batch_size: int = 64, save: bool = True) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        unique_texts = list(dict.fromkeys(texts))

        for text in unique_texts:
            if text in self.cache and text not in self.tensor_cache:
                self.tensor_cache[text] = self._as_tensor(self.cache[text])

        missing_texts = [
            text
            for text in unique_texts
            if text not in self.cache
        ]

        if not missing_texts:
            return 0

        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start:start + batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                ).detach()

            for text, emb in zip(batch, batch_embeddings):
                tensor = self._as_tensor(emb)
                self.tensor_cache[text] = tensor
                self.cache[text] = self._as_numpy(tensor)

        if save:
            self.save_cache()

        return len(missing_texts)

    def save_cache(self):
        serializable_cache = {
            text: self._as_numpy(embedding)
            for text, embedding in self.cache.items()
        }
        np.save(self.cache_file, serializable_cache)

    def get_distance(self, embed1, embed2):
        e1 = self._as_tensor(embed1)
        e2 = self._as_tensor(embed2)

        # Optionally truncate embeddings to a smaller Matryoshka dimension.
        if self.matryoshka_dim is not None and self.matryoshka_dim < e1.numel():
            e1 = e1[:self.matryoshka_dim]
            e2 = e2[:self.matryoshka_dim]

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = torch.linalg.vector_norm(e1)
            norm2 = torch.linalg.vector_norm(e2)

            # If one vector is zero, cosine similarity is undefined.
            # Returning 1.0 here means "max distance".
            if norm1 == 0 or norm2 == 0:
                return 1.0

            return float((1 - torch.dot(e1, e2) / (norm1 * norm2)).item())

        return float(torch.linalg.vector_norm(e1 - e2).item())

    def get_distances(self, embed1, embeddings):
        if not embeddings:
            return []

        e1 = self._as_tensor(embed1)
        matrix = torch.stack([self._as_tensor(embedding) for embedding in embeddings])

        if self.matryoshka_dim is not None and self.matryoshka_dim < e1.numel():
            e1 = e1[:self.matryoshka_dim]
            matrix = matrix[:, :self.matryoshka_dim]

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = torch.linalg.vector_norm(e1)
            norm2 = torch.linalg.vector_norm(matrix, dim=1)
            denominator = norm1 * norm2
            distances = torch.ones(matrix.shape[0], device=self.device, dtype=torch.float32)
            valid = denominator != 0
            distances[valid] = 1 - (matrix[valid] @ e1) / denominator[valid]
        else:
            distances = torch.linalg.vector_norm(matrix - e1, dim=1)

        return distances.detach().cpu().tolist()

    def get_model_name(self):
        return self.model_name


class GloveEmbeder:
    def __init__(
        self,
        glove_file_path: str,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self.distance_metric = distance_metric
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.word_to_idx = {}
        self.embedding_matrix = None
        self.default_dim = 300
        self.default_embedding = torch.ones(
            self.default_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.entity_cache = {}
        self.question_cache = {}
        self.relation_cache = {}

        # Match original causal-qa-rl preprocessing.
        self.stop_words = set(stopwords.words("english"))

        if not os.path.exists(glove_file_path):
            raise FileNotFoundError(
                f"Please unzip glove.6B.zip and place glove.6B.300d.txt at {glove_file_path}"
            )

        print("Loading glove.6B embeddings...")
        vectors = []
        with open(glove_file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                self.word_to_idx[word] = len(vectors)
                vectors.append(vector)

        self.embedding_matrix = torch.as_tensor(
            np.stack(vectors),
            device=self.device,
            dtype=torch.float32,
        )
        print(f"Loaded {len(self.word_to_idx)} GloVe vectors on {self.device}.")

    def _as_tensor(self, embedding) -> torch.Tensor:
        if isinstance(embedding, torch.Tensor):
            return embedding.to(device=self.device, dtype=torch.float32).flatten()

        return torch.as_tensor(
            embedding,
            device=self.device,
            dtype=torch.float32,
        ).flatten()

    def _remove_stop_words(self, context: str):
        # Original graph_utils.remove_stop_words:
        # tokens = word_tokenize(context)
        # return [t for t in tokens if t not in STOP_WORDS]
        tokens = word_tokenize(context)
        return [t for t in tokens if t not in self.stop_words]

    def _mean_embedding(self, parts) -> torch.Tensor:
        # Original GloveEmbeddingProvider._get_embedding:
        # part_embeddings = [np.array(self.embeddings[part]) for part in parts if part in self.embeddings]
        # emb = np.mean(part_embeddings, axis=0) if len(part_embeddings) > 0 else np.ones(self.num_dimensions)
        indices = [
            self.word_to_idx[part]
            for part in parts
            if part in self.word_to_idx
        ]

        if len(indices) == 0:
            return self.default_embedding

        index_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        return self.embedding_matrix.index_select(0, index_tensor).mean(dim=0)

    def preload_entities(
        self,
        texts,
        batch_size: int = 4096,
    ) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        unique_texts = list(dict.fromkeys(texts))

        missing_texts = [
            text for text in unique_texts
            if text not in self.entity_cache
        ]

        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start:start + batch_size]

            for text in batch:
                self.entity_cache[text] = self._mean_embedding(text.split(" "))

        return len(missing_texts)

    def preload_questions(self, texts) -> int:
        unique_texts = list(dict.fromkeys(texts))
        missing_texts = [
            text for text in unique_texts
            if text not in self.question_cache
        ]

        for text in missing_texts:
            self.question_cache[text] = self._mean_embedding(
                self._remove_stop_words(text)
            )

        return len(missing_texts)

    def preload_relations(self, texts) -> int:
        unique_texts = list(dict.fromkeys(texts))
        missing_texts = [
            text for text in unique_texts
            if text not in self.relation_cache
        ]

        for text in missing_texts:
            self.relation_cache[text] = self._mean_embedding(
                self._remove_stop_words(text)
            )

        return len(missing_texts)

    def embed_entity(self, text: str) -> torch.Tensor:
        # Original entity embedding:
        # entity.split(" ")
        #
        # Important: no custom tokenizer and no stopword removal here.
        if text not in self.entity_cache:
            self.entity_cache[text] = self._mean_embedding(text.split(" "))

        return self.entity_cache[text]

    def embed_question(self, text: str) -> torch.Tensor:
        # Original question embedding:
        # graph_utils.remove_stop_words(question)
        if text not in self.question_cache:
            self.question_cache[text] = self._mean_embedding(self._remove_stop_words(text))

        return self.question_cache[text]

    def embed_relation(self, text: str) -> torch.Tensor:
        # Original relation/source embedding:
        # if relation is a string, relation.split(" ") happens in original relation_embeddings()
        # for CauseNet sources, graph_sources already stores remove_stop_words(source).
        #
        # In our port, we receive the raw source sentence, so we apply remove_stop_words here.
        if text not in self.relation_cache:
            self.relation_cache[text] = self._mean_embedding(self._remove_stop_words(text))

        return self.relation_cache[text]

    def embed(self, text: str) -> torch.Tensor:
        # Keep old API for path-cost computation.
        return self.embed_entity(text)

    def get_distance(self, embed1, embed2):
        e1 = self._as_tensor(embed1)
        e2 = self._as_tensor(embed2)

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = torch.linalg.vector_norm(e1)
            norm2 = torch.linalg.vector_norm(e2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            return float((1 - torch.dot(e1, e2) / (norm1 * norm2)).item())

        return float(torch.linalg.vector_norm(e1 - e2).item())

    def get_distances(self, embed1, embeddings):
        if not embeddings:
            return []

        e1 = self._as_tensor(embed1)
        matrix = torch.stack([self._as_tensor(embedding) for embedding in embeddings])

        if self.distance_metric == DistanceMetric.COSINE:
            norm1 = torch.linalg.vector_norm(e1)
            norm2 = torch.linalg.vector_norm(matrix, dim=1)
            denominator = norm1 * norm2
            distances = torch.ones(matrix.shape[0], device=self.device, dtype=torch.float32)
            valid = denominator != 0
            distances[valid] = 1 - (matrix[valid] @ e1) / denominator[valid]
        else:
            distances = torch.linalg.vector_norm(matrix - e1, dim=1)

        return distances.detach().cpu().tolist()
