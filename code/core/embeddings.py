import json
import os
import time
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


def _embedding_cache_paths(cache_file):
    cache_file = Path(cache_file)
    cache_stem = cache_file.with_suffix("")

    return {
        "legacy": cache_file,
        "texts": cache_stem.with_name(f"{cache_stem.name}_texts.jsonl"),
        "vectors": cache_stem.with_name(f"{cache_stem.name}_vectors.npy"),
    }


def _move_corrupt_file(path):
    path = Path(path)

    if not path.exists():
        return None

    corrupt_path = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
    os.replace(path, corrupt_path)
    return corrupt_path


def _load_non_pickle_embedding_cache(paths):
    if not paths["texts"].exists() or not paths["vectors"].exists():
        return None

    try:
        vectors = np.load(
            paths["vectors"],
            allow_pickle=False,
            mmap_mode="r",
        )
        with open(paths["texts"], encoding="utf-8") as file:
            texts = [json.loads(line) for line in file]

        if len(texts) != vectors.shape[0]:
            raise ValueError(
                "Cache text/vector length mismatch: "
                f"{len(texts)} texts, {vectors.shape[0]} vectors"
            )

        print(
            "Loaded cached embeddings from "
            f"{paths['texts']} and {paths['vectors']}"
        )
        return texts, vectors
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            "Ignoring corrupt non-pickle embedding cache "
            f"{paths['texts']} / {paths['vectors']}: {exc}."
        )
        for path in (paths["texts"], paths["vectors"]):
            try:
                moved_path = _move_corrupt_file(path)
                if moved_path is not None:
                    print(f"Moved corrupt cache file to {moved_path}.")
            except OSError:
                print(f"Could not move corrupt cache file {path}.")

    return None


def _load_legacy_pickle_embedding_cache(paths, allow_legacy_pickle=True):
    if not allow_legacy_pickle or not paths["legacy"].exists():
        return {}

    try:
        cache = np.load(paths["legacy"], allow_pickle=True).item()
        print(
            "Loaded legacy pickle embedding cache from "
            f"{paths['legacy']}. Resaving will convert it to "
            "the non-pickle cache format."
        )
        return cache
    except (EOFError, OSError, ValueError) as exc:
        try:
            moved_path = _move_corrupt_file(paths["legacy"])
            print(
                "Ignoring corrupt legacy embedding cache "
                f"{paths['legacy']}: {exc}. "
                f"Moved it to {moved_path}."
            )
        except OSError:
            print(
                "Ignoring corrupt legacy embedding cache "
                f"{paths['legacy']}: {exc}. "
                "Could not move the corrupt file."
            )

    return {}


def load_st_embedding_cache(cache_file, allow_legacy_pickle=True):
    paths = _embedding_cache_paths(cache_file)
    non_pickle_cache = _load_non_pickle_embedding_cache(paths)

    if non_pickle_cache is not None:
        texts, vectors = non_pickle_cache
        return {
            text: vectors[index].astype("float32", copy=False)
            for index, text in enumerate(texts)
        }

    return _load_legacy_pickle_embedding_cache(
        paths,
        allow_legacy_pickle=allow_legacy_pickle,
    )


def load_st_embedding_cache_index(cache_file, allow_legacy_pickle=True):
    paths = _embedding_cache_paths(cache_file)
    non_pickle_cache = _load_non_pickle_embedding_cache(paths)

    if non_pickle_cache is not None:
        texts, vectors = non_pickle_cache
        text_to_idx = {
            text: index
            for index, text in enumerate(texts)
        }
        return {}, text_to_idx, vectors

    cache = _load_legacy_pickle_embedding_cache(
        paths,
        allow_legacy_pickle=allow_legacy_pickle,
    )

    return cache, {}, None


def save_st_embedding_cache(
    cache_file,
    cache,
    existing_text_to_idx=None,
    existing_vectors=None,
):
    paths = _embedding_cache_paths(cache_file)
    existing_text_to_idx = existing_text_to_idx or {}
    existing_texts = sorted(
        existing_text_to_idx,
        key=existing_text_to_idx.__getitem__,
    )
    new_texts = [
        text
        for text in cache
        if text not in existing_text_to_idx
    ]
    texts = existing_texts + new_texts

    if existing_vectors is not None and existing_vectors.shape[0] > 0:
        embedding_dim = existing_vectors.shape[1]
    elif new_texts:
        embedding_dim = np.asarray(cache[new_texts[0]], dtype="float32").size
    else:
        embedding_dim = 0

    tmp_texts = paths["texts"].with_name(
        f"{paths['texts'].name}.tmp.{os.getpid()}"
    )
    tmp_vectors = paths["vectors"].with_name(
        f"{paths['vectors'].name}.tmp.{os.getpid()}"
    )

    try:
        vectors = np.lib.format.open_memmap(
            tmp_vectors,
            mode="w+",
            dtype="float32",
            shape=(len(texts), embedding_dim),
        )

        write_start = 0

        if existing_vectors is not None and existing_texts:
            chunk_size = 100_000
            for start in range(0, len(existing_texts), chunk_size):
                end = min(start + chunk_size, len(existing_texts))
                vectors[start:end] = existing_vectors[start:end]

            write_start = len(existing_texts)

        if new_texts:
            chunk_size = 100_000
            for start in range(0, len(new_texts), chunk_size):
                batch_texts = new_texts[start:start + chunk_size]
                batch_vectors = np.stack([
                    np.asarray(cache[text], dtype="float32")
                    for text in batch_texts
                ])
                target_start = write_start + start
                vectors[target_start:target_start + len(batch_texts)] = (
                    batch_vectors
                )

        vectors.flush()
        del vectors

        with open(tmp_texts, "w", encoding="utf-8") as file:
            for text in texts:
                file.write(json.dumps(text, ensure_ascii=False))
                file.write("\n")

        os.replace(tmp_vectors, paths["vectors"])
        os.replace(tmp_texts, paths["texts"])
    finally:
        for path in (tmp_texts, tmp_vectors):
            if path.exists():
                os.remove(path)


def _resolve_device(device=None):
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = str(device)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}', but CUDA is not available.")

    if device == "cpu" or device.startswith("cuda"):
        return device

    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


class STEmbedder:
    def __init__(
        self,
        model_path: str,
        distance_metric: DistanceMetric,
        device=None,
        cache_suffix: str = None,
    ):
        self.device = _resolve_device(device)

        # Store only the last path component as a readable model name.
        self.model_name = os.path.basename(model_path.rstrip('/'))

        self.distance_metric = distance_metric

        # If set, distances will be computed only on the first k embedding dimensions.
        self.matryoshka_dim = None

        # CPU cache is persisted as NumPy. Tensor cache keeps hot embeddings on
        # the active torch device for traversal distance computations.
        self.cache = {}
        self.cache_text_to_idx = {}
        self.cache_vectors = None
        self.tensor_cache = {}
        self.tensor_cache_dim = None
        self.indexed_text_to_idx = {}
        self.embedding_table = None

        # Store embedding caches inside the mounted project directory.
        # This avoids writing to ../data, which resolves outside /app inside the Slurm container.
        cache_dir = DATA_DIR / "embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = self.model_name
        if cache_suffix:
            cache_name = f"{cache_name}_{cache_suffix}"
        self.cache_file = cache_dir / f"{cache_name}_embeddings.npy"

        (
            self.cache,
            self.cache_text_to_idx,
            self.cache_vectors,
        ) = load_st_embedding_cache_index(self.cache_file)

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

    def _active_tensor_dim(self):
        if self.matryoshka_dim is None:
            return None

        model_dim = self.get_model_dim()
        if self.matryoshka_dim < model_dim:
            return self.matryoshka_dim

        return None

    def _trim_to_active_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        active_dim = self._active_tensor_dim()

        if active_dim is not None and tensor.numel() > active_dim:
            return tensor[:active_dim]

        return tensor

    def _trim_matrix_to_active_dim(self, matrix: torch.Tensor) -> torch.Tensor:
        active_dim = self._active_tensor_dim()

        if (
            active_dim is not None
            and matrix.dim() >= 2
            and matrix.shape[1] > active_dim
        ):
            return matrix[:, :active_dim]

        return matrix

    def _as_active_tensor(self, embedding) -> torch.Tensor:
        return self._trim_to_active_dim(self._as_tensor(embedding))

    def _as_active_numpy(self, embedding) -> np.ndarray:
        array = self._as_numpy(embedding)
        active_dim = self._active_tensor_dim()

        if active_dim is not None and array.size > active_dim:
            return array[:active_dim]

        return array

    def _has_cached_embedding(self, text: str) -> bool:
        return text in self.cache or text in self.cache_text_to_idx

    def _get_cached_embedding(self, text: str):
        if text in self.cache:
            return self.cache[text]

        index = self.cache_text_to_idx.get(text)
        if index is None or self.cache_vectors is None:
            return None

        return self.cache_vectors[index]

    def _clear_device_cache(self):
        self.tensor_cache.clear()
        self.indexed_text_to_idx = {}
        self.embedding_table = None

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _ensure_tensor_cache_dim(self):
        active_dim = self._active_tensor_dim()

        if self.tensor_cache_dim != active_dim:
            self._clear_device_cache()
            self.tensor_cache_dim = active_dim

    def set_matryoshka_dim(self, dim: int):
        if dim is not None and dim <= 0:
            raise ValueError("Matryoshka dimension must be greater than 0")

        previous_active_dim = self._active_tensor_dim()
        self.matryoshka_dim = dim
        active_dim = self._active_tensor_dim()

        if previous_active_dim != active_dim:
            self._clear_device_cache()
            self.tensor_cache_dim = active_dim

    def get_model_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def get_active_embedding_dim(self):
        return self._active_tensor_dim() or self.get_model_dim()

    def embed(self, text: str) -> torch.Tensor:
        self._ensure_tensor_cache_dim()

        if text in self.tensor_cache:
            return self.tensor_cache[text]

        index = self.indexed_text_to_idx.get(text)
        if self.embedding_table is not None and index is not None:
            return self.embedding_table.weight[index].flatten()

        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            tensor = self._as_active_tensor(cached_embedding)
            self.tensor_cache[text] = tensor
            return tensor

        with torch.no_grad():
            emb = self.model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            ).detach()

        raw_tensor = self._as_tensor(emb)
        tensor = self._trim_to_active_dim(raw_tensor)
        self.tensor_cache[text] = tensor
        self.cache[text] = self._as_numpy(raw_tensor)
        return tensor

    def embed_numpy(self, text: str) -> np.ndarray:
        cached_embedding = self._get_cached_embedding(text)

        if cached_embedding is None:
            self.embed(text)
            cached_embedding = self._get_cached_embedding(text)

        return self._as_numpy(cached_embedding)

    def preload(self, texts, batch_size: int = 64, save: bool = True) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._ensure_tensor_cache_dim()

        unique_texts = list(dict.fromkeys(texts))

        for text in unique_texts:
            if self._has_cached_embedding(text) and text not in self.tensor_cache:
                cached_embedding = self._get_cached_embedding(text)
                self.tensor_cache[text] = self._as_active_tensor(cached_embedding)

        missing_texts = [
            text
            for text in unique_texts
            if not self._has_cached_embedding(text)
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
                raw_tensor = self._as_tensor(emb)
                tensor = self._trim_to_active_dim(raw_tensor)
                self.tensor_cache[text] = tensor
                self.cache[text] = self._as_numpy(raw_tensor)

        if save:
            self.save_cache()

        return len(missing_texts)

    def prepare_embedding_index(
        self,
        texts,
        batch_size: int = 64,
        save: bool = True,
        discard_tensor_cache: bool = False,
        populate_tensor_cache: bool = True,
        texts_are_unique: bool = False,
    ) -> int:
        """
        Preload texts and expose them through one torch embedding table.

        This is the traversal fast path: successors can be fetched by integer
        index instead of repeatedly collecting tensors and stacking them.
        """
        unique_texts = list(texts) if texts_are_unique else list(dict.fromkeys(texts))
        self._ensure_tensor_cache_dim()

        if not unique_texts:
            self.indexed_text_to_idx = {}
            self.embedding_table = None
            return 0

        if populate_tensor_cache:
            added = self.preload(unique_texts, batch_size=batch_size, save=save)

            matrix = torch.stack([
                self.embed(text)
                for text in unique_texts
            ])
        else:
            dim = self._active_tensor_dim() or self.get_model_dim()
            matrix = torch.empty(
                (len(unique_texts), dim),
                device=self.device,
                dtype=torch.float32,
            )
            added = 0

            index_batch_size = max(batch_size, 4096)
            for start in range(0, len(unique_texts), index_batch_size):
                batch = unique_texts[start:start + index_batch_size]
                cached_positions = []
                cached_texts = []
                missing_positions = []
                missing_texts = []

                for offset, text in enumerate(batch):
                    if self._has_cached_embedding(text):
                        cached_positions.append(start + offset)
                        cached_texts.append(text)
                    else:
                        missing_positions.append(start + offset)
                        missing_texts.append(text)

                if cached_texts:
                    cached_matrix = torch.as_tensor(
                        np.stack([
                            self._as_active_numpy(self._get_cached_embedding(text))
                            for text in cached_texts
                        ]),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    matrix[cached_positions] = cached_matrix

                if missing_texts:
                    for missing_start in range(0, len(missing_texts), batch_size):
                        missing_batch = missing_texts[
                            missing_start:missing_start + batch_size
                        ]
                        missing_batch_positions = missing_positions[
                            missing_start:missing_start + batch_size
                        ]

                        with torch.no_grad():
                            batch_embeddings = self.model.encode(
                                missing_batch,
                                batch_size=batch_size,
                                convert_to_tensor=True,
                                show_progress_bar=False,
                                device=self.device,
                            ).detach()

                        batch_embeddings = batch_embeddings.reshape(
                            batch_embeddings.shape[0],
                            -1,
                        )
                        active_batch_embeddings = self._trim_matrix_to_active_dim(
                            batch_embeddings.to(
                                device=self.device,
                                dtype=torch.float32,
                            )
                        )
                        matrix[missing_batch_positions] = active_batch_embeddings

                        if save:
                            for text, embedding in zip(
                                missing_batch,
                                batch_embeddings,
                            ):
                                self.cache[text] = self._as_numpy(embedding)

                        added += len(missing_batch)

            if save and added:
                self.save_cache()

        self.embedding_table = torch.nn.Embedding.from_pretrained(
            matrix,
            freeze=True,
        ).to(self.device)
        self.embedding_table.eval()
        self.indexed_text_to_idx = {
            text: index
            for index, text in enumerate(unique_texts)
        }

        if discard_tensor_cache:
            if populate_tensor_cache:
                for text in unique_texts:
                    self.tensor_cache.pop(text, None)
            else:
                self.tensor_cache.clear()

        return added

    def has_embedding_index(self) -> bool:
        return self.embedding_table is not None

    def embed_many(self, texts) -> torch.Tensor:
        self._ensure_tensor_cache_dim()

        texts = list(texts)

        if not texts:
            dim = self._active_tensor_dim() or self.get_model_dim()
            return torch.empty((0, dim), device=self.device, dtype=torch.float32)

        if self.embedding_table is not None:
            try:
                indices = [self.indexed_text_to_idx[text] for text in texts]
            except KeyError:
                pass
            else:
                index_tensor = torch.as_tensor(
                    indices,
                    device=self.device,
                    dtype=torch.long,
                )

                with torch.no_grad():
                    return self.embedding_table.weight.index_select(0, index_tensor)

        return torch.stack([
            self.embed(text)
            for text in texts
        ])

    def save_cache(self):
        serializable_cache = {
            text: self._as_numpy(embedding)
            for text, embedding in self.cache.items()
        }
        save_st_embedding_cache(
            self.cache_file,
            serializable_cache,
            existing_text_to_idx=self.cache_text_to_idx,
            existing_vectors=self.cache_vectors,
        )

    def get_distance(self, embed1, embed2):
        e1 = self._trim_to_active_dim(self._as_tensor(embed1))
        e2 = self._trim_to_active_dim(self._as_tensor(embed2))

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
        e1 = self._trim_to_active_dim(self._as_tensor(embed1))
        if isinstance(embeddings, torch.Tensor):
            if embeddings.numel() == 0:
                return []

            matrix = embeddings.to(device=self.device, dtype=torch.float32)

            if matrix.dim() == 1:
                matrix = matrix.unsqueeze(0)
            else:
                matrix = matrix.reshape(matrix.shape[0], -1)
        else:
            embeddings = list(embeddings)

            if not embeddings:
                return []

            matrix = torch.stack([
                self._as_tensor(embedding)
                for embedding in embeddings
            ])

        matrix = self._trim_matrix_to_active_dim(matrix)

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
        device=None,
    ):
        self.distance_metric = distance_metric
        self.device = _resolve_device(device)
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
