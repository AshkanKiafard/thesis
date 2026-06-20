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
from core.utils import (
    get_node_universe_for_cache_suffix,
    get_node_universe_path,
    normalize_node_universe,
    read_node_universe,
    write_node_universe,
)

# Embedding caches are NumPy .npy vector matrices plus one shared JSONL row-label
# file per node universe. The JSONL is deliberately not model-specific: vectors
# differ by model/dimension, but node ordering does not.
# code/core/embeddings.py -> repo root is two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"


class EmbeddingCacheValidationError(ValueError):
    pass


def _embedding_cache_paths(cache_file, node_universe=None):
    cache_file = Path(cache_file)
    cache_stem = cache_file.with_suffix("")
    node_universe = normalize_node_universe(node_universe)

    return {
        "texts": get_node_universe_path(cache_file.parent, node_universe),
        "vectors": cache_stem.with_name(f"{cache_stem.name}_vectors.npy"),
    }


def _embedding_checkpoint_paths(cache_file):
    cache_file = Path(cache_file)
    cache_stem = cache_file.with_suffix("")

    return {
        "meta": cache_stem.with_name(
            f"{cache_stem.name}_checkpoint_meta.json"
        ),
        "vectors": cache_stem.with_name(
            f"{cache_stem.name}_checkpoint_vectors.npy"
        ),
        "mask": cache_stem.with_name(
            f"{cache_stem.name}_checkpoint_mask.npy"
        ),
    }


def _move_corrupt_file(path):
    path = Path(path)

    if not path.exists():
        return None

    corrupt_path = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
    os.replace(path, corrupt_path)
    return corrupt_path


def _close_memmap(array):
    mmap = getattr(array, "_mmap", None)

    if mmap is not None:
        mmap.close()


def _load_embedding_cache_files(paths, node_universe=None, strict=False):
    if not paths["vectors"].exists():
        if paths["texts"].exists():
            print(
                "Embedding vectors missing for node universe "
                f"'{node_universe}': {paths['vectors']}"
            )
        return None

    if not paths["texts"].exists():
        message = (
            "Embedding node-order JSONL missing for node universe "
            f"'{node_universe}': {paths['texts']}"
        )
        if strict:
            raise FileNotFoundError(message)

        print(message)
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
            vector_rows = vectors.shape[0]
            _close_memmap(vectors)
            message = (
                "Embedding cache row-count mismatch for node universe "
                f"'{node_universe}': {paths['texts']} has "
                f"{len(texts):,} rows, but {paths['vectors']} has "
                f"{vector_rows:,} rows."
            )
            if strict:
                raise EmbeddingCacheValidationError(message)

            print(f"Ignoring invalid embedding cache. {message}")
            return None

        print(
            "Loaded cached embeddings from "
            f"{paths['texts']} and {paths['vectors']}"
        )
        return texts, vectors
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message = (
            "Ignoring corrupt JSONL/mmap embedding cache "
            f"{paths['texts']} / {paths['vectors']}: {exc}."
        )
        if strict:
            raise EmbeddingCacheValidationError(message) from exc

        print(message)
        try:
            moved_path = _move_corrupt_file(paths["vectors"])
            if moved_path is not None:
                print(f"Moved corrupt vector cache file to {moved_path}.")
        except OSError:
            print(f"Could not move corrupt vector cache file {paths['vectors']}.")

    return None


def load_st_embedding_cache(cache_file, node_universe=None, strict=False):
    node_universe = normalize_node_universe(node_universe)
    paths = _embedding_cache_paths(cache_file, node_universe)
    cache_files = _load_embedding_cache_files(
        paths,
        node_universe=node_universe,
        strict=strict,
    )

    if cache_files is not None:
        texts, vectors = cache_files
        try:
            return {
                text: np.array(vectors[index], dtype="float32", copy=True)
                for index, text in enumerate(texts)
            }
        finally:
            _close_memmap(vectors)

    return {}


def load_st_embedding_cache_index(cache_file, node_universe=None, strict=False):
    node_universe = normalize_node_universe(node_universe)
    paths = _embedding_cache_paths(cache_file, node_universe)
    cache_files = _load_embedding_cache_files(
        paths,
        node_universe=node_universe,
        strict=strict,
    )

    if cache_files is not None:
        texts, vectors = cache_files
        text_to_idx = {
            text: index
            for index, text in enumerate(texts)
        }
        return {}, text_to_idx, vectors

    return {}, {}, None


def get_embedding_cache_status(cache_file, node_universe=None):
    node_universe = normalize_node_universe(node_universe)
    paths = _embedding_cache_paths(cache_file, node_universe)
    status = {
        "node_universe": node_universe,
        "nodes_path": paths["texts"],
        "vectors_path": paths["vectors"],
        "nodes_exists": paths["texts"].exists(),
        "vectors_exists": paths["vectors"].exists(),
        "nodes_count": None,
        "vectors_shape": None,
        "covered": False,
        "reason": None,
    }

    if status["nodes_exists"]:
        with open(paths["texts"], encoding="utf-8") as file:
            status["nodes_count"] = sum(1 for _ in file)
    else:
        status["reason"] = f"missing universal node JSONL: {paths['texts']}"

    if status["vectors_exists"]:
        vectors = np.load(paths["vectors"], allow_pickle=False, mmap_mode="r")
        try:
            status["vectors_shape"] = tuple(vectors.shape)
        finally:
            _close_memmap(vectors)
    elif status["reason"] is None:
        status["reason"] = f"missing embedding vectors: {paths['vectors']}"

    if status["nodes_count"] is not None and status["vectors_shape"] is not None:
        if status["nodes_count"] == status["vectors_shape"][0]:
            status["covered"] = True
            status["reason"] = None
        else:
            status["reason"] = (
                "row-count mismatch: "
                f"{status['nodes_count']:,} node rows vs "
                f"{status['vectors_shape'][0]:,} vector rows"
            )

    return status


def save_st_embedding_cache(
    cache_file,
    cache,
    existing_text_to_idx=None,
    existing_vectors=None,
    node_universe=None,
    node_order=None,
):
    node_universe = normalize_node_universe(node_universe)
    paths = _embedding_cache_paths(cache_file, node_universe)
    existing_text_to_idx = existing_text_to_idx or {}

    if node_order is None:
        if not paths["texts"].exists():
            raise FileNotFoundError(
                "Cannot save embedding cache without a node-order JSONL. "
                f"Missing {paths['texts']}. Pass node_order when creating "
                "a new node universe."
            )
        node_order = read_node_universe(paths["texts"])
    else:
        node_order = list(node_order)
        if paths["texts"].exists():
            existing_order = read_node_universe(paths["texts"])
            if existing_order != node_order:
                raise EmbeddingCacheValidationError(
                    "Refusing to overwrite embedding node-order JSONL with "
                    f"a different order: {paths['texts']}"
                )
        else:
            write_node_universe(paths["texts"], node_order)

    node_to_idx = {
        text: index
        for index, text in enumerate(node_order)
    }
    unknown_texts = [
        text
        for text in cache
        if text not in node_to_idx
    ]
    if unknown_texts:
        raise EmbeddingCacheValidationError(
            "Embedding cache contains texts outside node universe "
            f"'{node_universe}'. First unknown text: {unknown_texts[0]!r}"
        )

    if existing_vectors is not None:
        if existing_vectors.shape[0] != len(node_order):
            raise EmbeddingCacheValidationError(
                "Existing embedding vector row count does not match node "
                f"universe '{node_universe}': {existing_vectors.shape[0]:,} "
                f"vectors for {len(node_order):,} nodes."
            )

    if existing_vectors is not None and existing_vectors.shape[0] > 0:
        embedding_dim = existing_vectors.shape[1]
    elif cache:
        first_text = next(iter(cache))
        embedding_dim = np.asarray(cache[first_text], dtype="float32").size
    else:
        embedding_dim = 0

    missing_texts = [
        text
        for text in node_order
        if text not in existing_text_to_idx and text not in cache
    ]
    if missing_texts:
        raise EmbeddingCacheValidationError(
            "Cannot save partial embedding cache for node universe "
            f"'{node_universe}'. Missing {len(missing_texts):,} vectors; "
            f"first missing text: {missing_texts[0]!r}"
        )

    tmp_vectors = paths["vectors"].with_name(
        f"{paths['vectors'].name}.tmp.{os.getpid()}"
    )

    try:
        vectors = np.lib.format.open_memmap(
            tmp_vectors,
            mode="w+",
            dtype="float32",
            shape=(len(node_order), embedding_dim),
        )

        if existing_vectors is not None and len(node_order):
            chunk_size = 100_000
            for start in range(0, len(node_order), chunk_size):
                end = min(start + chunk_size, len(node_order))
                vectors[start:end] = existing_vectors[start:end]

        if cache:
            chunk_size = 100_000
            cache_texts = sorted(cache, key=node_to_idx.__getitem__)
            for start in range(0, len(cache_texts), chunk_size):
                batch_texts = cache_texts[start:start + chunk_size]
                batch_vectors = np.stack([
                    np.asarray(cache[text], dtype="float32")
                    for text in batch_texts
                ])
                target_indices = [node_to_idx[text] for text in batch_texts]
                vectors[target_indices] = batch_vectors

        vectors.flush()
        del vectors
        _close_memmap(existing_vectors)

        os.replace(tmp_vectors, paths["vectors"])
    finally:
        if tmp_vectors.exists():
            os.remove(tmp_vectors)


def _move_checkpoint_as_corrupt(paths):
    for path in paths.values():
        try:
            moved_path = _move_corrupt_file(path)
            if moved_path is not None:
                print(f"Moved stale embedding checkpoint file to {moved_path}.")
        except OSError:
            print(f"Could not move stale embedding checkpoint file {path}.")


def _load_embedding_checkpoint(cache_file, texts, dim, node_universe):
    paths = _embedding_checkpoint_paths(cache_file)
    required_paths = tuple(paths.values())

    if not any(path.exists() for path in required_paths):
        return None

    if not all(path.exists() for path in required_paths):
        print(
            "Ignoring incomplete embedding checkpoint "
            f"{paths['meta']} / {paths['vectors']} / {paths['mask']}."
        )
        _move_checkpoint_as_corrupt(paths)
        return None

    vectors = None
    mask = None

    try:
        vectors = np.load(
            paths["vectors"],
            allow_pickle=False,
            mmap_mode="r+",
        )
        mask = np.load(
            paths["mask"],
            allow_pickle=False,
            mmap_mode="r+",
        )

        expected_vector_shape = (len(texts), dim)
        expected_mask_shape = (len(texts),)

        if vectors.shape != expected_vector_shape:
            raise ValueError(
                "checkpoint vector shape mismatch: "
                f"{vectors.shape} != {expected_vector_shape}"
            )
        if mask.shape != expected_mask_shape:
            raise ValueError(
                "checkpoint mask shape mismatch: "
                f"{mask.shape} != {expected_mask_shape}"
            )
        if mask.dtype != np.dtype("bool"):
            raise ValueError(f"checkpoint mask dtype is {mask.dtype}, not bool")

        with open(paths["meta"], encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("node_universe") != node_universe:
            raise ValueError(
                "checkpoint node universe mismatch: "
                f"{metadata.get('node_universe')} != {node_universe}"
            )
        if metadata.get("num_texts") != len(texts):
            raise ValueError(
                "checkpoint node count mismatch: "
                f"{metadata.get('num_texts')} != {len(texts)}"
            )
        if metadata.get("dim") != dim:
            raise ValueError(
                f"checkpoint dim mismatch: {metadata.get('dim')} != {dim}"
            )

        filled_rows = int(np.count_nonzero(mask))
        print(
            "Loaded embedding checkpoint: "
            f"{filled_rows:,}/{len(texts):,} rows already written "
            f"from {paths['vectors']}.",
            flush=True,
        )
        return paths, vectors, mask
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            "Ignoring stale embedding checkpoint "
            f"{paths['meta']} / {paths['vectors']} / {paths['mask']}: {exc}."
        )
        if vectors is not None:
            _close_memmap(vectors)
        if mask is not None:
            _close_memmap(mask)
        _move_checkpoint_as_corrupt(paths)

    return None


def _create_embedding_checkpoint(cache_file, texts, dim, node_universe):
    paths = _embedding_checkpoint_paths(cache_file)
    tmp_meta = paths["meta"].with_name(
        f"{paths['meta'].name}.tmp.{os.getpid()}"
    )

    try:
        with open(tmp_meta, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "node_universe": node_universe,
                    "num_texts": len(texts),
                    "dim": dim,
                },
                file,
                indent=2,
            )
        os.replace(tmp_meta, paths["meta"])

        vectors = np.lib.format.open_memmap(
            paths["vectors"],
            mode="w+",
            dtype="float32",
            shape=(len(texts), dim),
        )
        mask = np.lib.format.open_memmap(
            paths["mask"],
            mode="w+",
            dtype="bool",
            shape=(len(texts),),
        )
        mask[:] = False
        mask.flush()
        vectors.flush()

        print(
            "Created embedding checkpoint: "
            f"{paths['vectors']} with {len(texts):,} rows.",
            flush=True,
        )
        return paths, vectors, mask
    finally:
        if tmp_meta.exists():
            os.remove(tmp_meta)


def _open_embedding_checkpoint(cache_file, texts, dim, node_universe):
    loaded_checkpoint = _load_embedding_checkpoint(
        cache_file,
        texts,
        dim,
        node_universe,
    )

    if loaded_checkpoint is not None:
        return loaded_checkpoint

    return _create_embedding_checkpoint(cache_file, texts, dim, node_universe)


def _promote_embedding_checkpoint(cache_file, checkpoint_paths, node_universe):
    cache_paths = _embedding_cache_paths(cache_file, node_universe)
    os.replace(checkpoint_paths["vectors"], cache_paths["vectors"])

    if checkpoint_paths["mask"].exists():
        os.remove(checkpoint_paths["mask"])
    if checkpoint_paths["meta"].exists():
        os.remove(checkpoint_paths["meta"])

    return cache_paths


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
        node_universe: str = None,
    ):
        self.device = _resolve_device(device)
        if node_universe is None:
            node_universe = get_node_universe_for_cache_suffix(cache_suffix)
        self.node_universe = normalize_node_universe(node_universe)

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
        self.embedding_table_dim = None

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
        ) = load_st_embedding_cache_index(
            self.cache_file,
            node_universe=self.node_universe,
        )

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

    def _normalize_vector_for_cosine(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.distance_metric != DistanceMetric.COSINE:
            return tensor

        return torch.nn.functional.normalize(tensor, p=2, dim=0)

    def _normalize_matrix_for_cosine(
        self,
        matrix: torch.Tensor,
        inplace: bool = False,
    ) -> torch.Tensor:
        if self.distance_metric != DistanceMetric.COSINE:
            return matrix

        if not inplace:
            return torch.nn.functional.normalize(matrix, p=2, dim=1)

        norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
        return matrix.div_(norms.clamp_min_(1e-12))

    def _as_active_tensor(self, embedding) -> torch.Tensor:
        return self._normalize_vector_for_cosine(
            self._trim_to_active_dim(self._as_tensor(embedding))
        )

    def _as_active_numpy(self, embedding) -> np.ndarray:
        array = self._as_numpy(embedding)
        active_dim = self._active_tensor_dim()

        if active_dim is not None and array.size > active_dim:
            return array[:active_dim]

        return array

    def _has_cached_embedding(self, text: str) -> bool:
        return text in self.cache or text in self.cache_text_to_idx

    def has_cached_embedding(self, text: str) -> bool:
        return self._has_cached_embedding(text)

    def _get_cached_embedding(self, text: str):
        if text in self.cache:
            return self.cache[text]

        index = self.cache_text_to_idx.get(text)
        if index is None or self.cache_vectors is None:
            return None

        return self.cache_vectors[index]

    def _embedding_index_covers_active_dim(self):
        if self.embedding_table is None or self.embedding_table_dim is None:
            return False

        required_dim = self._active_tensor_dim() or self.get_model_dim()
        return self.embedding_table_dim >= required_dim

    def _clear_device_cache(self, clear_embedding_index=True):
        had_tensor_cache = bool(self.tensor_cache)
        had_embedding_index = self.embedding_table is not None

        self.tensor_cache.clear()
        if clear_embedding_index:
            self.indexed_text_to_idx = {}
            self.embedding_table = None
            self.embedding_table_dim = None

        if (
            self.device.startswith("cuda")
            and (
                had_tensor_cache
                or (clear_embedding_index and had_embedding_index)
            )
        ):
            torch.cuda.empty_cache()

    def _ensure_tensor_cache_dim(self):
        active_dim = self._active_tensor_dim()

        if self.tensor_cache_dim != active_dim:
            self._clear_device_cache(
                clear_embedding_index=not self._embedding_index_covers_active_dim()
            )
            self.tensor_cache_dim = active_dim

    def set_matryoshka_dim(self, dim: int):
        if dim is not None and dim <= 0:
            raise ValueError("Matryoshka dimension must be greater than 0")

        previous_active_dim = self._active_tensor_dim()
        self.matryoshka_dim = dim
        active_dim = self._active_tensor_dim()

        if previous_active_dim != active_dim:
            self._clear_device_cache(
                clear_embedding_index=not self._embedding_index_covers_active_dim()
            )
            self.tensor_cache_dim = active_dim

    def get_model_dim(self):
        return self.model.get_sentence_embedding_dimension()

    def get_active_embedding_dim(self):
        return self._active_tensor_dim() or self.get_model_dim()

    def get_dimension_cache_file(self, dim: int) -> Path:
        suffix = f"_dim{dim}_embeddings.npy"
        name = self.cache_file.name

        if name.endswith("_embeddings.npy"):
            return self.cache_file.with_name(
                f"{name[:-len('_embeddings.npy')]}{suffix}"
            )

        return self.cache_file.with_name(f"{self.cache_file.stem}_dim{dim}.npy")

    def get_node_universe_file(self) -> Path:
        return get_node_universe_path(self.cache_file.parent, self.node_universe)

    def _get_saved_node_order(self):
        node_file = self.get_node_universe_file()
        if not node_file.exists():
            return None

        return read_node_universe(node_file)

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
        tensor = self._normalize_vector_for_cosine(
            self._trim_to_active_dim(raw_tensor)
        )
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
                tensor = self._normalize_vector_for_cosine(
                    self._trim_to_active_dim(raw_tensor)
                )
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
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        requested_texts = (
            list(texts) if texts_are_unique else list(dict.fromkeys(texts))
        )
        unique_texts = requested_texts
        self._ensure_tensor_cache_dim()
        active_dim = self._active_tensor_dim() or self.get_model_dim()
        progress_start_time = time.time()

        saved_node_order = self._get_saved_node_order()
        if saved_node_order is not None:
            saved_node_set = set(saved_node_order)
            missing_from_universe = [
                text
                for text in requested_texts
                if text not in saved_node_set
            ]
            if missing_from_universe:
                raise EmbeddingCacheValidationError(
                    "Embedding index request contains text outside node "
                    f"universe '{self.node_universe}'. First unknown text: "
                    f"{missing_from_universe[0]!r}"
                )

            if save:
                unique_texts = saved_node_order
                texts_are_unique = True
        elif save:
            raise FileNotFoundError(
                "Cannot save embedding vectors without the universal "
                f"node-order JSONL for '{self.node_universe}'. Missing "
                f"{self.get_node_universe_file()}. Run pre-embedding to "
                "create the node universe first."
            )

        total_texts = len(unique_texts)

        print(
            "Preparing embedding index: "
            f"{total_texts:,} texts, active dim {active_dim}, "
            f"node universe {self.node_universe}, "
            f"batch size {batch_size}, "
            f"populate tensor cache {populate_tensor_cache}.",
            flush=True,
        )

        if not unique_texts:
            self.indexed_text_to_idx = {}
            self.embedding_table = None
            print("Embedding index preparation skipped: no texts.", flush=True)
            return 0

        if populate_tensor_cache:
            cached_before = sum(
                1
                for text in unique_texts
                if self._has_cached_embedding(text)
            )
            missing_before = total_texts - cached_before
            print(
                "Embedding index preload source summary: "
                f"{cached_before:,} cached, {missing_before:,} to encode.",
                flush=True,
            )

            added = self.preload(unique_texts, batch_size=batch_size, save=save)
            print(
                "Embedding index preload complete: "
                f"{added:,} newly encoded, "
                f"{time.time() - progress_start_time:.1f}s elapsed.",
                flush=True,
            )

            tensors = []
            stack_batch_size = max(batch_size, 4096)
            for start in range(0, total_texts, stack_batch_size):
                batch = unique_texts[start:start + stack_batch_size]
                tensors.extend(self.embed(text) for text in batch)

                done = min(start + len(batch), total_texts)
                print(
                    "Embedding index stack progress: "
                    f"{done:,}/{total_texts:,} rows "
                    f"({done / total_texts:.1%}), "
                    f"{time.time() - progress_start_time:.1f}s elapsed.",
                    flush=True,
                )

            matrix = torch.stack(tensors)
        else:
            dim = active_dim
            matrix = torch.empty(
                (len(unique_texts), dim),
                device=self.device,
                dtype=torch.float32,
            )
            added = 0
            dimension_cache_hits = 0
            full_cache_hits = 0
            encoded_rows = 0
            dimension_cache_text_to_idx = {}
            dimension_cache_vectors = None
            checkpoint_paths = None
            checkpoint_vectors = None
            checkpoint_mask = None
            checkpoint_hits = 0

            if self._active_tensor_dim() is not None:
                dimension_cache_file = self.get_dimension_cache_file(dim)
                (
                    _,
                    dimension_cache_text_to_idx,
                    dimension_cache_vectors,
                ) = load_st_embedding_cache_index(
                    dimension_cache_file,
                    node_universe=self.node_universe,
                )

                has_complete_dimension_cache = all(
                    text in dimension_cache_text_to_idx
                    for text in unique_texts
                )

                if save and not has_complete_dimension_cache:
                    (
                        checkpoint_paths,
                        checkpoint_vectors,
                        checkpoint_mask,
                    ) = _open_embedding_checkpoint(
                        dimension_cache_file,
                        unique_texts,
                        dim,
                        self.node_universe,
                    )

            dimension_cache_total = 0
            checkpoint_total = 0
            full_cache_total = 0
            missing_total = 0
            for position, text in enumerate(unique_texts):
                if text in dimension_cache_text_to_idx:
                    dimension_cache_total += 1
                elif (
                    checkpoint_mask is not None
                    and bool(checkpoint_mask[position])
                ):
                    checkpoint_total += 1
                elif self._has_cached_embedding(text):
                    full_cache_total += 1
                else:
                    missing_total += 1

            print(
                "Embedding index source summary: "
                f"{dimension_cache_total:,} dim-cache rows, "
                f"{checkpoint_total:,} checkpoint rows, "
                f"{full_cache_total:,} full-cache rows, "
                f"{missing_total:,} rows to encode.",
                flush=True,
            )

            index_batch_size = max(batch_size, 4096)
            encode_report_interval = max(batch_size, 4096)
            next_encode_report = (
                min(encode_report_interval, missing_total)
                if missing_total
                else 0
            )
            print(
                "Embedding index fill started: "
                f"row batch size {index_batch_size:,}.",
                flush=True,
            )

            try:
                for start in range(0, len(unique_texts), index_batch_size):
                    batch = unique_texts[start:start + index_batch_size]
                    dimension_cached_positions = []
                    dimension_cached_texts = []
                    checkpoint_cached_positions = []
                    full_cached_positions = []
                    full_cached_texts = []
                    missing_positions = []
                    missing_texts = []

                    for offset, text in enumerate(batch):
                        position = start + offset

                        if text in dimension_cache_text_to_idx:
                            dimension_cached_positions.append(position)
                            dimension_cached_texts.append(text)
                        elif (
                            checkpoint_mask is not None
                            and bool(checkpoint_mask[position])
                        ):
                            checkpoint_cached_positions.append(position)
                        elif self._has_cached_embedding(text):
                            full_cached_positions.append(position)
                            full_cached_texts.append(text)
                        else:
                            missing_positions.append(position)
                            missing_texts.append(text)

                    if dimension_cached_texts:
                        cached_matrix_np = np.stack([
                            dimension_cache_vectors[
                                dimension_cache_text_to_idx[text]
                            ].astype("float32", copy=False)
                            for text in dimension_cached_texts
                        ])
                        cached_matrix = torch.as_tensor(
                            cached_matrix_np,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        matrix[dimension_cached_positions] = cached_matrix

                        if checkpoint_vectors is not None:
                            checkpoint_vectors[dimension_cached_positions] = (
                                cached_matrix_np
                            )
                            checkpoint_mask[dimension_cached_positions] = True
                        dimension_cache_hits += len(dimension_cached_texts)

                    if checkpoint_cached_positions:
                        cached_matrix_np = checkpoint_vectors[
                            checkpoint_cached_positions
                        ].astype("float32", copy=False)
                        cached_matrix = torch.as_tensor(
                            cached_matrix_np,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        matrix[checkpoint_cached_positions] = cached_matrix
                        checkpoint_hits += len(checkpoint_cached_positions)

                    if full_cached_texts:
                        full_cached_matrix_np = np.stack([
                            self._as_active_numpy(self._get_cached_embedding(text))
                            for text in full_cached_texts
                        ])
                        full_cached_matrix = torch.as_tensor(
                            full_cached_matrix_np,
                            device=self.device,
                            dtype=torch.float32,
                        )
                        matrix[full_cached_positions] = full_cached_matrix

                        if checkpoint_vectors is not None:
                            checkpoint_vectors[full_cached_positions] = (
                                full_cached_matrix_np
                            )
                            checkpoint_mask[full_cached_positions] = True
                        full_cache_hits += len(full_cached_texts)

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
                            matrix[missing_batch_positions] = (
                                active_batch_embeddings
                            )

                            if checkpoint_vectors is not None:
                                active_batch_embeddings_np = (
                                    active_batch_embeddings
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype("float32", copy=False)
                                )
                                checkpoint_vectors[missing_batch_positions] = (
                                    active_batch_embeddings_np
                                )
                                checkpoint_mask[missing_batch_positions] = True
                            elif save:
                                for text, embedding in zip(
                                    missing_batch,
                                    batch_embeddings,
                                ):
                                    self.cache[text] = self._as_numpy(embedding)

                            added += len(missing_batch)
                            encoded_rows += len(missing_batch)

                            if (
                                missing_total
                                and encoded_rows >= next_encode_report
                            ):
                                rows_filled = (
                                    dimension_cache_hits
                                    + checkpoint_hits
                                    + full_cache_hits
                                    + encoded_rows
                                )
                                print(
                                    "Embedding index encode progress: "
                                    f"{encoded_rows:,}/{missing_total:,} "
                                    "missing rows encoded "
                                    f"({encoded_rows / missing_total:.1%}), "
                                    f"{rows_filled:,}/{total_texts:,} total "
                                    "rows filled, "
                                    f"{time.time() - progress_start_time:.1f}s "
                                    "elapsed.",
                                    flush=True,
                                )
                                while next_encode_report <= encoded_rows:
                                    next_encode_report += encode_report_interval

                    done = min(start + len(batch), total_texts)
                    print(
                        "Embedding index fill progress: "
                        f"{done:,}/{total_texts:,} rows "
                        f"({done / total_texts:.1%}), "
                        f"{encoded_rows:,} encoded, "
                        f"{dimension_cache_hits:,} dim-cache hits, "
                        f"{checkpoint_hits:,} checkpoint hits, "
                        f"{full_cache_hits:,} full-cache hits, "
                        f"{time.time() - progress_start_time:.1f}s elapsed.",
                        flush=True,
                    )

                    if checkpoint_vectors is not None:
                        checkpoint_vectors.flush()
                        checkpoint_mask.flush()

                if checkpoint_vectors is not None:
                    checkpoint_vectors.flush()
                    checkpoint_mask.flush()
                    checkpoint_filled = int(np.count_nonzero(checkpoint_mask))

                    if checkpoint_filled != total_texts:
                        raise RuntimeError(
                            "Embedding checkpoint incomplete after fill: "
                            f"{checkpoint_filled:,}/{total_texts:,} rows."
                        )

                    _close_memmap(checkpoint_vectors)
                    _close_memmap(checkpoint_mask)
                    checkpoint_vectors = None
                    checkpoint_mask = None
                    _close_memmap(dimension_cache_vectors)
                    dimension_cache_vectors = None
                    cache_paths = _promote_embedding_checkpoint(
                        dimension_cache_file,
                        checkpoint_paths,
                        self.node_universe,
                    )
                    print(
                        "Saved sliced embedding cache for dim "
                        f"{dim}: {cache_paths['vectors']}",
                        flush=True,
                    )
                elif save and added:
                    self.save_cache()
            finally:
                if dimension_cache_vectors is not None:
                    _close_memmap(dimension_cache_vectors)

                if checkpoint_vectors is not None:
                    checkpoint_vectors.flush()
                    _close_memmap(checkpoint_vectors)

                if checkpoint_mask is not None:
                    checkpoint_mask.flush()
                    _close_memmap(checkpoint_mask)

        runtime_matrix = self._normalize_matrix_for_cosine(matrix, inplace=True)

        self.embedding_table = torch.nn.Embedding.from_pretrained(
            runtime_matrix,
            freeze=True,
        ).to(self.device)
        self.embedding_table.eval()
        self.embedding_table_dim = int(runtime_matrix.shape[1])
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

        print(
            "Embedding index ready: "
            f"{total_texts:,} rows, {added:,} newly encoded, "
            f"{time.time() - progress_start_time:.1f}s elapsed.",
            flush=True,
        )

        return added

    def has_embedding_index(self) -> bool:
        return self.embedding_table is not None

    def has_normalized_runtime_embeddings(self) -> bool:
        if self.distance_metric != DistanceMetric.COSINE:
            return False

        active_dim = self._active_tensor_dim() or self.get_model_dim()
        return self.embedding_table_dim in {None, active_dim}

    def embed_index(self, index: int) -> torch.Tensor:
        if self.embedding_table is None:
            raise ValueError("Embedding index has not been prepared.")

        return self._trim_to_active_dim(self.embedding_table.weight[index].flatten())

    def embed_indices(self, indices) -> torch.Tensor:
        if self.embedding_table is None:
            raise ValueError("Embedding index has not been prepared.")

        index_tensor = torch.as_tensor(
            indices,
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            return self._trim_matrix_to_active_dim(
                self.embedding_table.weight.index_select(0, index_tensor)
            )

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
                return self.embed_indices(indices)

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
            node_universe=self.node_universe,
        )
        (
            self.cache,
            self.cache_text_to_idx,
            self.cache_vectors,
        ) = load_st_embedding_cache_index(
            self.cache_file,
            node_universe=self.node_universe,
        )

    def get_distance(self, embed1, embed2):
        e1 = self._trim_to_active_dim(self._as_tensor(embed1))
        e2 = self._trim_to_active_dim(self._as_tensor(embed2))

        if self.distance_metric == DistanceMetric.COSINE:
            e1 = self._normalize_vector_for_cosine(e1)
            e2 = self._normalize_vector_for_cosine(e2)
            return float((1 - torch.dot(e1, e2)).item())

        return float(torch.linalg.vector_norm(e1 - e2).item())

    def get_distances(self, embed1, embeddings, assume_normalized=False):
        e1 = self._trim_to_active_dim(self._as_tensor(embed1))
        embeddings_is_tensor = isinstance(embeddings, torch.Tensor)
        if embeddings_is_tensor:
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
            if not assume_normalized:
                e1 = self._normalize_vector_for_cosine(e1)
                matrix = self._normalize_matrix_for_cosine(matrix)

            distances = 1 - (matrix @ e1)
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
            e1 = torch.nn.functional.normalize(e1, p=2, dim=0)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=0)
            return float((1 - torch.dot(e1, e2)).item())

        return float(torch.linalg.vector_norm(e1 - e2).item())

    def get_distances(self, embed1, embeddings, assume_normalized=False):
        if not embeddings:
            return []

        e1 = self._as_tensor(embed1)
        matrix = torch.stack([self._as_tensor(embedding) for embedding in embeddings])

        if self.distance_metric == DistanceMetric.COSINE:
            if not assume_normalized:
                e1 = torch.nn.functional.normalize(e1, p=2, dim=0)
                matrix = torch.nn.functional.normalize(matrix, p=2, dim=1)
            distances = 1 - (matrix @ e1)
        else:
            distances = torch.linalg.vector_norm(matrix - e1, dim=1)

        return distances.detach().cpu().tolist()
