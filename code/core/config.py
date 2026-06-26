from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LIGHTNING_DIR = DATA_DIR / "models" / "lightning"
GLOVE_300D_PATH = EMBEDDINGS_DIR / "glove.6B" / "glove.6B.300d.txt"

DEFAULT_RUN_SUFFIX = "v3"

EMBEDDING_INDEX_MIN_SUCCESSORS = 16

BASE_MODELS = (
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "ibm-granite/granite-embedding-english-r2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen3-Embedding-0.6B",
)

DEFAULT_ABLATION_BASE_MODEL_NAME = "granite-embedding-english-r2"
DEFAULT_ABLATION_NORMALIZE_STR = "nonorm"
DEFAULT_ABLATION_MRL_STR = "matryoshka"
DEFAULT_ABLATION_REFERENCE_COMBO = ("relu", "euclid")
DEFAULT_ABLATION_COMBOS = (
    ("relu", "cosine"),
    ("gelu", "euclid"),
    ("gelu", "cosine"),
)
DEFAULT_ABLATION_CAP_SOURCE_DATASET = "msmarco_train"
DEFAULT_ABLATION_CAP_SOURCE_GRAPH = "causenet"

DEFAULT_VALIDATION_GRAPH = "causenet"
DEFAULT_TEST_GRAPHS = ("causenet", "causalbank", "causenet_full")
DEFAULT_VALIDATION_DATASET = "data/datasets/filtered/msmarco_valid_filtered.json"
DEFAULT_TEST_DATASETS = (
    "data/datasets/filtered/msmarco_test_filtered.json",
    "data/datasets/filtered/sem_test_filtered.json",
)
