from core.constants import FILTERED_DATASETS_DIR

DEFAULT_RUN_SUFFIX = "v3"

EMBEDDING_INDEX_MIN_SUCCESSORS = 16
DEFAULT_EMBEDDING_BATCH_SIZE = 128

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
DEFAULT_P95_CONFIG_SOURCE_DATASET = "msmarco_train"
DEFAULT_P95_CONFIG_SOURCE_GRAPH = "causenet"

DEFAULT_VALIDATION_GRAPH = "causenet"
DEFAULT_TEST_GRAPHS = ("causenet", "ceg", "causenet_full")
DEFAULT_VALIDATION_DATASET = str(
    FILTERED_DATASETS_DIR / "msmarco_valid_filtered.json"
)
DEFAULT_TEST_DATASETS = (
    str(FILTERED_DATASETS_DIR / "msmarco_test_filtered.json"),
    str(FILTERED_DATASETS_DIR / "sem_test_filtered.json"),
)

# Shared settings for evaluating the validation-selected fine-tuned models at
# fixed A* visit budgets. Model descriptors are path-independent so evaluation,
# visualization, and reporting code can reuse the same stable configuration.
BUDGET_TRADEOFF_VISIT_BUDGETS = (
    5,
    10,
    20,
    30,
    40,
    50,
    75,
    100,
    150,
    200,
)

VALIDATION_SELECTED_FINETUNED_MODELS = (
    {
        "model": "FT A*: MPNet",
        "checkpoint_name": (
            "all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v3_finetuned"
        ),
        "embedding_dimension": 128,
        "activation_function": "ReLU",
        "distance_metric": "Cosine",
        "existing_validation_budget": 34,
    },
    {
        "model": "FT A*: BGE",
        "checkpoint_name": (
            "bge-large-en-v1.5_relu_euclid_nonorm_matryoshka_v3_finetuned"
        ),
        "embedding_dimension": 1024,
        "activation_function": "ReLU",
        "distance_metric": "Euclidean",
        "existing_validation_budget": 191,
    },
    {
        "model": "FT A*: mxbai",
        "checkpoint_name": (
            "mxbai-embed-large-v1_relu_euclid_nonorm_matryoshka_v3_finetuned"
        ),
        "embedding_dimension": 768,
        "activation_function": "ReLU",
        "distance_metric": "Euclidean",
        "existing_validation_budget": 154,
    },
    {
        "model": "FT A*: Qwen",
        "checkpoint_name": (
            "Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_v3_finetuned"
        ),
        "embedding_dimension": 32,
        "activation_function": "ReLU",
        "distance_metric": "Euclidean",
        "existing_validation_budget": 39,
    },
    {
        "model": "FT A*: Granite",
        "checkpoint_name": (
            "granite-embedding-english-r2_relu_euclid_nonorm_"
            "matryoshka_v3_finetuned"
        ),
        "embedding_dimension": 32,
        "activation_function": "ReLU",
        "distance_metric": "Euclidean",
        "existing_validation_budget": 27,
    },
)
