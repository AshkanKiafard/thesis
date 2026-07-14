from enum import Enum
from pathlib import Path


# Repository and data directories
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DATASETS_DIR = DATA_DIR / "datasets"
FILTERED_DATASETS_DIR = DATASETS_DIR / "filtered"
EVALUATION_DIR = DATA_DIR / "evaluation"
ABLATION_EVALUATION_DIR = EVALUATION_DIR / "ablation"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
GRAPHS_DIR = DATA_DIR / "graphs"
MODELS_DIR = DATA_DIR / "models"
LIGHTNING_MODELS_DIR = MODELS_DIR / "lightning"
RL_MODELS_DIR = MODELS_DIR / "rl"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LIGHTNING_LOGS_DIR = DATA_DIR / "lightning_logs"
FINAL_TRAINING_LOGS_DIR = LIGHTNING_LOGS_DIR / "final_training"
HPARAM_SEARCH_LOGS_DIR = LIGHTNING_LOGS_DIR / "hparam_search"
OPTUNA_STUDIES_DIR = DATA_DIR / "optuna_studies"
HPARAM_SEARCH_STUDIES_DIR = OPTUNA_STUDIES_DIR / "hparam_search"
REPORTS_DIR = DATA_DIR / "reports"
PLOTS_DIR = DATA_DIR / "plots"
THESIS_PLOTS_DIR = PLOTS_DIR / "thesis"
CACHE_DIR = DATA_DIR / "cache"
WEB_DEMO_GRAPH_CACHE_DIR = CACHE_DIR / "web_demo_graphs"
WEB_DEMO_GRAPH_PREVIEW_CACHE_DIR = CACHE_DIR / "web_demo_graph_previews"
DOCS_DIR = DATA_DIR / "docs"
DOCKER_DIR = DATA_DIR / "docker"

# Frequently used files
GLOVE_300D_PATH = EMBEDDINGS_DIR / "glove.6B" / "glove.6B.300d.txt"
CAUSENET_GRAPH_PATH = GRAPHS_DIR / "causenet-precision.jsonl"
CAUSENET_FULL_GRAPH_PATH = GRAPHS_DIR / "causenet-full.jsonl"
CEG_GRAPH_PATH = GRAPHS_DIR / "Lexical_Cause_Effect_Graph.filtered.txt"
CEG_FULL_GRAPH_PATH = GRAPHS_DIR / "Lexical_Cause_Effect_Graph.txt"
DEFAULT_RL_MODEL_PATH = RL_MODELS_DIR / "msmarco_no_inverse_state_dict.pt"


class ActivationFunc(Enum):
    RELU = 1
    GELU = 2


class DistanceMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


BFS_CAPPED_BASELINE_MODEL = "BFS_Baseline"
BFS_UNCAPPED_BASELINE_MODEL = "BFS_Uncapped_Baseline"
RL_BASELINE_MODEL = "RL_Baseline"

BASELINE_MODEL_NAMES = frozenset(
    {
        BFS_CAPPED_BASELINE_MODEL,
        BFS_UNCAPPED_BASELINE_MODEL,
        RL_BASELINE_MODEL,
    }
)

MERGED_NODE_UNIVERSE = "merged_causenet_ceg"
CAUSENET_FULL_NODE_UNIVERSE = "causenet_full"
