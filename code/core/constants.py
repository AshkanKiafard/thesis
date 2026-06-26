from enum import Enum


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

MERGED_NODE_UNIVERSE = "merged_causenet_causalbank"
CAUSENET_FULL_NODE_UNIVERSE = "causenet_full"
