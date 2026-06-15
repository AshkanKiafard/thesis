from enum import Enum


class ActivationFunc(Enum):
    RELU = 1
    GELU = 2


class DistanceMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


LIGHTNING_DIR = "data/models/lightning"
GLOVE_300D_PATH = "data/embeddings/glove.6B/glove.6B.300d.txt"
EMBEDDING_INDEX_MIN_SUCCESSORS = 16
