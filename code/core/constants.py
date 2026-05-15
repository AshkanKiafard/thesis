from enum import Enum


class ActivationFunc(Enum):
    RELU = 1
    GELU = 2


class DistanceMetric(Enum):
    COSINE = 1
    EUCLIDEAN = 2


LIGHTNING_DIR = "data/models/lightning"
