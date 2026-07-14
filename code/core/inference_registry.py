from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_RL_MODEL_PATH, GLOVE_300D_PATH
from core.graph_config import SUPPORTED_INFERENCE_GRAPHS


@dataclass(frozen=True)
class RLPolicyConfig:
    id: str
    label: str
    description: str
    checkpoint_path: Path
    parameters: str
    embedding_dimension: int | None
    supported_graphs: tuple[str, ...]
    beam_width: int
    max_path_len: int
    max_actions: int
    max_visits: int
    glove_path: Path

    def runtime_config(self) -> dict[str, Any]:
        return {
            "rl_model_path": str(self.checkpoint_path),
            "rl_beam_width": self.beam_width,
            "rl_max_path_len": self.max_path_len,
            "rl_max_actions": self.max_actions,
            "rl_max_visits": self.max_visits,
        }

    def public_config(self) -> dict[str, Any]:
        return {
            "policy_config_id": self.id,
            "policy_label": self.label,
            "description": self.description,
            "checkpoint": str(self.checkpoint_path),
            "parameters": self.parameters,
            "embedding_dimension": self.embedding_dimension,
            "rl_beam_width": self.beam_width,
            "rl_max_path_len": self.max_path_len,
            "rl_max_actions": self.max_actions,
            "rl_max_visits": self.max_visits,
        }


DEFAULT_RL_POLICY_ID = "bluebaum_heindorf_lstm_msmarco_no_inverse"

RL_POLICY_CONFIGS: dict[str, RLPolicyConfig] = {
    DEFAULT_RL_POLICY_ID: RLPolicyConfig(
        id=DEFAULT_RL_POLICY_ID,
        label="RL",
        description="LSTM policy model from Bluebaum and Heindorf",
        checkpoint_path=DEFAULT_RL_MODEL_PATH,
        parameters="12M",
        embedding_dimension=None,
        supported_graphs=SUPPORTED_INFERENCE_GRAPHS,
        beam_width=50,
        max_path_len=2,
        max_actions=5000,
        max_visits=-1,
        glove_path=GLOVE_300D_PATH,
    )
}


def get_rl_policy_config(policy_config_id: str | None = None) -> RLPolicyConfig:
    policy_config_id = policy_config_id or DEFAULT_RL_POLICY_ID
    try:
        return RL_POLICY_CONFIGS[policy_config_id]
    except KeyError as exc:
        choices = ", ".join(sorted(RL_POLICY_CONFIGS))
        raise ValueError(
            f"Unknown RL policy config '{policy_config_id}'. Choices: {choices}"
        ) from exc


def graph_supports_rl(graph_id: str, policy_config_id: str | None = None) -> bool:
    policy = get_rl_policy_config(policy_config_id)
    return graph_id in policy.supported_graphs
