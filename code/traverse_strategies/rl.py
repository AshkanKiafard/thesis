from dataclasses import dataclass
from itertools import islice
from typing import Iterable, List, Tuple, Any, Dict, Optional

import torch
import torch.nn.functional as F

from core.rl_model import LSTMActorCriticAgent

# Use GPU if available, otherwise run on CPU.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Avoid reloading the RL checkpoint for every evaluated question.
_RL_MODEL_CACHE = {}


@dataclass
class BeamCandidate:
    # One candidate path in the beam search.
    path: List[str]

    # Accumulated log-probability of the path.
    prob: float

    # LSTM hidden state associated with this candidate.
    lstm_state: Any

    # Index of the previous beam item.
    # Kept for compatibility with the original beam-search structure.
    prev_id: Optional[int] = 0


def _load_rl_model(model_path: str) -> LSTMActorCriticAgent:
    if model_path in _RL_MODEL_CACHE:
        return _RL_MODEL_CACHE[model_path]

    # Recreate the RL policy/value model and load pretrained weights.
    model = LSTMActorCriticAgent(
        input_dim=600,
        output_dim=600,
        hidden_dim_mlp=2048,
        hidden_dim_lstm=1024,
    )

    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        state_dict = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _RL_MODEL_CACHE[model_path] = model
    return model


def _edge_sentence(graph: Any, current_node: str, neighbor: str) -> str:
    data = graph.get_edge_data(current_node, neighbor, default={})
    sentence = data.get("sentence", "")

    if sentence:
        return sentence

    # Fallback if no sentence was stored.
    return f"{current_node} can cause {neighbor}"


def _to_device_tensor(embedding: Any) -> torch.Tensor:
    if isinstance(embedding, torch.Tensor):
        return embedding.to(device=DEVICE, dtype=torch.float32)

    return torch.as_tensor(embedding, device=DEVICE, dtype=torch.float32)


def _build_observation(question_text: str, current_node: str, embeder: Any) -> torch.Tensor:
    # Original EnvironmentTorch:
    # current_question_emb = question.embedding
    # current_node_emb = graph.entity_embeddings[node]
    # observation = [question_embedding, current_node_embedding]
    question_emb = _to_device_tensor(embeder.embed_question(question_text))
    current_emb = _to_device_tensor(embeder.embed_entity(current_node))

    if question_emb.numel() != 300 or current_emb.numel() != 300:
        raise ValueError(
            f"RL Agent requires 300-dim GloVe embeddings. "
            f"Got question={question_emb.numel()}, entity={current_emb.numel()}."
        )

    return torch.cat([question_emb, current_emb], dim=0).view(1, 1, -1)


def _build_action_tensor(
    graph: Any,
    current_node: str,
    neighbors: Iterable[str],
    embeder: Any,
    max_actions: int,
) -> Tuple[torch.Tensor, List[str]]:
    # Original environment always inserts the stop action at index 0.
    #
    # In the original code, stop_action is a real artificial entity:
    #     "stop stop action"
    #
    # During decoding, if the selected entity is stop_action, the path appends
    # the current node again. So here we score stop using the stop-action
    # embedding, but map it back to current_node as the next path label.
    action_vectors = []
    path_labels = []

    stop_relation_emb = _to_device_tensor(embeder.embed_relation("stop"))
    stop_entity_emb = _to_device_tensor(embeder.embed_entity("stop stop action"))

    action_vectors.append(torch.cat([stop_relation_emb, stop_entity_emb], dim=0))
    path_labels.append(current_node)

    # Original max_actions behavior: neighbors are truncated after the stop action
    # exists in the adjacency list. Here we reserve one slot for stop.
    for neighbor in islice(neighbors, max_actions - 1):
        relation_text = _edge_sentence(graph, current_node, neighbor)

        relation_emb = _to_device_tensor(embeder.embed_relation(relation_text))
        neighbor_emb = _to_device_tensor(embeder.embed_entity(neighbor))

        action_vectors.append(torch.cat([relation_emb, neighbor_emb], dim=0))
        path_labels.append(neighbor)

    real_action_count = len(action_vectors)

    if real_action_count == 0:
        raise ValueError("Action tensor cannot be empty.")

    action_tensor = torch.stack(action_vectors)

    # Pad to max_actions like the original EnvironmentTorch.
    if real_action_count < max_actions:
        padding = torch.zeros(
            (max_actions - real_action_count, action_tensor.shape[1]),
            device=DEVICE,
            dtype=torch.float32,
        )
        action_tensor = torch.cat([action_tensor, padding], dim=0)

    # Shape: (batch=1, seq_len=1, max_actions, action_dim=600)
    return action_tensor.view(1, 1, max_actions, -1), path_labels


def _run_agent(agent, obs, action_tensor, state):
    with torch.no_grad():
        action_pred, _, agent_state = agent(obs, action_tensor, state)

    # Original code masks zero scores, mainly to remove padded actions.
    action_pred = action_pred.clone()
    action_pred.masked_fill_(action_pred == 0.0, float("-inf"))

    log_probs = torch.log(
        F.softmax(action_pred, dim=-1).clamp(min=torch.finfo(torch.float32).eps)
    )

    return log_probs.view(-1).detach().cpu().numpy(), agent_state


def rl_traverse(
    graph: Any,
    start_node: str,
    end_node: str,
    embeder: Any,
    config: Dict[str, Any] = None,
) -> Tuple[List[Any], int]:
    # Optional runtime config for model path and search constraints.
    if config is None:
        config = {}

    model_path = config.get(
        "rl_model_path",
        "data/models/rl/msmarco_no_inverse_state_dict.pt",
    )

    # Original evaluation defaults.
    beam_width = config.get("rl_beam_width", 50)
    max_path_len = config.get("rl_max_path_len", 2)
    max_actions = config.get("rl_max_actions", 5000)

    # Keep this optional, but do not use it for reproducing the original RL baseline.
    max_visits = config.get("rl_max_visits", -1)

    question_text = config.get("question")
    if question_text is None:
        # Fallback, but this is not exactly the original MS MARCO question.
        question_text = f"can {start_node} cause {end_node}?"

    agent = _load_rl_model(model_path)

    # Start beam with a single candidate containing only the start node.
    paths = [
        BeamCandidate(
            path=[start_node],
            prob=0.0,
            lstm_state=agent.get_initial_state(DEVICE),
            prev_id=0,
        )
    ]

    # Original node-count semantics:
    # unique nodes in retained beam paths until the effect is first seen.
    nodes = {start_node}
    found = False

    for _ in range(max_path_len):
        candidates = []

        for beam_idx, p in enumerate(paths):
            current_node = p.path[-1]

            obs = _build_observation(question_text, current_node, embeder)

            neighbors = graph.successors(current_node)

            action_tensor, action_path_labels = _build_action_tensor(
                graph=graph,
                current_node=current_node,
                neighbors=neighbors,
                embeder=embeder,
                max_actions=max_actions,
            )

            log_probs, agent_state = _run_agent(
                agent=agent,
                obs=obs,
                action_tensor=action_tensor,
                state=p.lstm_state,
            )

            # Only keep real actions: stop + actual graph neighbors.
            log_probs = log_probs[: len(action_path_labels)]

            for action_idx, log_prob in enumerate(log_probs):
                entity_label = action_path_labels[action_idx]

                candidates.append(
                    BeamCandidate(
                        path=p.path + [entity_label],
                        prob=p.prob + float(log_prob),
                        lstm_state=agent_state,
                        prev_id=beam_idx,
                    )
                )

        if not candidates:
            break

        candidates = sorted(candidates, key=lambda x: x.prob, reverse=True)
        paths = candidates[:beam_width]

        # Match original beam_search node counting.
        for p in paths:
            if not found:
                nodes.update(p.path)

            if end_node in nodes:
                found = True

        if max_visits != -1 and len(nodes) > max_visits:
            return [], len(nodes)

    # Original prediction is positive if the effect occurs anywhere in any
    # retained candidate path.
    for p in paths:
        if end_node in p.path:
            return p.path, len(nodes)

    return [], len(nodes)
