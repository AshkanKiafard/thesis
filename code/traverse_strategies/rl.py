from dataclasses import dataclass
from typing import List, Tuple, Any
import torch
import torch.nn.functional as F
import networkx as nx

from rl_model import LSTMActorCriticAgent

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class BeamCandidate:
    path: List[str]
    prob: float
    lstm_state: Any


MODEL_PATH = "data/models/rl/msmarco_evaluation_state_dict.pt"
BEAM_WIDTH = 5
MAX_PATH_LEN = 1000


def rl_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: Any) -> Tuple[List[Any], int]:
    model = LSTMActorCriticAgent(
        input_dim=600,  # 300 (target) + 300 (current)
        output_dim=600,  # 300 (entity) + 300 (padding/relation)
        hidden_dim_mlp=2048,
        hidden_dim_lstm=1024
    )

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    initial_state = model.get_initial_state(DEVICE)
    candidates = [BeamCandidate(path=[start_node], prob=0.0, lstm_state=initial_state)]
    visited_count = 0

    for _ in range(MAX_PATH_LEN):
        next_candidates = []

        for cand in candidates:
            current_node = cand.path[-1]

            if current_node == end_node:
                return cand.path, visited_count

            curr_emb = torch.tensor(embeder.embed(current_node), device=DEVICE, dtype=torch.float32)
            target_emb = torch.tensor(embeder.embed(end_node), device=DEVICE, dtype=torch.float32)

            if curr_emb.shape[0] != 300:
                raise ValueError(f"RL Agent requires 300-dim Glove embeddings. Got {curr_emb.shape[0]}.")

            obs = torch.cat([target_emb, curr_emb], dim=0).view(1, 1, -1)

            neighbors = list(graph.successors(current_node))
            visited_count += 1

            if not neighbors:
                continue

            neighbor_embs = [torch.tensor(embeder.embed(n), device=DEVICE, dtype=torch.float32) for n in neighbors]

            padded_actions = []
            for n_emb in neighbor_embs:
                padded = torch.cat([n_emb, torch.zeros(300, device=DEVICE)], dim=0)
                padded_actions.append(padded)

            action_tensor = torch.stack(padded_actions).view(1, 1, len(neighbors), -1)

            with torch.no_grad():
                scores, _, new_state = model(obs, action_tensor, cand.lstm_state)

            valid_scores = scores.view(-1)
            log_probs = F.log_softmax(valid_scores, dim=0).cpu().numpy()

            for i, neighbor in enumerate(neighbors):
                new_prob = cand.prob + log_probs[i]
                next_candidates.append(BeamCandidate(
                    path=cand.path + [neighbor],
                    prob=new_prob,
                    lstm_state=new_state
                ))

        if not next_candidates:
            break

        candidates = sorted(next_candidates, key=lambda x: x.prob, reverse=True)[:BEAM_WIDTH]

        if candidates[0].path[-1] == end_node:
            return candidates[0].path, visited_count

    return candidates[0].path, visited_count
