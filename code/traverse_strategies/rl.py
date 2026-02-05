from typing import List, Tuple, Any, Dict
import torch
import torch.nn.functional as F
import networkx as nx
from dataclasses import dataclass

from rl_model import LSTMActorCriticAgent

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class BeamCandidate:
    path: List[str]
    prob: float
    lstm_state: Any
    visited: set


def rl_traverse(graph: nx.DiGraph, start_node: str, end_node: str, embeder: Any, config: Dict[str, Any] = None) -> \
Tuple[List[Any], int]:
    if config is None: config = {}

    model_path = config.get('rl_model_path', "data/models/rl/msmarco_evaluation_state_dict.pt")
    beam_width = config.get('rl_beam_width', 5)
    max_path_len = config.get('rl_max_path_len', -1)
    max_visits = config.get('rl_max_visits', -1)

    model = LSTMActorCriticAgent(input_dim=600, output_dim=600, hidden_dim_mlp=2048, hidden_dim_lstm=1024)

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    initial_state = model.get_initial_state(DEVICE)
    candidates = [BeamCandidate(path=[start_node], prob=0.0, lstm_state=initial_state, visited={start_node})]
    visited_count = 0

    step_count = 0

    while candidates:
        if max_path_len != -1 and step_count >= max_path_len:
            break

        step_count += 1
        next_candidates = []

        for cand in candidates:
            current_node = cand.path[-1]

            if current_node == end_node:
                return cand.path, visited_count

            curr_vec = embeder.embed(current_node)
            target_vec = embeder.embed(end_node)

            if len(curr_vec) != 300:
                raise ValueError(f"RL Agent requires 300-dim Glove embeddings. Got {len(curr_vec)}.")

            curr_emb = torch.tensor(curr_vec, device=DEVICE, dtype=torch.float32)
            target_emb = torch.tensor(target_vec, device=DEVICE, dtype=torch.float32)
            obs = torch.cat([target_emb, curr_emb], dim=0).view(1, 1, -1)

            neighbors = list(graph.successors(current_node))
            visited_count += 1
            if max_visits != -1 and visited_count > max_visits:
                return [], visited_count

            valid_neighbors = [n for n in neighbors if n not in cand.visited]

            if not valid_neighbors:
                continue

            neighbor_embs = [torch.tensor(embeder.embed(n), device=DEVICE, dtype=torch.float32) for n in
                             valid_neighbors]

            zero_relation = torch.zeros(300, device=DEVICE)
            padded_actions = [torch.cat([zero_relation, n_emb], dim=0) for n_emb in neighbor_embs]
            action_tensor = torch.stack(padded_actions).view(1, 1, len(valid_neighbors), -1)

            with torch.no_grad():
                scores, _, new_state = model(obs, action_tensor, cand.lstm_state)

            valid_scores = scores.view(-1)
            log_probs = F.log_softmax(valid_scores, dim=0).cpu().numpy()

            for i, neighbor in enumerate(valid_neighbors):
                new_prob = cand.prob + log_probs[i]
                new_visited = cand.visited.copy()
                new_visited.add(neighbor)

                next_candidates.append(BeamCandidate(
                    path=cand.path + [neighbor],
                    prob=new_prob,
                    lstm_state=new_state,
                    visited=new_visited
                ))

        if not next_candidates:
            break

        candidates = sorted(next_candidates, key=lambda x: x.prob, reverse=True)[:beam_width]

        if candidates[0].path[-1] == end_node:
            return candidates[0].path, visited_count

    return [], visited_count
