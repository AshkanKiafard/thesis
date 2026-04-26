from typing import List, Tuple, Any, Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import networkx as nx

from core.rl_model import LSTMActorCriticAgent

# Use GPU if available, otherwise run on CPU.
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class BeamCandidate:
    # One candidate path in the beam search.
    path: List[str]

    # Accumulated log-probability of the path.
    prob: float

    # LSTM hidden state associated with this candidate.
    lstm_state: Any

    # Nodes already visited by this candidate to avoid cycles.
    visited: set


def rl_traverse(
    graph: nx.DiGraph,
    start_node: str,
    end_node: str,
    embeder: Any,
    config: Dict[str, Any] = None
) -> Tuple[List[Any], int]:
    # Optional runtime config for model path and search constraints.
    if config is None:
        config = {}

    model_path = config.get('rl_model_path', "data/models/rl/msmarco_evaluation_state_dict.pt")
    beam_width = config.get('rl_beam_width', 5)
    max_path_len = config.get('rl_max_path_len', -1)
    max_visits = config.get('rl_max_visits', -1)

    # Recreate the RL policy/value model and load pretrained weights.
    model = LSTMActorCriticAgent(
        input_dim=600,
        output_dim=600,
        hidden_dim_mlp=2048,
        hidden_dim_lstm=1024
    )

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Start beam with a single candidate containing only the start node.
    initial_state = model.get_initial_state(DEVICE)
    candidates = [
        BeamCandidate(
            path=[start_node],
            prob=0.0,
            lstm_state=initial_state,
            visited={start_node}
        )
    ]

    visited_count = 0
    step_count = 0

    while candidates:
        # Optional limit on path length / number of decision steps.
        if max_path_len != -1 and step_count >= max_path_len:
            break

        step_count += 1
        next_candidates = []

        for cand in candidates:
            current_node = cand.path[-1]

            # If any beam candidate already reached the target, return immediately.
            if current_node == end_node:
                return cand.path, visited_count

            curr_vec = embeder.embed(current_node)
            target_vec = embeder.embed(end_node)

            # This RL model was trained on 300-dim GloVe embeddings.
            if len(curr_vec) != 300:
                raise ValueError(f"RL Agent requires 300-dim Glove embeddings. Got {len(curr_vec)}.")

            curr_emb = torch.tensor(curr_vec, device=DEVICE, dtype=torch.float32)
            target_emb = torch.tensor(target_vec, device=DEVICE, dtype=torch.float32)

            # Observation = concatenation of target embedding and current node embedding.
            # Shape after view: (batch=1, seq_len=1, feature_dim=600)
            obs = torch.cat([target_emb, curr_emb], dim=0).view(1, 1, -1)

            neighbors = list(graph.successors(current_node))
            visited_count += 1

            # Optional safety cap to keep RL traversal bounded.
            if max_visits != -1 and visited_count > max_visits:
                return [], visited_count

            # Prevent revisiting nodes that are already on this candidate path.
            valid_neighbors = [n for n in neighbors if n not in cand.visited]

            if not valid_neighbors:
                continue

            neighbor_embs = [
                torch.tensor(embeder.embed(n), device=DEVICE, dtype=torch.float32)
                for n in valid_neighbors
            ]

            # The model expects 600-dim action vectors.
            # Here relation part is padded with zeros and neighbor embedding fills the second half.
            zero_relation = torch.zeros(300, device=DEVICE)
            padded_actions = [torch.cat([zero_relation, n_emb], dim=0) for n_emb in neighbor_embs]

            # Shape: (batch=1, seq_len=1, num_actions, action_dim=600)
            action_tensor = torch.stack(padded_actions).view(1, 1, len(valid_neighbors), -1)

            with torch.no_grad():
                scores, _, new_state = model(obs, action_tensor, cand.lstm_state)

            # Convert scores into log-probabilities for beam search expansion.
            valid_scores = scores.view(-1)
            log_probs = F.log_softmax(valid_scores, dim=0).cpu().numpy()

            for i, neighbor in enumerate(valid_neighbors):
                new_prob = cand.prob + log_probs[i]
                new_visited = cand.visited.copy()
                new_visited.add(neighbor)

                next_candidates.append(
                    BeamCandidate(
                        path=cand.path + [neighbor],
                        prob=new_prob,
                        lstm_state=new_state,
                        visited=new_visited
                    )
                )

        if not next_candidates:
            break

        # Keep only the top-k most likely candidates.
        candidates = sorted(next_candidates, key=lambda x: x.prob, reverse=True)[:beam_width]

        # Small shortcut: if the best candidate already ends at the target, return it.
        if candidates[0].path[-1] == end_node:
            return candidates[0].path, visited_count

    # No path found within the beam / limits.
    return [], visited_count