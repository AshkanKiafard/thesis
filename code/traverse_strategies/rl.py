from typing import Any
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
import zipfile

from traverse_strategies.rl_agent import LSTMActorCriticAgent


def load_embeddings(file_path="data/embeddings/glove.6B.zip"):
    embeddings_dict = {}
    with zipfile.ZipFile(file_path) as z:
        with z.open("glove.6B.300d.txt", 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip().split(' ')
                embeddings_dict[line[0]] = [float(value) for value in line[1:]]
    return embeddings_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize agent
agent = LSTMActorCriticAgent()
agent.load_state_dict(torch.load("data/models/msmarco_rl_state_dict.pt", map_location=device))
agent.to(device)
agent.eval()

# Load embeddings
embeddings = load_embeddings()
tensor_embeddings = {k: torch.tensor(v, dtype=torch.float32) for k, v in embeddings.items()}

# Add stop token embedding (random or ones)
STOP_TOKEN = "<STOP>"
tensor_embeddings[STOP_TOKEN] = torch.ones(300, dtype=torch.float32)


def get_embedding(parts) -> torch.Tensor:
    part_embeddings = [tensor_embeddings[p] for p in parts if p in tensor_embeddings]
    if len(part_embeddings) == 0:
        return torch.ones(300, dtype=torch.float32)
    return torch.stack(part_embeddings).mean(dim=0)


def rl_traverse(graph: nx.DiGraph, start_node: str, end_node: str) -> tuple[list[Any], int]:
    current = start_node
    path = [current]
    visited = 0
    steps = 0

    question_text = f"Can {start_node} cause {end_node}?"
    question_emb = get_embedding(question_text.split())
    rel_emb = get_embedding(["causes"])

    agent_state = agent.get_initial_state(device)
    max_actions = len(graph)  # can be dynamic; padding ensures fixed size if required

    done = False

    while not done and steps < 50:
        visited += 1
        steps += 1

        # Node + question embedding
        node_emb = tensor_embeddings.get(current, torch.zeros(300))
        state_ob = torch.cat([question_emb, node_emb], dim=-1)

        # Neighbor embeddings
        neighbors = list(graph.successors(current))
        neighbor_embs_list = [torch.cat([rel_emb, tensor_embeddings[STOP_TOKEN]], dim=-1)]

        for n in neighbors:
            neighbor_emb = tensor_embeddings.get(n, torch.zeros(300))
            neighbor_embs_list.append(torch.cat([rel_emb, neighbor_emb], dim=-1))

        neighbor_embs = torch.stack(neighbor_embs_list)

        # Pad to max_actions
        if neighbor_embs.shape[0] < max_actions:
            pad = torch.zeros(
                (max_actions - neighbor_embs.shape[0], neighbor_embs.shape[1]),
                dtype=torch.float32
            )
            neighbor_embs = torch.cat([neighbor_embs, pad], dim=0)

        with torch.no_grad():
            action_pred, _, agent_state = agent(
                state_ob.view(1, 1, -1).to(device),
                neighbor_embs.view(1, 1, *neighbor_embs.shape).to(device),
                agent_state
            )
            # Mask invalid actions
            action_pred.masked_fill_(action_pred == 0.0, float('-inf'))
            action_prob = torch.log(
                F.softmax(action_pred, dim=-1).clamp(min=np.finfo(np.float32).eps)
            ).cpu().numpy()

        # Pick best neighbor
        action = int(np.argmax(action_prob))
        if action == 0:  # stop-action
            break
        else:
            current = neighbors[action - 1]  # offset by 1 because stop-action is index 0
            path.append(current)

        if current == end_node:
            done = True

    return path, visited
