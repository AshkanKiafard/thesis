import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim_lstm):
        super().__init__()

        # Single-layer LSTM that processes the observation history of the agent.
        # batch_first=True means inputs have shape:
        # (batch_size, sequence_length, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim_lstm, batch_first=True)

    def get_initial_state(self, device, batch_size=1):
        # Initial hidden state (h_0) and cell state (c_0) for the LSTM.
        # Shape: (num_layers, batch_size, hidden_size)
        return (
            torch.zeros((1, batch_size, self.lstm.hidden_size), device=device),
            torch.zeros((1, batch_size, self.lstm.hidden_size), device=device)
        )

    def forward(self, observations, agent_state):
        # observations: sequence of input features
        # agent_state: tuple of (hidden_state, cell_state)
        return self.lstm(observations, agent_state)


class MLPReinforceAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim_mlp, output_dim):
        super().__init__()

        # Simple 2-layer MLP used as actor head.
        self.fc_1 = nn.Linear(input_dim, hidden_dim_mlp)
        self.fc_2 = nn.Linear(hidden_dim_mlp, output_dim)

    def forward(self, observations, actions):
        # Project observation into action embedding space.
        x = F.relu(self.fc_1(observations))
        x = self.fc_2(x)

        # Score each candidate action by dot product with the projected state.
        #
        # actions is expected to contain action embeddings, so this computes
        # compatibility between the current state representation and each action.
        scores = torch.matmul(actions, x.unsqueeze(-1))
        return scores.squeeze()


class LSTMActorCriticAgent(nn.Module):
    def __init__(self, input_dim=600, output_dim=600, hidden_dim_mlp=2048, hidden_dim_lstm=1024):
        super().__init__()

        # Shared recurrent backbone.
        self.lstm_layer = LSTMLayer(input_dim, hidden_dim_lstm)

        # Actor head:
        # produces action scores for the candidate actions.
        self.actor_head = MLPReinforceAgent(
            input_dim=hidden_dim_lstm,
            hidden_dim_mlp=hidden_dim_mlp,
            output_dim=output_dim
        )

        # Critic head:
        # estimates the value of the current state.
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim_lstm, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1)
        )

    def get_initial_state(self, device, batch_size=1):
        return self.lstm_layer.get_initial_state(device, batch_size)

    def forward(self, observations, actions, agent_state):
        # First pass the observation sequence through the LSTM.
        lstm_out, new_agent_state = self.lstm_layer(observations, agent_state)

        # Actor output: scores over the candidate actions.
        scores = self.actor_head(lstm_out, actions)

        # Critic output: scalar state-value estimate.
        values = self.critic_head(lstm_out)

        return scores, values, new_agent_state