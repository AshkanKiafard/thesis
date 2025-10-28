import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int = 600, hidden_dim_lstm: int = 1024):
        super().__init__()
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm = nn.LSTM(input_dim, hidden_dim_lstm, batch_first=True)

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        h = torch.zeros((1, batch_size, self.hidden_dim_lstm), dtype=torch.float32, device=device)
        c = torch.zeros_like(h)
        return h, c

    def forward(self, observations, agent_state):
        output, agent_state = self.lstm(observations, agent_state)
        return output, agent_state


class MLPReinforceAgent(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim_mlp: int = 2048, output_dim: int = 600):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim_mlp)
        self.fc_2 = nn.Linear(hidden_dim_mlp, output_dim)

    def weight_init(self):
        self.apply(init_weights)

    @staticmethod
    def get_initial_state():
        return ()

    def forward(self, observations, actions, _):
        obs_proj = F.relu(self.fc_1(observations))
        obs_proj = self.fc_2(obs_proj)
        scores = torch.matmul(actions, obs_proj.unsqueeze(-1)).squeeze(-1)
        return scores, None, _


class LSTMReinforceAgent(nn.Module):
    def __init__(self, input_dim: int = 600,
                 hidden_dim_lstm: int = 1024,
                 hidden_dim_mlp: int = 2048,
                 output_dim: int = 600):
        super().__init__()
        self.lstm_layer = LSTMLayer(input_dim=input_dim, hidden_dim_lstm=hidden_dim_lstm)
        self.head = MLPReinforceAgent(input_dim=hidden_dim_lstm,
                                      hidden_dim_mlp=hidden_dim_mlp,
                                      output_dim=output_dim)

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        return self.lstm_layer.get_initial_state(device, batch_size)

    def forward(self, observations, actions, agent_state):
        lstm_out, agent_state = self.lstm_layer(observations, agent_state)
        policy, _, _ = self.head(lstm_out, actions, agent_state)
        return policy, None, agent_state


class LSTMActorCriticAgent(nn.Module):
    def __init__(self, input_dim: int = 600,
                 hidden_dim_lstm: int = 1024,
                 hidden_dim_mlp: int = 2048,
                 output_dim: int = 600):
        super().__init__()
        self.lstm_layer = LSTMLayer(input_dim=input_dim, hidden_dim_lstm=hidden_dim_lstm)
        self.actor_head = MLPReinforceAgent(input_dim=hidden_dim_lstm,
                                            hidden_dim_mlp=hidden_dim_mlp,
                                            output_dim=output_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim_lstm, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1)
        )

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        return self.lstm_layer.get_initial_state(device, batch_size)

    def forward(self, observations, actions, agent_state):
        lstm_out, agent_state = self.lstm_layer(observations, agent_state)
        policy, _, _ = self.actor_head(lstm_out, actions, agent_state)
        values = self.critic_head(lstm_out).squeeze(-1)
        return policy, values, agent_state