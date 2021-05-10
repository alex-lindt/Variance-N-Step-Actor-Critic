import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):

    def __init__(self, n_state_features, n_actions, n_hidden, device):
        super(Actor, self).__init__()
        self.device = device

        self.n_state_features = n_state_features
        self.n_actions = n_actions

        self.n_h1 = n_hidden[0]
        self.n_h2 = n_hidden[1]

        self.actor_network = torch.nn.Sequential(
            nn.Linear(n_state_features, self.n_h1),
            nn.ReLU(),
            nn.Linear(self.n_h1, self.n_h2),
            nn.ReLU(),
            nn.Linear(self.n_h2, n_actions),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        if isinstance(state, np.int64) or isinstance(state, int):
            # Convert int into onehot vector
            state = torch.nn.functional.one_hot(torch.tensor(state), self.n_state_features)
            state = state.type(torch.FloatTensor)
        else:
            state = torch.FloatTensor(state).to(self.device)
        policy = self.actor_network(state)
        return Categorical(policy)


class ValueFunction(nn.Module):
    def __init__(self, n_state_features, n_hidden, device):
        super(ValueFunction, self).__init__()
        self.device = device

        self.n_state_features = n_state_features

        self.n_h1 = n_hidden[0]
        self.n_h2 = n_hidden[1]

        self.critic_network = torch.nn.Sequential(
            nn.Linear(n_state_features, self.n_h1),
            nn.ReLU(),
            nn.Linear(self.n_h1, self.n_h2),
            nn.ReLU(),
            nn.Linear(self.n_h2, 1)
        )

    def forward(self, state):
        if isinstance(state, np.int64) or isinstance(state, int):
            # Convert int into onehot vector
            state = torch.nn.functional.one_hot(torch.tensor(state), self.n_state_features)
            state = state.type(torch.FloatTensor).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        state_value = self.critic_network(state)
        return state_value


class QValueFunction(nn.Module):
    def __init__(self, n_state_features, n_actions, n_hidden, device):
        super(QValueFunction, self).__init__()
        self.device = device

        self.n_state_features = n_state_features
        self.n_actions = n_actions

        self.n_h1 = n_hidden[0]
        self.n_h2 = n_hidden[1]

        self.critic_network = torch.nn.Sequential(
            nn.Linear(n_state_features, self.n_h1),
            nn.ReLU(),
            nn.Linear(self.n_h1, self.n_h2),
            nn.ReLU(),
            nn.Linear(self.n_h2, n_actions)
        )

    def forward(self, state):
        if isinstance(state, np.int64) or isinstance(state, int):
            # Convert int into onehot vector
            state = torch.nn.functional.one_hot(torch.tensor(state), self.n_state_features)
            state = state.type(torch.FloatTensor).to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        state_value = self.critic_network(state)
        return state_value
