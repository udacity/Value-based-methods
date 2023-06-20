import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units, fc2_units)
        self.fc2b = nn.Linear(fc2_units, 1)
        self.fc3  = nn.Linear(fc2_units, action_size)
        
        self.feature_stream = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )
        
        self.value_stream = nn.Sequential(
            self.fc2,
            nn.ReLU(),
            self.fc2b
        )

        self.advantage_stream = nn.Sequential(
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        features = self.feature_stream(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
