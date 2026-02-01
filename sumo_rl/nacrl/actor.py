import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """Initialize the Actor module.

        Args:
            state_dim (int): Dimension of the input state s_i.
            action_dim (int): Dimension of the output action distribution.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s_i):
        """Compute the action distribution for the given state.

        Args:
            s_i (torch.Tensor): Input state tensor (shape: [batch_size, state_dim]).

        Returns:
            torch.distributions.Categorical: Action distribution.
        """
        logits = self.network(s_i)  # Compute logits for the action distribution
        action_dist = Categorical(logits=logits)  # Create a categorical distribution
        return action_dist
