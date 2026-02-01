import torch
import torch.nn as nn

class StateEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize the StateEmbedding module.

        Args:
            input_dim (int): Dimension of the input state s_i.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output embedding h_i.
        """
        super(StateEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, s_i):
        """Forward pass to compute the state embedding.

        Args:
            s_i (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output embedding tensor h_i.
        """
        h_i = self.mlp(s_i)
        return h_i