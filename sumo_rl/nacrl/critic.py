import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, embedding_dim):
        """Initialize the Critic module.

        Args:
            embedding_dim (int): Dimension of the input embedding h_i_new.
        """
        super(Critic, self).__init__()
        self.value_function = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)  # Outputs a scalar value V_i
        )

    def forward(self, h_i_new):
        """Compute the value V_i for the given context-aware embedding.

        Args:
            h_i_new (torch.Tensor): Context-aware embedding (shape: [batch_size, embedding_dim]).

        Returns:
            torch.Tensor: Scalar value V_i (shape: [batch_size, 1]).
        """
        V_i = self.value_function(h_i_new)
        return V_i

# Loss function for the critic
def critic_loss(y_i, V_i):
    """Compute the loss for the critic.

    Args:
        y_i (torch.Tensor): Target value (shape: [batch_size, 1]).
        V_i (torch.Tensor): Predicted value from the critic (shape: [batch_size, 1]).

    Returns:
        torch.Tensor: Mean squared error loss.
    """
    return nn.MSELoss()(V_i, y_i)