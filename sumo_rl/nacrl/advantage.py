import torch

def compute_advantage(y_i, V_i):
    """Compute the advantage A_i for NACRL.

    Args:
        y_i (torch.Tensor): Target value (shape: [batch_size, 1]).
        V_i (torch.Tensor): Predicted value from the critic (shape: [batch_size, 1]).

    Returns:
        torch.Tensor: Advantage A_i (shape: [batch_size, 1]).
    """
    A_i = y_i - V_i
    return A_i