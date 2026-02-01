import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentAttention(nn.Module):
    def __init__(self, embed_dim):
        """Initialize the AgentAttention module.

        Args:
            embed_dim (int): Dimension of the input embeddings h_i,t and h_j,t.
        """
        super(AgentAttention, self).__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5  # Scaling factor for dot-product attention

    def forward(self, h_i, h_others):
        """Compute attention weights and context-aware embedding.

        Args:
            h_i (torch.Tensor): Embedding of the current agent (shape: [batch_size, embed_dim]).
            h_others (torch.Tensor): Embeddings of other agents (shape: [batch_size, num_agents-1, embed_dim]).

        Returns:
            torch.Tensor: Attention weights (alpha_ij) (shape: [batch_size, num_agents-1]).
            torch.Tensor: Context-aware embedding (h_i_new) (shape: [batch_size, embed_dim]).
        """
        # Compute attention scores using scaled dot-product
        query = h_i.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
        keys = h_others  # Shape: [batch_size, num_agents-1, embed_dim]
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # Shape: [batch_size, num_agents-1]
        scores = scores / self.scale

        # Compute attention weights
        alpha_i = F.softmax(scores, dim=-1)  # Shape: [batch_size, num_agents-1]

        # Compute context vector (weighted sum of h_others)
        context = torch.bmm(alpha_i.unsqueeze(1), h_others).squeeze(1)  # Shape: [batch_size, embed_dim]

        # Compute new context-aware embedding
        h_i_new = h_i + context  # Shape: [batch_size, embed_dim]

        return alpha_i, h_i_new