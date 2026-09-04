"""Neural network architectures for intersection agents.

Foundation-model design
------------------------
A single shared architecture must work for ANY intersection topology
(3/4/5-way, protected lefts, pedestrian phases, ...) and for ANY amount of
available neighbor information -- 0 neighbors up to K_MAX neighbors, any
subset of which may be dropped out to simulate a communication failure.

Observation contract (per intersection, per tick)
--------------------------------------------------
    own_obs        (D_own,)          fixed-size own-intersection features
    neighbor_obs   (K_MAX, D_nbr)    fixed-size per-neighbor features,
                                     zero-padded for missing/dropped slots
    neighbor_mask  (K_MAX,)          1.0 = valid neighbor this tick,
                                     0.0 = padded OR comm-dropped
    hop_dist       (K_MAX,)          integer hop distance (1..K) of each
                                     neighbor slot; 0 for padding slots
    action_mask    (A_MAX,)          1.0 = this Q-slot is a real action for
                                     THIS intersection, 0.0 = doesn't exist
                                     (replaces manual phase_mapping)

The network never sees which city or topology an observation came from --
that's the whole point. Everything topology-specific is expressed purely
through the masks.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Kept for backward compatibility / simple non-federated baselines."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeighborAttentionQNetwork(nn.Module):
    """Permutation-invariant, topology-agnostic Q-network.

    Own observation is the attention query; neighbor observations (up to
    K_MAX hops away, zero-padded, individually maskable) are the keys/
    values. Because masked attention naturally handles a variable number
    of *valid* neighbors, the exact same weights work for:

      - an isolated intersection (mask is all zero)
      - an intersection with 1 live neighbor out of K_MAX slots
      - an intersection with a full K_MAX-neighbor, multi-hop neighborhood

    No per-city or per-topology code is needed anywhere in this class.
    """

    def __init__(
        self,
        own_dim: int,
        neighbor_dim: int,
        action_dim: int,
        k_max: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_hops: int = 4,
        head_fix: bool = True,
        dueling: bool = False,
        actor_critic: bool = False,
    ):
        super().__init__()
        if dueling and actor_critic:
            raise ValueError("dueling and actor_critic are mutually exclusive head types.")
        self.k_max = k_max
        self.d_model = d_model
        self.n_hops = n_hops
        self.head_fix = head_fix
        self.dueling = dueling
        self.actor_critic = actor_critic

        self.own_encoder = nn.Sequential(
            nn.Linear(own_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # +1 so "padding" (hop 0) gets its own embedding, distinct from a
        # real hop-1 neighbor.
        self.hop_embedding = nn.Embedding(n_hops + 1, d_model)

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        try:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, batch_first=True
            )
            self._attn_batch_first = True
        except TypeError:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads
            )
            self._attn_batch_first = False
        self.attn_norm = nn.LayerNorm(d_model)

        # Learnable fallback so a fully isolated intersection (mask all
        # zero) still attends to a well-defined value instead of a
        # degenerate all-masked softmax.
        self.no_neighbor_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Trunk shared by both the plain and dueling heads. Kept as a
        # 4-element Sequential (Linear, ReLU, Linear, ReLU) so the plain
        # (non-dueling) path can still append a single final Linear at
        # index 4 and keep the "head.4.weight"/"head.4.bias" key names
        # `federated/aggregation.py::masked_head_weighted_average` already
        # looks for by default -- no aggregation-side change needed unless
        # dueling is actually turned on.
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        if self.dueling:
            # V(s): one scalar per intersection, no action_mask involved --
            # every client updates every element of this every step, so
            # (unlike the fully action-indexed plain head) it aggregates
            # cleanly with an ordinary weighted average across cities of
            # any action_dim. A(s,a): the actual action-indexed stream,
            # still exactly action_dim wide -- masked-head aggregation
            # (see federated/aggregation.py) still applies to this one.
            self.value_head = nn.Linear(d_model, 1)
            self.advantage_head = nn.Linear(d_model, action_dim)
        elif self.actor_critic:
            # Same shared trunk as the dueling head, split into a policy
            # (action logits, masked+softmaxed by the caller -- this class
            # never applies action_mask itself, matching the plain/dueling
            # Q-head convention) and a state-value scalar. Kept as two
            # separate Linears (not fused) so PPOAgent can read raw logits
            # and value independently without slicing one tensor apart.
            self.policy_head = nn.Linear(d_model, action_dim)
            self.ac_value_head = nn.Linear(d_model, 1)
        else:
            self.head.append(nn.Linear(d_model, action_dim))

        if not self.head_fix:
            self.pool_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

    def _q_from_features(self, combined: torch.Tensor) -> torch.Tensor:
        """Shared trunk -> Q-values, either straight through the plain head
        or combined dueling-style (Q = V + A - mean(A)) if ``dueling``."""
        feat = self.head(combined)
        if not self.dueling:
            return feat
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def _combined_features(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Own-obs + attended (or mean-pooled, if not head_fix) neighbor
        summary, concatenated -- everything upstream of ``self.head``.
        Factored out of ``forward`` so ``forward_actor_critic`` can reuse
        the exact same trunk instead of duplicating this logic."""
        B, K, _ = neighbor_obs.shape

        own_emb = self.own_encoder(own_obs)  # (B, d_model)
        nbr_emb = self.neighbor_encoder(neighbor_obs)  # (B, K, d_model)

        if not self.head_fix:
            valid = neighbor_mask.clamp(min=0.0, max=1.0)
            nbr_pool = torch.sum(nbr_emb * valid.unsqueeze(-1), dim=1)
            nbr_count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            nbr_pool = nbr_pool / nbr_count
            nbr_pool = self.pool_head(nbr_pool)
            return torch.cat([own_emb, nbr_pool], dim=-1)

        if hop_dist is not None:
            hop_dist = hop_dist.clamp(0, self.n_hops)
            nbr_emb = nbr_emb + self.hop_embedding(hop_dist.long())

        fallback = self.no_neighbor_token.expand(B, 1, -1)
        kv = torch.cat([nbr_emb, fallback], dim=1)  # (B, K+1, d_model)
        fallback_mask = torch.ones(B, 1, device=neighbor_mask.device)
        full_mask = torch.cat([neighbor_mask, fallback_mask], dim=1)  # (B, K+1)

        key_padding_mask = full_mask < 0.5  # True = ignore this position

        query = own_emb.unsqueeze(1)  # (B, 1, d_model)

        if self._attn_batch_first:
            attn_out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
            attn_out = attn_out.squeeze(1)
        else:
            # Older torch versions only support (seq, batch, embed).
            query_t = query.transpose(0, 1)
            kv_t = kv.transpose(0, 1)
            attn_out, _ = self.attn(
                query_t,
                kv_t,
                kv_t,
                key_padding_mask=key_padding_mask,
            )
            attn_out = attn_out.transpose(0, 1).squeeze(1)

        attn_out = self.attn_norm(attn_out + own_emb)  # residual

        return torch.cat([own_emb, attn_out], dim=-1)

    def forward(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DQN entry point: own+neighbor obs -> masked-ready Q-values
        (plain or dueling-combined depending on ``self.dueling``)."""
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        return self._q_from_features(combined)

    def forward_actor_critic(
        self,
        own_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hop_dist: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PPO entry point: own+neighbor obs -> (action logits, state
        value). Only valid when ``self.actor_critic=True``. Logits are
        raw (not masked/softmaxed) -- the caller applies action_mask,
        matching the convention ``forward``'s Q-values follow with
        ``_mask_q`` in agents/dqn.py."""
        if not self.actor_critic:
            raise RuntimeError("forward_actor_critic() called on a non-actor-critic network.")
        combined = self._combined_features(own_obs, neighbor_obs, neighbor_mask, hop_dist)
        feat = self.head(combined)
        logits = self.policy_head(feat)
        value = self.ac_value_head(feat).squeeze(-1)
        return logits, value
