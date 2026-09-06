"""Quantile Regression DQN (QR-DQN, Dabney et al. 2017), item 3 of the four
candidate "significantly improve on what we have" ideas (fidings/
divergence_investigation.md sec 91, per direct user request). Learns a
distribution over returns per action (n_quantiles values) instead of a single
scalar Q-value -- structurally resists this project's own extensively-
characterized "confident lock-in" pathology (sec 32-34/51-57), since
collapsing to an overconfident POINT estimate is exactly what a
distributional value function has to avoid representing: it always keeps a
spread across quantiles, even for a confidently-preferred action. A
genuinely different mechanism from every fix tried in this project so far --
none of them changed what KIND of value function is being learned.

Design: reuses NeighborAttentionQNetwork's distributional=True head
(agents/networks.py) -- action_dim*n_quantiles outputs, reshaped to
(B, action_dim, n_quantiles). Ordinary forward() (used by act/act_batch/
q_values, all inherited unchanged from DQNAgent) already returns correct
(B, action_dim) MEAN Q-values via that network's _q_from_features, so this
class only needs to override optimize() -- action selection, exploration,
replay, and the training loop are identical to plain DQNAgent.

Scope limits (deliberate, matching this project's "one clean idea at a
time" convention): --q_entropy_weight and --cql_weight are designed for a
scalar Q head and not supported here (a distributional analogue of either
is a separate design question); FedProx (--fedprox_mu) IS supported since
it's a pure weight-space penalty, independent of the loss's value semantics.
"""
from typing import Optional

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from agents.dqn import DQNAgent, _collate, _mask_q


class QRDQNAgent(DQNAgent):
    def __init__(self, n_quantiles: int = 21, **kwargs):
        super().__init__(distributional=True, n_quantiles=n_quantiles, **kwargs)
        self.n_quantiles = n_quantiles
        # Standard QR-DQN quantile midpoints: tau_i = (i + 0.5) / N, i=0..N-1.
        self.quantile_tau = ((torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles).to(self.device)

    def optimize(self) -> Optional[float]:
        if len(self.replay) < max(4, self.batch_size):
            return None
        self.q.train()

        obs, actions, rewards, next_obs, dones, ns = self.replay.sample(self.batch_size)

        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, self.device)
        n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, self.device)

        B = len(actions)
        batch_idx = torch.arange(B, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        # gamma**n per-sample bootstrap discount -- n=1 for every sample (the
        # n_step<=1 default) makes this exactly self.gamma, same convention
        # as DQNAgent.optimize().
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device)
        discount_t = torch.full_like(ns_t, self.gamma).pow(ns_t)

        quantiles = self.q.forward_quantiles(own, neighbors, neighbor_mask, hop_dist)  # (B, A, N)
        taken_quantiles = quantiles[batch_idx, actions_t]  # (B, N)

        with torch.no_grad():
            # Double DQN, same convention as DQNAgent.optimize(): select the
            # next action with the online net's MEAN Q-values (masked to
            # valid actions), then read that action's quantiles off the
            # TARGET net for the distributional Bellman backup.
            next_quantiles_online = self.q.forward_quantiles(n_own, n_neighbors, n_neighbor_mask, n_hop_dist)
            next_q_online = _mask_q(next_quantiles_online.mean(dim=-1), n_action_mask)
            next_actions = next_q_online.argmax(dim=1)
            next_quantiles_target = self.q_target.forward_quantiles(n_own, n_neighbors, n_neighbor_mask, n_hop_dist)
            next_taken_quantiles = next_quantiles_target[batch_idx, next_actions]  # (B, N)
            target_quantiles = (
                rewards_t.unsqueeze(1)
                + (1.0 - dones_t.unsqueeze(1)) * discount_t.unsqueeze(1) * next_taken_quantiles
            )  # (B, N)

        # Quantile regression (Huber, kappa=1.0) loss, pairwise over predicted
        # quantile i and target quantile j -- the standard QR-DQN loss
        # (Dabney et al. 2017, eq. 10).
        td_errors = target_quantiles.unsqueeze(1) - taken_quantiles.unsqueeze(2)  # (B, N_pred, N_target)
        huber = torch.where(
            td_errors.abs() <= 1.0, 0.5 * td_errors.pow(2), td_errors.abs() - 0.5
        )
        tau = self.quantile_tau.view(1, -1, 1)
        quantile_loss = (
            torch.abs(tau - (td_errors.detach() < 0).float()) * huber
        ).mean(dim=2).sum(dim=1).mean()

        loss = quantile_loss

        if self.mu > 0 and self._global_params is not None:
            diffs = torch._foreach_sub(list(self.q.parameters()), self._global_params)
            norms = torch._foreach_norm(diffs, 2)
            prox_term = sum(n.pow(2) for n in norms)
            loss = loss + (self.mu / 2.0) * prox_term

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.tau > 0:
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        elif self.learn_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            self.q_target.eval()

        return float(loss.item())
