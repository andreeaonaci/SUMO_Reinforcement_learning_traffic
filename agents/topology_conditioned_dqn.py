"""TC-FedAvg: Topology-Conditioned FedAvg (fidings/divergence_investigation.md,
"Open questions" queue, added after item 23 per direct user request for a
bespoke design rather than an existing named federated-learning method).

Motivation, from this project's own evidence: every AGGREGATION-strategy
tweak tried here (EMA-loss/-alignment weighting, clustered-by-action-dim,
gradient-survival, velocity-novelty) came back null, and federation vs.
no-federation makes no measurable difference either (sec 49/50/64). That
combination says the problem was never in HOW weights get combined across
cities -- it's that the ONE shared function being averaged has no way to
behave differently for a 3-way vs. a 5-way intersection in the first place,
other than the raw masked inputs. Only the final Q-head (via action_mask)
gets any explicit topology-specific treatment today.

Design: a small shared hypernetwork (`NeighborAttentionQNetwork.topo_hyper`,
see that class for the actual FiLM math) maps a 4-dim structural descriptor
-- valid-action fraction, valid-neighbor fraction, mean/max hop distance of
live neighbors, ALL computable for ANY intersection including one never
trained on -- to a scale/shift applied to the fused own+neighbor
representation. FedAvg aggregation itself is completely UNCHANGED: the
hypernetwork's weights are shared and averaged exactly like every other
layer. This class only exists to thread `action_mask` into the network's
forward() calls (needed to build the descriptor, not to mask Q-value
OUTPUT, which still happens via `_mask_q` exactly as in DQNAgent) --
everything else (act/act_batch/train/_remember_step) is inherited
unchanged from DQNAgent, since those methods dispatch through
`_greedy_action_batch`/`q_values`/`optimize`, the three methods overridden
here.
"""
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from agents.dqn import DQNAgent, Observation, _collate, _mask_q


class TopologyConditionedDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(topology_conditioned=True, **kwargs)

    def _greedy_action_batch(self, obs_list: List[Observation]) -> List[int]:
        self.q.eval()
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
            q = self.q(own, neighbors, neighbor_mask, hop_dist, action_mask=action_mask)
            q = _mask_q(q, action_mask)
            q_np = q.cpu().numpy()

        actions = []
        for row in q_np:
            max_q = np.max(row)
            tied = np.flatnonzero(np.isclose(row, max_q, atol=1e-4))
            actions.append(int(np.random.choice(tied)))
        return actions

    def q_values(self, obs: Observation, ts_id: Optional[str] = None) -> np.ndarray:
        self.q.eval()
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            q = self.q(own, neighbors, neighbor_mask, hop_dist, action_mask=action_mask)
            q = q.squeeze(0).cpu().numpy()
        mask = obs["action_mask"] > 0.5
        out = np.full_like(q, np.nan)
        out[mask] = q[mask]
        return out

    def optimize(self) -> Optional[float]:
        if len(self.replay) < max(4, self.batch_size):
            return None
        self.q.train()

        obs, actions, rewards, next_obs, dones, ns = self.replay.sample(self.batch_size)

        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, self.device)
        n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, self.device)

        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(1)
        discount_t = torch.full_like(ns_t, self.gamma).pow(ns_t)

        q_values = self.q(own, neighbors, neighbor_mask, hop_dist, action_mask=action_mask)
        q_taken = q_values.gather(1, actions_t)

        with torch.no_grad():
            next_q_online = _mask_q(
                self.q(n_own, n_neighbors, n_neighbor_mask, n_hop_dist, action_mask=n_action_mask),
                n_action_mask,
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(
                n_own, n_neighbors, n_neighbor_mask, n_hop_dist, action_mask=n_action_mask
            ).gather(1, next_actions)
            expected = rewards_t + (1.0 - dones_t) * discount_t * next_q_target

        loss = nn.functional.smooth_l1_loss(q_taken, expected, beta=1.0)

        if self.q_entropy_weight > 0:
            q_masked_for_entropy = _mask_q(q_values, action_mask)
            probs = torch.softmax(q_masked_for_entropy, dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            loss = loss - self.q_entropy_weight * entropy.mean()

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
