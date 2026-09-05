"""Item 23 (fidings/divergence_investigation.md, "Open questions" queue): a
recurrent policy -- every architecture tried in sec 73-76 (wider, deeper, more
attention layers) was still a purely reactive function of one tick's snapshot.
This gives the shared network a GRUCell-based memory of recent ticks, a
genuinely different axis (time, not capacity) from anything tested so far.

Design: "stored state" DRQN (Hausknecht & Stone 2015), not full R2D2-style
sequence replay. Each transition in the replay buffer carries the ACTUAL
hidden state that was live during rollout going into that tick (``h_in``) and
its successor (``h_out``, the hidden state that was live going into the NEXT
tick) -- both captured once, during ``act_batch``, and reused unmodified at
train time by both the online and target networks. This is simpler than
replaying whole episode sequences with burn-in, at the cost of some staleness
(the online network at train time has different weights than whatever
produced ``h_out`` during rollout) -- an accepted, well-precedented tradeoff,
not a full R2D2 implementation.

Known scope cuts (deliberate, not oversights):
  - n_step>1 is not supported combined with recurrence (raises ValueError) --
    an n-step accumulator window spans multiple ticks with its own hidden-
    state bookkeeping this class doesn't implement.
  - Evaluation-time statefulness requires the caller to pass ``ts_id`` into
    ``act()``/``q_values()`` and call ``reset_hidden()`` at episode
    boundaries. ``federated/evaluator.py`` does both (see its
    ``reset_hidden``/``ts_id`` call sites). Standalone diagnostic scripts
    (``diagnostics/finetune_on_holdout.py``, ``diagnostics/swa_reeval.py``)
    do NOT support this agent type yet, matching how they also don't support
    ``--algo ppo``/``munchausen``.
"""
from typing import Dict, List, Optional, Tuple
import random
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from agents.dqn import DQNAgent, Observation, _collate, _mask_q


class RecurrentReplayBuffer:
    """Like ``agents.dqn.ReplayBuffer``, but each transition also carries the
    hidden state that was actually live during rollout going into this tick
    (``h_in``) and going into the next tick (``h_out``) -- see this module's
    docstring for why these are stored rather than recomputed."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs: Observation, action: int, reward: float, next_obs: Observation,
            done: bool, h_in: "torch.Tensor", h_out: "torch.Tensor") -> None:
        self.buffer.append((obs, int(action), float(reward), next_obs, float(done), h_in, h_out))

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones, h_ins, h_outs = zip(*batch)
        return obs, actions, rewards, next_obs, dones, h_ins, h_outs

    def __len__(self):
        return len(self.buffer)


class RecurrentDQNAgent(DQNAgent):
    """DQN with a GRUCell-based hidden state carried per intersection across
    ticks within an episode. Same interface as ``DQNAgent`` (``act``,
    ``act_batch``, ``train``, ``optimize``, ``state_dict``/``load_state_dict``,
    ``q_values``) so it plugs into ``experiments/federated_training.py`` and
    ``federated/parallel_server.py`` via ``--algo recurrent`` exactly like PPO
    and Munchausen-DQN did.
    """

    def __init__(self, n_step: int = 1, **kwargs):
        if n_step != 1:
            raise ValueError(
                "RecurrentDQNAgent does not support n_step>1 -- an n-step "
                "accumulator window's hidden-state bookkeeping across multiple "
                "ticks is not implemented (see this module's docstring). Pass "
                "n_step=1 (the default)."
            )
        super().__init__(n_step=1, recurrent=True, **kwargs)
        self._hidden_dim = self.q.d_model * 2
        # Per-ts_id running hidden state -- the state that will be fed in as
        # h_in the NEXT time act()/act_batch() is called for that ts_id.
        self._hidden: Dict[str, "torch.Tensor"] = {}
        # Per-ts_id hidden state that was live going INTO the most recent
        # act()/act_batch() call -- what _remember_step stores as h_in, and
        # what q_values() reads (see q_values's docstring for why).
        self._last_h_in: Dict[str, "torch.Tensor"] = {}
        # Replaces the plain ReplayBuffer DQNAgent.__init__ already built --
        # same capacity, but stores h_in/h_out per transition.
        self.replay = RecurrentReplayBuffer(self.replay.buffer.maxlen)

    # ------------------------------------------------------------------
    # Hidden-state bookkeeping
    # ------------------------------------------------------------------

    def _zero_hidden(self) -> "torch.Tensor":
        return torch.zeros(self._hidden_dim, device=self.device)

    def reset_hidden(self, ts_id: Optional[str] = None) -> None:
        """Drop recurrent hidden state -- call at every episode boundary.
        ``train()`` does this automatically via ``_on_episode_start()``;
        eval callers (``federated/evaluator.py``) call it themselves once per
        episode, since evaluation doesn't go through ``train()``. ``ts_id=None``
        (default) clears every intersection's state; pass a specific id to
        clear just that one."""
        if ts_id is None:
            self._hidden = {}
            self._last_h_in = {}
        else:
            self._hidden.pop(ts_id, None)
            self._last_h_in.pop(ts_id, None)

    def _on_episode_start(self) -> None:
        self.reset_hidden()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: Observation, explore: bool = True, eps: Optional[float] = None,
            ts_id: Optional[str] = None) -> int:
        key = ts_id if ts_id is not None else "__default__"
        h_in = self._hidden.get(key, self._zero_hidden())

        self.q.eval()
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            q, h_out = self.q.forward_recurrent(own, neighbors, neighbor_mask, hop_dist, h_in.unsqueeze(0))
            q = _mask_q(q, action_mask)
            q_np = q.squeeze(0).cpu().numpy()

        # Hidden state advances every tick regardless of explore/exploit --
        # it's a function of having OBSERVED obs, not of which action got
        # taken, so an exploring tick still needs the "real" forward pass.
        self._last_h_in[key] = h_in
        self._hidden[key] = h_out.squeeze(0)

        if explore:
            eps = self._current_epsilon() if eps is None else eps
            if random.random() < eps:
                return self._random_valid_action(obs["action_mask"])

        max_q = np.max(q_np)
        tied = np.flatnonzero(np.isclose(q_np, max_q, atol=1e-4))
        return int(np.random.choice(tied))

    def act_batch(
        self,
        obs_dict: Dict[str, Observation],
        eps: Optional[float] = None,
        explore: bool = True,
    ) -> Dict[str, int]:
        """Batched version of ``act()`` across every intersection in
        ``obs_dict`` this tick. Unlike ``DQNAgent.act_batch`` (which skips the
        network entirely for exploring intersections, since a stateless
        network has nothing to update), EVERY intersection's hidden state
        must advance every tick here -- so every intersection goes through
        one batched forward pass regardless of the explore/exploit draw."""
        if explore and eps is None:
            eps = self._current_epsilon()

        ts_ids = list(obs_dict.keys())
        obs_list = [obs_dict[t] for t in ts_ids]
        h_in_batch = torch.stack([self._hidden.get(t, self._zero_hidden()) for t in ts_ids], dim=0)

        self.q.eval()
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
            q, h_out_batch = self.q.forward_recurrent(own, neighbors, neighbor_mask, hop_dist, h_in_batch)
            q = _mask_q(q, action_mask)
            q_np = q.cpu().numpy()

        actions: Dict[str, int] = {}
        for i, ts_id in enumerate(ts_ids):
            self._last_h_in[ts_id] = h_in_batch[i]
            self._hidden[ts_id] = h_out_batch[i]
            if explore and random.random() < eps:
                actions[ts_id] = self._random_valid_action(obs_dict[ts_id]["action_mask"])
            else:
                row = q_np[i]
                max_q = np.max(row)
                tied = np.flatnonzero(np.isclose(row, max_q, atol=1e-4))
                actions[ts_id] = int(np.random.choice(tied))
        return actions

    def q_values(self, obs: Observation, ts_id: Optional[str] = None) -> np.ndarray:
        """Masked Q-values for a single observation, read-only: uses the
        hidden state that was live going INTO the most recent act()/act_batch()
        call for this ts_id (``_last_h_in``), NOT the current (already-advanced)
        ``_hidden`` entry -- this is a diagnostic-only call (evaluator.py's
        Q-gap logging, separate from and after the act() call that already
        produced this tick's action); running another stateful forward here
        would double-advance the hidden state for that tick, and reading the
        already-advanced state would report next tick's Q-values under this
        tick's label."""
        key = ts_id if ts_id is not None else "__default__"
        h_in = self._last_h_in.get(key, self._zero_hidden())

        self.q.eval()
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            q, _ = self.q.forward_recurrent(own, neighbors, neighbor_mask, hop_dist, h_in.unsqueeze(0))
            q = q.squeeze(0).cpu().numpy()

        mask = obs["action_mask"] > 0.5
        out = np.full_like(q, np.nan)
        out[mask] = q[mask]
        return out

    # ------------------------------------------------------------------
    # Replay + optimization
    # ------------------------------------------------------------------

    def _remember_step(self, ts_id: str, obs: Observation, action: int, reward: float,
                        next_obs: Observation, done: bool) -> None:
        h_in = self._last_h_in.get(ts_id, self._zero_hidden())
        h_out = self._hidden.get(ts_id, self._zero_hidden())
        self.replay.add(obs, action, self._clip_reward(reward), next_obs, done, h_in, h_out)

    def optimize(self) -> Optional[float]:
        if len(self.replay) < max(4, self.batch_size):
            return None
        self.q.train()

        obs, actions, rewards, next_obs, dones, h_ins, h_outs = self.replay.sample(self.batch_size)

        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, self.device)
        n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, self.device)
        h_in_batch = torch.stack(h_ins, dim=0).to(self.device)
        h_out_batch = torch.stack(h_outs, dim=0).to(self.device)

        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values, _ = self.q.forward_recurrent(own, neighbors, neighbor_mask, hop_dist, h_in_batch)
        q_taken = q_values.gather(1, actions_t)

        with torch.no_grad():
            # Double DQN, same as DQNAgent.optimize() -- both networks reuse
            # the SAME stored h_out_batch (the hidden state actually observed
            # during rollout) rather than each recomputing its own trajectory;
            # see this module's docstring for the "stored state" tradeoff.
            next_q_online, _ = self.q.forward_recurrent(
                n_own, n_neighbors, n_neighbor_mask, n_hop_dist, h_out_batch
            )
            next_q_online = _mask_q(next_q_online, n_action_mask)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.q_target.forward_recurrent(
                n_own, n_neighbors, n_neighbor_mask, n_hop_dist, h_out_batch
            )
            next_q_target = next_q_target.gather(1, next_actions)
            # n_step is forced to 1 in __init__, so gamma**1 == self.gamma exactly.
            expected = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

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
