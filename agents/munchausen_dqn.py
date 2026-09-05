"""Munchausen-DQN (Vieillard et al. 2020, "Munchausen Reinforcement Learning")
for the same federated, topology-agnostic intersection-control setup
DQNAgent (agents/dqn.py) and PPOAgent (agents/ppo.py) were built for.

Why this, after PPO: agents/ppo.py's module docstring covers the
motivation for wanting a policy that structurally resists the "confident
lock-in" failure mode (fidings/divergence_investigation.md sec 32-34/51-57/
70) -- argmax(Q) can collapse onto one repeating action; a stochastic
policy can't the same way. PPO's first real pilot (sec 72, pilot D) showed
a real confound: PPO is ON-policy (each round's rollout is used once, then
discarded), while DQN is OFF-policy with a persistent replay buffer reused
across many gradient steps -- at this project's standard tiny
`--local_episodes 2`, that gives DQN a large, uncontrolled sample-
efficiency advantage that has nothing to do with which algorithm actually
suits this task better.

Munchausen-DQN keeps DQN's off-policy machinery (replay buffer, same
network, same sample efficiency, directly comparable at the SAME episode
budget) but replaces the hard argmax(Q) policy with a Boltzmann/softmax
policy over Q-values, softly regularized:

  - Action selection: sample from softmax(Q(s,.)/tau) instead of
    epsilon-greedy. No epsilon schedule -- temperature tau controls
    exploration directly, annealed on the same schedule shape DQN's
    epsilon uses (see current_epsilon()'s reuse below, TEMPERATURE not
    EPSILON, but same decay math).
  - Training target adds two entropy-regularized terms on top of the
    ordinary 1-step (or n-step) TD target:
      1. The "Munchausen term" -- alpha * tau * log pi(a|s), the target
         network's own log-probability of the action actually taken,
         clipped from below at l0 so a near-zero-probability action
         doesn't inject an enormous bonus/penalty. This is what makes it
         "Munchausen" (the agent bootstraps off its own logged policy) --
         empirically the paper's biggest single contributor, distinct
         from plain soft-Q-learning.
      2. The soft state value of the NEXT state,
         V_soft(s') = tau * logsumexp(Q_target(s',.)/tau) over valid
         actions only, replacing DQN's max_a' Q_target(s',a') -- this is
         what keeps the target itself entropy-aware, not just the acting
         policy.
    Both terms are computed off the TARGET network, matching the paper.

Interface contract: identical to DQNAgent/PPOAgent (see agents/ppo.py's
module docstring for the exact list) -- start_round/current_epsilon/train/
decay_lr/state_dict/load_state_dict/act, so no federated plumbing changes
were needed beyond a new --algo munchausen choice. Unlike PPOAgent, this
uses the PLAIN (non-actor-critic) NeighborAttentionQNetwork -- same Q-head
key names as DQNAgent, so masked-head aggregation works unmodified (no
head_fix-off guard needed, unlike PPO).
"""
from typing import Dict, List, Optional, Tuple, Any
import random
import logging
import math
from collections import Counter, deque

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from agents.networks import NeighborAttentionQNetwork
    from agents.dqn import ReplayBuffer, _collate, _mask_q

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


Observation = Dict[str, np.ndarray]


class MunchausenDQNAgent:
    """Off-policy, entropy-regularized DQN variant. One instance trained
    against every intersection in a city simultaneously, same "foundation
    model per city" design as DQNAgent."""

    def __init__(
        self,
        own_dim: int = 4,
        neighbor_dim: int = 3,
        action_dim: int = 2,
        k_max: int = 8,
        lr: float = 1e-3,
        device: str = "cpu",
        buffer_size: int = 50000,
        batch_size: int = 64,
        gamma: float = 0.99,
        target_update: int = 200,
        d_model: int = 128,
        n_heads: int = 4,
        n_hops: int = 4,
        reward_clip: Optional[float] = 10.0,
        temp_decay: float = 20000.0,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
        head_fix: bool = True,
        tau: float = 0.005,
        dueling: bool = False,
        n_step: int = 1,
        init_steps_done: int = 0,
        munchausen_alpha: float = 0.9,
        munchausen_temp: float = 0.03,
        munchausen_l0: float = -1.0,
        **_ignored_dqn_only_kwargs: Any,
    ):
        # _ignored_dqn_only_kwargs: shares one CLI/factory surface with
        # DQNAgent and PPOAgent (mu, q_entropy_weight, ...) -- see
        # agents/ppo.py's matching comment. Explicit no-op, not a typo trap.
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.action_dim = action_dim
        self.k_max = k_max
        self.device = torch.device(device)

        net_kwargs = dict(
            own_dim=own_dim, neighbor_dim=neighbor_dim, action_dim=action_dim,
            k_max=k_max, d_model=d_model, n_heads=n_heads, n_hops=n_hops,
            head_fix=head_fix, dueling=dueling,
        )
        self.q = NeighborAttentionQNetwork(**net_kwargs).to(self.device)
        self.q_target = NeighborAttentionQNetwork(**net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.AdamW(self.q.parameters(), lr=lr, weight_decay=1e-5, eps=1e-6)
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.replay = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update = target_update
        self.learn_steps = 0
        self.reward_clip = reward_clip

        # Softmax temperature schedule -- same exponential-decay shape as
        # DQNAgent's epsilon (current_epsilon()), just named for what it
        # actually is here: the Boltzmann policy's temperature, not an
        # explore-vs-exploit coin flip. temp_end (not 0) is a floor so the
        # policy never fully degenerates to hard argmax even late in
        # training -- the whole point of this agent.
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay = temp_decay
        self.steps_done = init_steps_done

        self.n_step = max(1, int(n_step))
        self._nstep_buffers: Dict[str, "deque"] = {}

        # Munchausen RL's own hyperparameters (Vieillard et al. 2020,
        # defaults close to the paper's): alpha weights the "bootstrap off
        # your own logged policy" bonus term; munchausen_temp (their tau)
        # is a SEPARATE, typically much smaller temperature used only
        # inside the soft-Bellman target math (both the log-policy term and
        # the next-state soft value) -- kept distinct from the
        # action-selection temp_start/temp_end schedule above, matching the
        # paper's own separation of acting temperature from target
        # temperature. l0 clips the log-policy term from below so a
        # near-zero-probability action doesn't inject an unbounded bonus.
        self.munchausen_alpha = float(munchausen_alpha)
        self.munchausen_temp = float(munchausen_temp)
        self.munchausen_l0 = float(munchausen_l0)

        logger.info(
            "MunchausenDQNAgent: own_dim=%d neighbor_dim=%d action_dim=%d k_max=%d "
            "temp_start=%.2f temp_end=%.2f munchausen_alpha=%.2f munchausen_temp=%.4f",
            own_dim, neighbor_dim, action_dim, k_max,
            temp_start, temp_end, munchausen_alpha, munchausen_temp,
        )

    # ------------------------------------------------------------------
    # Interface parity with DQNAgent/PPOAgent
    # ------------------------------------------------------------------

    def current_epsilon(self) -> float:
        """Logging-only stub for interface parity -- returns the current
        softmax TEMPERATURE (not an epsilon), same schedule math as
        DQNAgent's epsilon decay."""
        return self._current_temperature()

    def _current_temperature(self) -> float:
        return self.temp_end + (self.temp_start - self.temp_end) * math.exp(
            -1.0 * self.steps_done / self.temp_decay
        )

    def start_round(self, global_state: Dict[str, "torch.Tensor"]) -> None:
        self.load_state_dict(global_state)

    def clear_replay(self) -> None:
        """See agents/dqn.py::ReplayBuffer.clear()'s docstring (item 20,
        fidings sec 78)."""
        self.replay.clear()
        self._nstep_buffers = {}

    def decay_lr(self) -> float:
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.lr_decay, self.min_lr)
            group["lr"] = new_lr
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        return {k: v.cpu() for k, v in self.q.state_dict().items()}

    def load_state_dict(self, state: Dict[str, "torch.Tensor"]) -> None:
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(state)

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(state)

    # ------------------------------------------------------------------
    # Action selection -- Boltzmann policy over Q, not epsilon-greedy
    # ------------------------------------------------------------------

    def _sample_batch(self, obs_list: List[Observation], temperature: float) -> List[int]:
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
            q = _mask_q(self.q(own, neighbors, neighbor_mask, hop_dist), action_mask)
            dist = torch.distributions.Categorical(logits=q / max(temperature, 1e-6))
            actions = dist.sample()
        return [int(a) for a in actions.cpu().numpy()]

    def act_batch(
        self,
        obs_dict: Dict[str, Observation],
        eps: Optional[float] = None,   # accepted for interface parity, unused
        explore: bool = True,
    ) -> Dict[str, int]:
        ts_ids = list(obs_dict.keys())
        obs_list = [obs_dict[t] for t in ts_ids]
        if explore:
            temperature = self._current_temperature()
            actions = self._sample_batch(obs_list, temperature)
        else:
            with torch.no_grad():
                own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
                q = _mask_q(self.q(own, neighbors, neighbor_mask, hop_dist), action_mask)
                actions = torch.argmax(q, dim=-1).cpu().numpy().tolist()
        return dict(zip(ts_ids, actions))

    def act(self, obs: Observation, explore: bool = True, eps: Optional[float] = None) -> int:
        """Single-observation action -- what federated/evaluator.py calls
        at eval time (explore=False -> deterministic argmax, matching this
        project's pure-argmax-at-eval convention, sec 35)."""
        return self.act_batch({"__single__": obs}, explore=explore)["__single__"]

    # ------------------------------------------------------------------
    # Replay + n-step accumulation (identical to DQNAgent's, reused
    # verbatim rather than imported to keep this file self-contained and
    # avoid coupling to DQNAgent internals that might change independently)
    # ------------------------------------------------------------------

    def _clip_reward(self, r: float) -> float:
        if self.reward_clip is not None:
            return float(np.clip(r, -self.reward_clip, self.reward_clip))
        return r

    def _remember_step(self, ts_id: str, obs: Observation, action: int, reward: float,
                        next_obs: Observation, done: bool) -> None:
        if self.n_step <= 1:
            r = self._clip_reward(reward)
            self.replay.add(obs, action, r, next_obs, done, 1)
            return
        r = self._clip_reward(reward)
        buf = self._nstep_buffers.setdefault(ts_id, deque())
        buf.append((obs, action, r))
        if len(buf) >= self.n_step:
            self._flush_nstep(buf, next_obs, done)
        if done:
            while buf:
                self._flush_nstep(buf, next_obs, done)

    def _flush_nstep(self, buf: "deque", next_obs: Observation, done: bool) -> None:
        obs0, action0, _ = buf[0]
        n = len(buf)
        ret = 0.0
        for i, (_, _, r) in enumerate(buf):
            ret += (self.gamma ** i) * r
        self.replay.add(obs0, action0, ret, next_obs, done, n)
        buf.popleft()

    # ------------------------------------------------------------------
    # Optimization -- Munchausen soft-Bellman target
    # ------------------------------------------------------------------

    def optimize(self) -> Optional[float]:
        if len(self.replay) < max(4, self.batch_size):
            return None

        obs, actions, rewards, next_obs, dones, ns = self.replay.sample(self.batch_size)

        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, self.device)
        n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, self.device)

        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(1)
        discount_t = torch.full_like(ns_t, self.gamma).pow(ns_t)

        q_values = self.q(own, neighbors, neighbor_mask, hop_dist)
        q_taken = q_values.gather(1, actions_t)

        mtau = self.munchausen_temp
        with torch.no_grad():
            # -- Munchausen term: target network's own log-policy of the
            # action actually taken, clipped from below at l0.
            q_target_cur = _mask_q(self.q_target(own, neighbors, neighbor_mask, hop_dist), action_mask)
            log_pi_cur = torch.log_softmax(q_target_cur / mtau, dim=1)
            log_pi_taken = log_pi_cur.gather(1, actions_t)
            munchausen_term = self.munchausen_alpha * torch.clamp(
                mtau * log_pi_taken, min=self.munchausen_l0, max=0.0
            )

            # -- Soft value of the next state under the target network:
            # V_soft(s') = tau * logsumexp(Q_target(s',.)/tau) over valid
            # actions. Invalid actions are already -inf (via _mask_q), so
            # logsumexp naturally excludes them without a separate mask
            # renormalization step.
            q_target_next = _mask_q(self.q_target(n_own, n_neighbors, n_neighbor_mask, n_hop_dist), n_action_mask)
            v_soft_next = mtau * torch.logsumexp(q_target_next / mtau, dim=1, keepdim=True)

            expected = (
                rewards_t + munchausen_term
                + (1.0 - dones_t) * discount_t * v_soft_next
            )

        loss = nn.functional.smooth_l1_loss(q_taken, expected, beta=1.0)

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

        return float(loss.item())

    # ------------------------------------------------------------------
    # Training loop -- multi-agent aware (structurally identical to
    # DQNAgent.train(): off-policy, one gradient step per tick against a
    # persistent replay buffer -- the whole point of this agent is to keep
    # this loop shape so it's directly, fairly comparable to DQN at the
    # same episode budget, unlike PPOAgent's on-policy rollout-then-update)
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        episodes: int = 5,
        log_loss_every_steps: int = 50,
    ) -> Tuple[Dict[str, "torch.Tensor"], int, Optional[float], Dict[int, int]]:
        total_steps = 0
        all_losses: List[float] = []
        action_counts: Counter = Counter()

        for ep in range(1, episodes + 1):
            obs_dict = env.reset()
            if not isinstance(obs_dict, dict):
                obs_dict = {"__single__": obs_dict}
            self._nstep_buffers = {}

            done = False
            ep_steps = 0
            ep_losses: List[float] = []

            while not done:
                actions = self.act_batch(obs_dict, explore=True)
                self.steps_done += 1
                action_counts.update(actions.values())

                if len(actions) == 1 and "__single__" in actions:
                    next_obs, reward, done, _ = env.step(actions["__single__"])
                    next_obs_dict = {"__single__": next_obs}
                    rewards = {"__single__": reward}
                    dones = {"__single__": done, "__all__": done}
                else:
                    next_obs_dict, rewards, dones, _ = env.step(actions)

                for ts_id, o in obs_dict.items():
                    r = rewards.get(ts_id, 0.0)
                    no = next_obs_dict.get(ts_id, o)
                    d = dones.get(ts_id, dones.get("__all__", False))
                    self._remember_step(ts_id, o, actions[ts_id], r, no, d)

                loss = self.optimize()
                if loss is not None:
                    ep_losses.append(loss)
                    all_losses.append(loss)

                obs_dict = next_obs_dict
                done = dones.get("__all__", all(dones.values()) if dones else True)
                ep_steps += 1
                total_steps += 1

                if log_loss_every_steps > 0 and ep_losses and ep_steps % log_loss_every_steps == 0:
                    window = ep_losses[-log_loss_every_steps:]
                    logger.info(
                        "  [train] ep=%d/%d  step=%d  loss(last %d)=%.6f  temp=%.4f",
                        ep, episodes, ep_steps, len(window), np.mean(window),
                        self._current_temperature(),
                    )

            if ep_losses:
                logger.info(
                    "[train] ep=%d/%d  steps=%d  loss  mean=%.6f min=%.6f max=%.6f updates=%d",
                    ep, episodes, ep_steps, np.mean(ep_losses), np.min(ep_losses),
                    np.max(ep_losses), len(ep_losses),
                )
            else:
                logger.info("[train] ep=%d/%d  steps=%d  loss=n/a (buffer not yet full)",
                            ep, episodes, ep_steps)

        mean_loss = float(np.mean(all_losses)) if all_losses else None
        return self.state_dict(), total_steps, mean_loss, dict(action_counts)
