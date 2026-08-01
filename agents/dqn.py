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

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


Observation = Dict[str, np.ndarray]
# Expected keys: "own", "neighbors", "neighbor_mask", "hop_dist", "action_mask"


class ReplayBuffer:
    """Stores structured (dict) observations from ALL intersections of a
    city, pooled together. Pooling across intersections -- rather than one
    buffer per intersection -- is what forces the shared network to learn a
    genuinely topology-agnostic policy instead of overfitting to one
    intersection's quirks.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs: Observation, action: int, reward: float, next_obs: Observation, done: bool):
        self.buffer.append((obs, int(action), float(reward), next_obs, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


def _collate(obs_list, device):
    """Batch a list of observation dicts into tensors."""
    own = torch.tensor(
        np.array([o["own"] for o in obs_list]), dtype=torch.float32, device=device
    )
    neighbors = torch.tensor(
        np.array([o["neighbors"] for o in obs_list]), dtype=torch.float32, device=device
    )
    neighbor_mask = torch.tensor(
        np.array([o["neighbor_mask"] for o in obs_list]), dtype=torch.float32, device=device
    )
    hop_dist = torch.tensor(
        np.array([
            o.get("hop_dist", np.zeros(o["neighbors"].shape[0], dtype=np.int64))
            for o in obs_list
        ]),
        dtype=torch.long,
        device=device,
    )
    action_mask = torch.tensor(
        np.array([o["action_mask"] for o in obs_list]), dtype=torch.float32, device=device
    )
    return own, neighbors, neighbor_mask, hop_dist, action_mask


def _mask_q(q: "torch.Tensor", action_mask: "torch.Tensor") -> "torch.Tensor":
    """Set Q-values of invalid actions to -inf so argmax / target
    computation never picks or bootstraps off an action that doesn't
    exist for this intersection's topology."""
    neg_inf = torch.finfo(q.dtype).min
    return q.masked_fill(action_mask < 0.5, neg_inf)


class DQNAgent:
    """Torch-based DQN agent with a shared, topology-agnostic Q-network.

    One instance of this agent is trained against every intersection in a
    city simultaneously (see ``train``): the same weights are used to act
    for every ``ts_id`` each tick, and every intersection's transitions are
    stored in the same replay buffer. This is what makes it a "foundation
    model" for intersections rather than N separate per-intersection
    models.
    """

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
        eps_decay: float = 20000.0,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
        head_fix: bool = True,
        tau: float = 0.005,
    ):
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.action_dim = action_dim
        self.k_max = k_max
        self.device = torch.device(device)

        net_kwargs = dict(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            action_dim=action_dim,
            k_max=k_max,
            d_model=d_model,
            n_heads=n_heads,
            n_hops=n_hops,
            head_fix=head_fix,
        )
        self.q = NeighborAttentionQNetwork(**net_kwargs).to(self.device)
        self.q_target = NeighborAttentionQNetwork(**net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # AdamW, not Adam: plain torch.optim.Adam folds weight_decay into the
        # gradient before the moment estimates, so a parameter that gets
        # ~zero true gradient (e.g. a masked-out Q-head row for an action a
        # low-action-count city never takes) still gets a "gradient" of
        # weight_decay*param every step. Once Adam's running estimates lock
        # onto that as a near-constant signal, the normalized update
        # collapses to essentially -lr*sign(param) -- decaying that weight
        # at a rate set by the LEARNING RATE, not by the (tiny) weight_decay
        # coefficient. Over the hundreds of optimizer steps in one local
        # round this silently erases untouched action rows/features.
        # AdamW applies weight_decay as a separate, decoupled term
        # (param -= lr*weight_decay*param), which behaves like the mild
        # regularizer weight_decay=1e-5 was actually meant to be.
        self.optimizer = optim.AdamW(self.q.parameters(), lr=lr, weight_decay=1e-5, eps=1e-6)
        self.lr_decay = lr_decay   # multiplicative decay applied once per federated round
        self.min_lr = min_lr
        self.replay = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update = target_update
        self.tau = tau          # 0 = legacy hard copy; >0 = Polyak soft update every step
        self.learn_steps = 0
        # Reward clipping: applied when a transition is stored, not just
        # at optimize() time -- this is what actually stopped city_2's
        # loss blow-up (a few large-magnitude congestion rewards were
        # dominating the squared-error gradient). Set to None to disable.
        self.reward_clip = reward_clip
        logger.info(
            "own_dim=%d neighbor_dim=%d action_dim=%d k_max=%d eps_decay=%.0f",
            self.own_dim, self.neighbor_dim, self.action_dim, self.k_max, self.eps_decay,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_actions(action_mask: np.ndarray) -> np.ndarray:
        valid = np.nonzero(action_mask > 0.5)[0]
        return valid

    def _current_epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

    def current_epsilon(self) -> float:
        """Public read-only view of the current epsilon value."""
        return self._current_epsilon()

    def _greedy_action(self, obs: Observation) -> int:
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            q = self.q(own, neighbors, neighbor_mask, hop_dist)
            q = _mask_q(q, action_mask)
            q_np = q.squeeze(0).cpu().numpy()
            # Break ties randomly instead of always taking the lowest
            # index. Early in training (or on a symmetric network where
            # two actions are genuinely equivalent) Q-values for different
            # actions can be numerically indistinguishable; torch.argmax
            # always resolves that to the lowest index, which silently
            # produces a "the model always picks action 0" pattern that
            # looks like a learned policy but is actually just tie-break
            # bias. Ties within a small tolerance are resolved uniformly
            # at random instead.
            max_q = np.max(q_np)
            tied = np.flatnonzero(np.isclose(q_np, max_q, atol=1e-4))
            return int(np.random.choice(tied))

    def q_values(self, obs: Observation) -> np.ndarray:
        """Masked Q-values for a single observation (invalid actions are
        NaN, not -inf, so callers can distinguish 'invalid' from 'valid
        but low' when inspecting/logging)."""
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            q = self.q(own, neighbors, neighbor_mask, hop_dist).squeeze(0).cpu().numpy()
        mask = obs["action_mask"] > 0.5
        out = np.full_like(q, np.nan)
        out[mask] = q[mask]
        return out

    def _epsilon_action(self, obs: Observation, eps: float) -> int:
        # NOTE: steps_done is advanced once per environment TICK (see
        # `train()`), not once per intersection per tick. Incrementing it
        # here (per-intersection) would make epsilon decay N times faster
        # in a city with N intersections than in a single-intersection
        # city, which desyncs the exploration schedule across the very
        # cities that get FedAvg'd together every round.
        if random.random() < eps:
            valid = self._valid_actions(obs["action_mask"])
            if len(valid) == 0:
                valid = np.arange(self.action_dim)
            return int(random.choice(valid))
        return self._greedy_action(obs)

    def act(self, obs: Observation, explore: bool = True, eps: Optional[float] = None) -> int:
        if explore:
            eps = self._current_epsilon() if eps is None else eps
            return self._epsilon_action(obs, eps)
        return self._greedy_action(obs)

    def select_action(self, obs: Observation) -> int:
        return self.act(obs, explore=True)

    # ------------------------------------------------------------------
    # Replay + optimization
    # ------------------------------------------------------------------

    def remember(self, obs: Observation, action: int, reward: float, next_obs: Observation, done: bool) -> None:
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        self.replay.add(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        return self.optimize()

    def optimize(self) -> Optional[float]:
        if len(self.replay) < max(4, self.batch_size):
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)

        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs, self.device)
        n_own, n_neighbors, n_neighbor_mask, n_hop_dist, n_action_mask = _collate(next_obs, self.device)

        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q(own, neighbors, neighbor_mask, hop_dist)
        q_taken = q_values.gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: select next action with the online net, but only
            # among actions valid for that next observation's topology.
            next_q_online = _mask_q(
                self.q(n_own, n_neighbors, n_neighbor_mask, n_hop_dist), n_action_mask
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(n_own, n_neighbors, n_neighbor_mask, n_hop_dist).gather(
                1, next_actions
            )
            expected = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        # Huber loss instead of MSE: MSE squares the TD-error, so a single
        # large-magnitude transition (e.g. a congestion spike) can
        # dominate the whole gradient step -- that's what was driving
        # city_2's loss climbing into the double digits. Huber is linear
        # past `delta`, so outliers contribute a bounded gradient instead
        # of an exploding one.
        loss = nn.functional.smooth_l1_loss(q_taken, expected, beta=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.tau > 0:
            # Polyak (soft) update: target = (1-tau)*target + tau*online
            # Applied every step with small tau so the target drifts
            # smoothly instead of snapping every target_update steps.
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        elif self.learn_steps % self.target_update == 0:
            # Legacy hard copy: kept for ablation (pass tau=0 to enable).
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def decay_lr(self) -> float:
        """Multiply the optimizer's LR by ``lr_decay``, floored at ``min_lr``.

        Call once per federated round (not per gradient step) -- e.g. from
        the client after ``train()`` returns, or from the server after
        aggregation. A per-city ``lr_decay`` close to 1.0 keeps the rate
        essentially fixed; a value like 0.97 gives a gentle exponential
        decay so a noisy, small city's updates get progressively gentler
        as training proceeds, without needing a fixed schedule length.
        Returns the new LR for logging.
        """
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.lr_decay, self.min_lr)
            group["lr"] = new_lr
        return self.optimizer.param_groups[0]["lr"]

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(state)

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        return {k: v.cpu() for k, v in self.q.state_dict().items()}

    def load_state_dict(self, state: Dict[str, "torch.Tensor"]) -> None:
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(state)

    # ------------------------------------------------------------------
    # Training loop -- multi-agent aware
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        episodes: int = 5,
        log_loss_every_steps: int = 50,
    ) -> Tuple[Dict[str, "torch.Tensor"], int]:
        """Train against a multi-agent city environment.

        Contract expected of ``env``:
            env.reset()            -> Dict[ts_id, obs_dict]
            env.step(actions)      -> (Dict[ts_id, obs_dict],
                                        Dict[ts_id, float] rewards,
                                        Dict[ts_id, bool] dones (may
                                            include "__all__"),
                                        info dict)

        Every intersection in the city is stepped simultaneously each
        tick, using the SAME shared network to pick each intersection's
        action; every intersection's transition is stored in the same
        replay buffer.
        """
        total_steps = 0
        all_losses: List[float] = []   # across every episode this call
        action_counts: Counter = Counter()   # {action_index: times_taken}, this call

        for ep in range(1, episodes + 1):
            obs_dict = env.reset()
            if not isinstance(obs_dict, dict):
                # Backward-compat: single-agent env returning a flat obs.
                obs_dict = {"__single__": obs_dict}

            done = False
            ep_steps = 0
            ep_losses: List[float] = []

            while not done:
                # One shared epsilon for this tick, used by every
                # intersection -- see _current_epsilon's docstring for
                # why this must NOT be per-intersection.
                eps = self._current_epsilon()
                actions = {ts_id: self.act(o, explore=True, eps=eps) for ts_id, o in obs_dict.items()}
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
                    self.remember(o, actions[ts_id], r, no, d)

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
                    msg = (
                        f"  [train] ep={ep}/{episodes}  step={ep_steps}"
                        f"  loss(last {len(window)})={np.mean(window):.6f}"
                    )
                    logger.info(msg)

            if ep_losses:
                msg = (
                    f"[train] ep={ep}/{episodes}  steps={ep_steps}"
                    f"  loss  mean={np.mean(ep_losses):.6f}"
                    f"  min={np.min(ep_losses):.6f}"
                    f"  max={np.max(ep_losses):.6f}"
                    f"  updates={len(ep_losses)}"
                )
            else:
                msg = f"[train] ep={ep}/{episodes}  steps={ep_steps}  loss=n/a (buffer not yet full)"
            logger.info(msg)

        # Mean loss across every gradient step this call -- forwarded by
        # FederatedClient/CityClient to the server so aggregation strategies
        # that need a loss signal (EMA-based ones) have real data instead
        # of None every round.
        mean_loss = float(np.mean(all_losses)) if all_losses else None
        # Plain dict, not Counter -- keeps it picklable/simple across the
        # process boundary in the parallel path.
        return self.state_dict(), total_steps, mean_loss, dict(action_counts)
