"""PPO agent for the same federated, topology-agnostic intersection-control
setup DQNAgent (agents/dqn.py) was built for.

Why: fidings/divergence_investigation.md sec 32-34/51-57/70 characterized a
"confident lock-in" failure mode in the DQN pipeline -- once argmax(Q)
becomes very peaked toward one action, the (deterministic, exploit-only-at-
eval) policy keeps repeating it regardless of the actual traffic state,
sometimes producing byte-identical rewards across 30 different SUMO seeds.
An on-policy actor-critic with an entropy-regularized STOCHASTIC policy
(PPO) can't collapse to a single deterministic action the same way -- the
policy is a distribution, sampled every tick, with an explicit entropy bonus
in the loss keeping it from fully collapsing. This is the algorithm-level
lever discussed as a follow-on to pilots A/B/C (sec 72), which all targeted
the DQN pipeline's training/fine-tuning without changing the underlying
algorithm.

Interface contract (deliberately matched to DQNAgent so the federated
plumbing needs zero changes -- see federated/client.py's docstring: both
the sequential (FederatedClient) and parallel (parallel_server.py) paths
only ever call ``agent.start_round()``, ``agent.current_epsilon()``,
``agent.train(env, episodes=, log_loss_every_steps=)``, ``agent.decay_lr()``,
``agent.state_dict()``/``load_state_dict()`` -- never anything DQN-specific
like ``.replay`` or ``.optimize()`` directly):

    start_round(global_state)                  -> None
    current_epsilon()                           -> float (always 0.0 here --
                                                    PPO's exploration is the
                                                    stochastic policy itself,
                                                    not an epsilon schedule;
                                                    this is a pure logging
                                                    stub, never used for a
                                                    decision)
    train(env, episodes, log_loss_every_steps)  -> (state_dict, n_samples,
                                                      mean_loss, action_counts)
    decay_lr()                                   -> float (new LR)
    state_dict() / load_state_dict()             -> network weights only,
                                                     same as DQNAgent (no
                                                     optimizer state crosses
                                                     the federation boundary)

Scope note: wired into the SEQUENTIAL path only for now
(federated/client.py + FederatedServer), via --algo ppo. NOT yet wired into
the parallel multiprocess path (parallel_server.py) -- that needs the agent
class to be importable/constructible inside a spawned worker process, which
this class supports (plain constructor args, no shared state), but the
parallel-path plumbing itself hasn't been touched yet. Intentional: this is
a small/cheap dummy-training smoke test to see if PPO is worth the bigger
integration effort before doing it.
"""
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from agents.networks import NeighborAttentionQNetwork
    from agents.dqn import _collate

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


Observation = Dict[str, np.ndarray]


def _mask_logits(logits: "torch.Tensor", action_mask: "torch.Tensor") -> "torch.Tensor":
    """Same convention as agents/dqn.py::_mask_q -- invalid actions get
    -inf so softmax assigns them exactly 0 probability."""
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(action_mask < 0.5, neg_inf)


class RolloutBuffer:
    """Per-(episode, ts_id) trajectory storage.

    Unlike DQNAgent's ReplayBuffer (pooled across intersections, sampled
    off-policy, reused across many rounds), PPO is on-policy: this buffer
    holds exactly one round's worth of fresh trajectories, grouped by
    (episode, ts_id) so GAE can be computed correctly per-trajectory
    (each intersection's own sequential experience within one episode),
    then flattened across intersections for the actual gradient update --
    same "pool across intersections for the update, but respect per-
    intersection sequence for anything temporal" split DQNAgent uses
    between its per-ts_id n-step windows and its pooled replay buffer.
    """

    def __init__(self):
        self.trajectories: Dict[Tuple[int, str], List[dict]] = {}

    def add(self, ep: int, ts_id: str, obs: Observation, action: int,
            log_prob: float, value: float, reward: float, done: bool) -> None:
        key = (ep, ts_id)
        self.trajectories.setdefault(key, []).append(dict(
            obs=obs, action=action, log_prob=log_prob, value=value,
            reward=reward, done=done,
        ))

    def clear(self) -> None:
        self.trajectories = {}

    def __len__(self) -> int:
        return sum(len(t) for t in self.trajectories.values())


class PPOAgent:
    """PPO agent sharing one network across every intersection in a city,
    same "foundation model per city" design as DQNAgent."""

    def __init__(
        self,
        own_dim: int = 4,
        neighbor_dim: int = 3,
        action_dim: int = 2,
        k_max: int = 8,
        lr: float = 3e-4,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        d_model: int = 128,
        n_heads: int = 4,
        n_hops: int = 4,
        reward_clip: Optional[float] = 10.0,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
        head_fix: bool = True,
        **_ignored_dqn_only_kwargs: Any,
    ):
        # _ignored_dqn_only_kwargs: --algo ppo shares one CLI surface with
        # DQN (tau, target_update, mu, dueling, n_step, q_entropy_weight,
        # init_steps_done, ...) via _make_agent's default-arg closure in
        # experiments/federated_training.py. Rather than duplicate every
        # DQN-only flag's plumbing here, PPOAgent just accepts and ignores
        # them -- explicit no-op, not a silent typo trap, since every name
        # that lands here is DQN-specific by construction (the call site
        # only forwards args DQNAgent's signature already declares).
        self.own_dim = own_dim
        self.neighbor_dim = neighbor_dim
        self.action_dim = action_dim
        self.k_max = k_max
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.reward_clip = reward_clip
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        self.net = NeighborAttentionQNetwork(
            own_dim=own_dim, neighbor_dim=neighbor_dim, action_dim=action_dim,
            k_max=k_max, d_model=d_model, n_heads=n_heads, n_hops=n_hops,
            head_fix=head_fix, actor_critic=True,
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        logger.info(
            "PPOAgent: own_dim=%d neighbor_dim=%d action_dim=%d k_max=%d "
            "clip_eps=%.2f ppo_epochs=%d entropy_coef=%.4f",
            own_dim, neighbor_dim, action_dim, k_max, clip_eps, ppo_epochs, entropy_coef,
        )

    # ------------------------------------------------------------------
    # Interface parity with DQNAgent
    # ------------------------------------------------------------------

    def current_epsilon(self) -> float:
        """Logging-only stub -- see module docstring. PPO's exploration is
        the sampled stochastic policy itself, not an epsilon schedule."""
        return 0.0

    def start_round(self, global_state: Dict[str, "torch.Tensor"]) -> None:
        self.load_state_dict(global_state)

    def decay_lr(self) -> float:
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.lr_decay, self.min_lr)
            group["lr"] = new_lr
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state: Dict[str, "torch.Tensor"]) -> None:
        self.net.load_state_dict(state)

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)

    # ------------------------------------------------------------------
    # Action selection (batched across every intersection in a city, one
    # forward pass per tick -- same batching shape as DQNAgent.act_batch)
    # ------------------------------------------------------------------

    def _act_batch_with_logprob_value(
        self, obs_dict: Dict[str, Observation]
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        ts_ids = list(obs_dict.keys())
        obs_list = [obs_dict[t] for t in ts_ids]
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
            logits, value = self.net.forward_actor_critic(own, neighbors, neighbor_mask, hop_dist)
            logits = _mask_logits(logits, action_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        actions = {t: int(a) for t, a in zip(ts_ids, action.cpu().numpy())}
        log_probs = {t: float(lp) for t, lp in zip(ts_ids, log_prob.cpu().numpy())}
        values = {t: float(v) for t, v in zip(ts_ids, value.cpu().numpy())}
        return actions, log_probs, values

    def act(self, obs: Observation, explore: bool = True, eps: Optional[float] = None,
            ts_id: Optional[str] = None) -> int:
        """Single-observation action, matching DQNAgent.act()'s signature
        -- this is what federated/evaluator.py calls at eval time
        (``explore=False``). ``eps``/``ts_id`` are accepted only for
        interface parity and ignored (PPO has no epsilon schedule and no
        per-intersection state; ``ts_id`` exists so evaluator.py can pass
        it uniformly to every agent type, including RecurrentDQNAgent,
        which actually uses it).

        explore=True:  sample from the policy distribution (same
                        stochastic behavior as during training).
        explore=False: argmax of the (masked) policy logits -- deterministic
                        greedy, matching the pure-argmax-at-eval convention
                        DQNAgent uses (fidings/divergence_investigation.md
                        sec 35 confirmed this matches field/RESCO practice),
                        so eval numbers stay comparable across algorithms.
        """
        with torch.no_grad():
            own, neighbors, neighbor_mask, hop_dist, action_mask = _collate([obs], self.device)
            logits, _ = self.net.forward_actor_critic(own, neighbors, neighbor_mask, hop_dist)
            logits = _mask_logits(logits, action_mask)
            if explore:
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())
            return int(torch.argmax(logits, dim=-1).item())

    # ------------------------------------------------------------------
    # Training loop -- multi-agent aware, on-policy (collect a full
    # round's rollout across `episodes` episodes, then one PPO update)
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        episodes: int = 5,
        log_loss_every_steps: int = 50,
    ) -> Tuple[Dict[str, "torch.Tensor"], int, Optional[float], Dict[int, int]]:
        """Same contract as DQNAgent.train(): steps `env` for `episodes`
        episodes, using the SAME shared network to act for every
        intersection each tick, then returns
        (state_dict, total_steps, mean_loss, action_counts).

        Unlike DQNAgent (which does one off-policy gradient step per tick
        against a persistent replay buffer), this collects the whole
        round's on-policy rollout first and does the PPO update once at
        the end -- `log_loss_every_steps` is accepted for interface parity
        but unused (there's nothing to log mid-rollout; the loss only
        exists once the update runs).
        """
        total_steps = 0
        action_counts: Counter = Counter()
        buffer = RolloutBuffer()

        for ep in range(1, episodes + 1):
            obs_dict = env.reset()
            if not isinstance(obs_dict, dict):
                obs_dict = {"__single__": obs_dict}

            done = False
            ep_steps = 0
            while not done:
                actions, log_probs, values = self._act_batch_with_logprob_value(obs_dict)
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
                    if self.reward_clip is not None:
                        r = float(np.clip(r, -self.reward_clip, self.reward_clip))
                    d = dones.get(ts_id, dones.get("__all__", False))
                    buffer.add(ep, ts_id, o, actions[ts_id], log_probs[ts_id],
                               values[ts_id], r, d)

                obs_dict = next_obs_dict
                done = dones.get("__all__", all(dones.values()) if dones else True)
                ep_steps += 1
                total_steps += 1

            logger.info("[train] ep=%d/%d  steps=%d  (rollout collected, PPO update pending)",
                        ep, episodes, ep_steps)

        mean_loss = self._ppo_update(buffer)
        buffer.clear()
        return self.state_dict(), total_steps, mean_loss, dict(action_counts)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _compute_gae(self, traj: List[dict]) -> Tuple[List[float], List[float]]:
        """Standard GAE-lambda over one (episode, ts_id) trajectory.
        Every trajectory here ends at the env's own episode boundary (no
        mid-episode truncation in this environment), so the bootstrap
        value after the last step is always 0."""
        advantages = [0.0] * len(traj)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(traj))):
            r = traj[t]["reward"]
            v = traj[t]["value"]
            not_done = 0.0 if traj[t]["done"] else 1.0
            delta = r + self.gamma * next_value * not_done - v
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = v
        returns = [advantages[t] + traj[t]["value"] for t in range(len(traj))]
        return advantages, returns

    def _ppo_update(self, buffer: RolloutBuffer) -> Optional[float]:
        if len(buffer) == 0:
            return None

        obs_list, actions, old_log_probs, advantages, returns = [], [], [], [], []
        for traj in buffer.trajectories.values():
            adv, ret = self._compute_gae(traj)
            for step, a, r in zip(traj, adv, ret):
                obs_list.append(step["obs"])
                actions.append(step["action"])
                old_log_probs.append(step["log_prob"])
                advantages.append(a)
                returns.append(r)

        n = len(obs_list)
        own, neighbors, neighbor_mask, hop_dist, action_mask = _collate(obs_list, self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize advantages for update stability -- standard PPO practice,
        # keeps the surrogate objective's scale consistent round to round
        # regardless of this round's raw reward magnitude.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        losses = []
        idx = np.arange(n)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                logits, values = self.net.forward_actor_critic(
                    own[mb_t], neighbors[mb_t], neighbor_mask[mb_t], hop_dist[mb_t]
                )
                logits = _mask_logits(logits, action_mask[mb_t])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[mb_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_t])
                mb_adv = advantages_t[mb_t]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns_t[mb_t])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                self.optimizer.step()
                losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else None
