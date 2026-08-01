"""Simulates unreliable inter-intersection communication during training.

Wraps a multi-agent city environment (one that returns ``Dict[ts_id, obs]``
from ``reset``/``step``) and randomly degrades each intersection's
``neighbor_mask`` (and zeroes the corresponding ``neighbors`` rows) so the
shared policy learns to be robust to:

  - individual link failures      (``p_link``, applied independently per
                                    neighbor slot, every tick)
  - a fully isolated intersection (``p_isolate``, drops *all* neighbors
                                    for that intersection this tick)
  - a cut relay beyond some hop   (``p_hop_cutoff``, picks a random cutoff
                                    hop and drops every neighbor farther
                                    away than that -- simulates a broken
                                    link somewhere upstream taking an
                                    entire branch of the graph offline)

Wrap the env once per city, at construction time in ``load_clients``:

    env = build_federated_env(cfg)
    env = CommDropoutWrapper(env, p_link=0.1, p_isolate=0.05, p_hop_cutoff=0.1)

Set all three probabilities to 0 to disable (e.g. for the holdout
evaluator, if you want to evaluate under "clean" communication) or leave
them on to evaluate robustness under realistic conditions too.
"""
from typing import Optional
import random
import numpy as np


class CommDropoutWrapper:
    """Wraps an environment so training sees lossy inter-intersection communication."""

    def __init__(
        self,
        env,
        p_link: float = 0.1,
        p_isolate: float = 0.05,
        p_hop_cutoff: float = 0.1,
        max_hop: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.p_link = p_link
        self.p_isolate = p_isolate
        self.p_hop_cutoff = p_hop_cutoff
        self.max_hop = max_hop
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def _corrupt_one(self, obs: dict) -> dict:
        mask = np.array(obs["neighbor_mask"], copy=True)
        hop = obs.get("hop_dist")

        if self.p_isolate and self._rng.random() < self.p_isolate:
            mask[:] = 0.0
        else:
            if self.p_hop_cutoff and hop is not None and self._rng.random() < self.p_hop_cutoff:
                hop_arr = np.array(hop)
                upper = self.max_hop or int(hop_arr.max()) if hop_arr.size else 1
                upper = max(upper, 1)
                cutoff = self._rng.randint(1, upper)
                mask[hop_arr > cutoff] = 0.0
            if self.p_link:
                drop = self._np_rng.random_sample(mask.shape) < self.p_link
                mask[drop] = 0.0

        obs["neighbor_mask"] = mask
        obs["neighbors"] = np.array(obs["neighbors"], copy=True) * mask[:, None]
        return obs

    def _corrupt(self, obs_dict: dict) -> dict:
        for ts_id, obs in obs_dict.items():
            obs_dict[ts_id] = self._corrupt_one(obs)
        return obs_dict

    def reset(self, *a, **kw):
        obs = self.env.reset(*a, **kw)
        if isinstance(obs, tuple):
            return self._corrupt(obs[0]), *obs[1:]
        return self._corrupt(obs)

    def step(self, actions):
        next_obs, rewards, dones, info = self.env.step(actions)
        return self._corrupt(next_obs), rewards, dones, info

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate everything else (own_dim, neighbor_dim, k_max,
        # max_action_dim, ts_ids, ...) to the wrapped env.
        return getattr(self.env, name)
