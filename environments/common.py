from typing import Any, Dict, Optional
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


class MockEnv:
    """A tiny deterministic mock environment for quick testing.

    Observations are fixed-size vectors; step dynamics are random.
    """

    def __init__(self, obs_dim: int = 4, action_dim: int = 2, seed: Optional[int] = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seed = seed or 0
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._t = 0

    def reset(self):
        self._t = 0
        return np.zeros(self.obs_dim, dtype=float)

    def step(self, action: int):
        self._t += 1
        obs = np.random.randn(self.obs_dim).astype(float)
        reward = float(random.random())
        done = self._t >= 20
        info = {}
        return obs, reward, done, info

    def close(self):
        return


def build_env_from_config(cfg: Dict[str, Any]):
    """Return an environment instance based on cfg.

    If SUMO/traci are available and cfg points to real files, this can be
    extended to return a SUMO environment. For now returns `MockEnv`.
    """
    return MockEnv(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2), seed=cfg.get("seed", None))
