from typing import Any, Dict, Optional
import logging
import random
import os
import sys

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


def _ensure_sumo_tools_on_path():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def build_env_from_config(cfg: Dict[str, Any]):
    """Return an environment instance based on cfg.

    If cfg specifies SUMO files (mode: 'sumo' or contains 'net_file'/'route_file'),
    instantiate the repository's `sumo_rl.environment.env.SumoEnvironment`.
    Otherwise returns `MockEnv` for quick tests.
    """
    mode = cfg.get("mode", "mock")
    if mode == "sumo" or ("net_file" in cfg and "route_file" in cfg):
        try:
            _ensure_sumo_tools_on_path()
            from sumo_rl.environment.env import SumoEnvironment

            net = cfg["net_file"]
            route = cfg["route_file"]
            params = {}
            # pass through commonly used SUMO env parameters
            for k in ["use_gui", "delta_time", "min_green", "yellow_time", "num_seconds", "sumo_seed", "single_agent", "reward_fn"]:
                if k in cfg:
                    params[k] = cfg[k]

            env = SumoEnvironment(net_file=net, route_file=route, **params)
            return env
        except Exception as e:
            logger.exception("Failed to build SUMO environment: %s", e)
            raise

    return MockEnv(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2), seed=cfg.get("seed", None))

import time

class PaddingWrapper:
    def __init__(self, env, target_obs_dim: int, target_action_n: int):
        self.env = env
        self.target_obs_dim = target_obs_dim
        self.target_action_n = target_action_n
        self._orig_action_n = None

    def reset(self):
        # pe WSL, SUMO are nevoie de timp sa elibereze portul intre episoade
        if hasattr(self.env, 'episode') and self.env.episode > 0:
            time.sleep(3)

        reset_ret = self.env.reset()

        try:
            self._orig_action_n = self.env.action_space.n
        except Exception:
            self._orig_action_n = self.target_action_n

        if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
            obs = reset_ret[0]
        else:
            obs = reset_ret
        return self._pad_obs(obs)

    def step(self, action):
        orig_n = self._orig_action_n
        if orig_n is not None and action >= orig_n:
            mapped = int(orig_n - 1)
        else:
            mapped = int(action)
        next_obs, reward, done, info = self.env.step(mapped)
        return self._pad_obs(next_obs), reward, done, info

    def close(self):
        return self.env.close()

    def _pad_obs(self, obs):
        try:
            arr = np.array(obs, dtype=float)
            if arr.ndim == 1 and arr.dtype != object:
                flat = arr
            else:
                raise Exception()
        except Exception:
            pieces = []
            for el in obs:
                try:
                    if isinstance(el, dict):
                        sub = []
                        for v in el.values():
                            try:
                                sub.append(np.asarray(v, dtype=float).ravel())
                            except Exception:
                                sub.append(np.asarray([float(v)]))
                        a = np.concatenate(sub) if sub else np.zeros(0, dtype=float)
                    else:
                        try:
                            a = np.asarray(el, dtype=float).ravel()
                        except Exception:
                            a = np.asarray([float(el)])
                except Exception:
                    a = np.zeros(0, dtype=float)
                pieces.append(a)
            flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)

        if flat.shape[0] >= self.target_obs_dim:
            return flat[: self.target_obs_dim]
        pad = np.zeros(self.target_obs_dim - flat.shape[0], dtype=float)
        return np.concatenate([flat, pad])