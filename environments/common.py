from typing import Any, Dict, Optional
import logging
import random
import os
import sys
import time

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


def find_active_traffic_signal(cfg: Dict[str, Any], probe_steps: int = 20) -> Optional[str]:
    """Probe a city config in multi-agent mode and return the ts_id with the most traffic.

    This solves a generic problem: SumoEnvironment with single_agent=True
    controls only ts_ids[0] by default. For networks with multiple traffic
    signals, ts_ids[0] is arbitrary and may have zero traffic passing
    through it, silently breaking training/evaluation (reward always 0)
    without any error. This probes all signals and picks the one that
    actually sees traffic.

    Args:
        cfg: city config dict (as loaded from config.yaml).
        probe_steps: how many simulation steps to run while probing.

    Returns:
        The ts_id (str) with the highest cumulative |reward|, or None if no
        traffic signal saw any meaningful traffic.
    """
    probe_cfg = dict(cfg)
    probe_cfg["single_agent"] = False
    probe_cfg.pop("ts_ids", None)

    env = build_env_from_config(probe_cfg, validate=False)
    try:
        env.reset()
        ts_ids = list(env.ts_ids)
        if not ts_ids:
            return None

        cumulative_abs_reward = {ts: 0.0 for ts in ts_ids}

        for _ in range(probe_steps):
            actions = {ts: 0 for ts in ts_ids}
            _, rewards, dones, _ = env.step(actions)
            for ts, r in rewards.items():
                cumulative_abs_reward[ts] += abs(float(r))
            if dones.get("__all__"):
                break

        best_ts = max(cumulative_abs_reward, key=cumulative_abs_reward.get)
        best_score = cumulative_abs_reward[best_ts]

        logger.info(
            "Probed %d traffic signals for '%s', best=%s (cumulative |reward|=%.4f)",
            len(ts_ids), cfg.get("name", "?"), best_ts, best_score
        )

        if best_score <= 0.0:
            logger.warning(
                "No traffic signal showed any reward signal during probing "
                "(%d steps) for config '%s'.", probe_steps, cfg.get("name", "?")
            )
            return None

        return best_ts
    finally:
        try:
            env.close()
        except Exception:
            pass
        time.sleep(2)


def validate_and_patch_config(cfg: Dict[str, Any], probe_steps: int = 20) -> Dict[str, Any]:
    """Return a copy of cfg with ts_ids explicitly set to the active traffic signal.

    If cfg already specifies ts_ids, returns it unchanged. Otherwise probes
    the network and pins ts_ids to the traffic signal with real traffic.

    Raises:
        RuntimeError: if no traffic signal in the network shows any traffic
            during probing. This means the config is broken (e.g. flow
            routes don't pass through any signalized intersection) and
            should be fixed rather than silently trained/evaluated with
            reward=0.
    """
    if cfg.get("ts_ids"):
        return cfg

    if not cfg.get("single_agent", False):
        # multi-agent mode controls all signals anyway, nothing to patch
        return cfg

    chosen = find_active_traffic_signal(cfg, probe_steps=probe_steps)
    if chosen is None:
        raise RuntimeError(
            f"Config '{cfg.get('name', '?')}' has no traffic signal with "
            f"observable traffic (probed {probe_steps} steps, all rewards "
            f"were 0). Check that the route file's flows actually pass "
            f"through a signalized intersection, or set single_agent=False."
        )

    patched = dict(cfg)
    patched["ts_ids"] = [chosen]
    logger.info("Config '%s': pinned single-agent control to ts_id='%s'", cfg.get("name", "?"), chosen)
    return patched


def build_env_from_config(cfg: Dict[str, Any], validate: bool = True, probe_steps: int = 20):
    """Return an environment instance based on cfg.

    If cfg specifies SUMO files (mode: 'sumo' or contains 'net_file'/'route_file'),
    instantiate the repository's `sumo_rl.environment.env.SumoEnvironment`.
    Otherwise returns `MockEnv` for quick tests.

    Args:
        cfg: city config dict.
        validate: if True and single_agent=True and ts_ids not explicitly set,
            probes the network to find a traffic signal with real traffic and
            pins ts_ids to it. Prevents silently training/evaluating on a
            dead signal (reward always 0). Set to False internally during
            the probe itself to avoid infinite recursion.
        probe_steps: number of steps used during the validation probe.
    """
    mode = cfg.get("mode", "mock")
    if mode == "sumo" or ("net_file" in cfg and "route_file" in cfg):
        try:
            _ensure_sumo_tools_on_path()
            from sumo_rl.environment.env import SumoEnvironment

            if validate and cfg.get("single_agent", False) and not cfg.get("ts_ids"):
                cfg = validate_and_patch_config(cfg, probe_steps=probe_steps)

            net = cfg["net_file"]
            route = cfg["route_file"]
            params = {}
            # pass through commonly used SUMO env parameters
            for k in ["use_gui", "delta_time", "min_green", "yellow_time", "num_seconds", "sumo_seed", "single_agent", "reward_fn", "ts_ids"]:
                if k in cfg:
                    params[k] = cfg[k]

            env = SumoEnvironment(net_file=net, route_file=route, **params)
            return env
        except Exception as e:
            logger.exception("Failed to build SUMO environment: %s", e)
            raise

    return MockEnv(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2), seed=cfg.get("seed", None))


class PaddingWrapper:
    """Wrap an environment to pad observations to a fixed size and unify action space."""

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