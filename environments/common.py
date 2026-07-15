from typing import Any, Dict, Optional, Tuple
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
            allowed_keys = {"out_csv_name", "use_gui", "virtual_display", "begin_time", 
                            "num_seconds", "max_depart_delay", "waiting_time_memory", 
                            "time_to_teleport", "delta_time", "yellow_time", "min_green", 
                            "max_green", "enforce_max_green", "single_agent", "reward_fn", 
                            "reward_weights", "observation_class", "add_system_info", "add_per_agent_info", 
                            "sumo_seed", "ts_ids", "fixed_ts", "sumo_warnings", "additional_sumo_cmd", "render_mode"}

            for k in allowed_keys:
                if k in cfg:
                    params[k] = cfg[k]

            env = SumoEnvironment(net_file=net, route_file=route, **params)
            return env
        except Exception as e:
            logger.exception("Failed to build SUMO environment: %s", e)
            raise

    return MockEnv(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2), seed=cfg.get("seed", None))


class LaneEncoder:
    """Canonical lane feature schema shared by every city."""

    FEATURES = [
        ("queue", lambda l, n: l.queue / n.max_queue),
        ("waiting_time", lambda l, n: l.waiting_time / n.max_wait),
        ("occupancy", lambda l, n: l.occupancy),
        ("speed", lambda l, n: l.speed / n.max_speed),
        ("is_left", lambda l, n: float(l.is_left)),
        ("is_straight", lambda l, n: float(l.is_straight)),
        ("is_right", lambda l, n: float(l.is_right)),
    ]

    GLOBAL_FEATURES = [
        ("current_phase", lambda phase, elapsed, yellow: phase / 16.0),
        ("elapsed_green", lambda phase, elapsed, yellow: elapsed / 120.0),
        ("yellow_time", lambda phase, elapsed, yellow: yellow / 10.0),
    ]


class LaneExtractor:
    """Override extract() for each city/environment."""

    def __init__(self, env):
        self.env = env

    def extract(self):
        raise NotImplementedError


class LaneNormalizer:
    def __init__(self, max_queue=50, max_wait=300,
                 max_speed=20.0, encoder=LaneEncoder):
        self.max_queue = max_queue
        self.max_wait = max_wait
        self.max_speed = max_speed
        self.encoder = encoder

    def normalize(self, lane):
        return np.asarray(
            [fn(lane, self) for _, fn in self.encoder.FEATURES],
            dtype=np.float32
        )


class LaneSorter:
    def __init__(self, key=None):
        self.key = key or (
            lambda l: (l.queue, l.waiting_time, l.occupancy)
        )

    def sort(self, lanes):
        return sorted(lanes, key=self.key, reverse=True)


class TopKEncoder:
    def __init__(self, normalizer, max_lanes=16):
        self.normalizer = normalizer
        self.max_lanes = max_lanes
        self.features_per_lane = len(
            self.normalizer.encoder.FEATURES
        )
        self.output_dim = (
            self.max_lanes * self.features_per_lane +
            len(self.normalizer.encoder.GLOBAL_FEATURES)
        )

    def encode(self, lanes, current_phase,
               elapsed_green, yellow_time=0):

        lanes = lanes[:self.max_lanes]
        features = []

        for lane in lanes:
            features.extend(self.normalizer.normalize(lane))

        while len(lanes) < self.max_lanes:
            features.extend(
                np.zeros(
                    self.features_per_lane,
                    dtype=np.float32
                )
            )
            lanes.append(None)

        for _, fn in self.normalizer.encoder.GLOBAL_FEATURES:
            features.append(
                fn(current_phase, elapsed_green, yellow_time)
            )

        return np.asarray(features, dtype=np.float32)


class ActionMapper:
    def __init__(self, mapping):
        self.mapping = mapping

    def map(self, action):
        return self.mapping.get(int(action), 0)


class FederatedWrapper:
    def __init__(self, env, extractor,
                 sorter, encoder, mapper):
        self.env = env
        self.extractor = extractor
        self.sorter = sorter
        self.encoder = encoder
        self.mapper = mapper

    def _state(self):
        lanes, phase, elapsed = self.extractor.extract()
        lanes = self.sorter.sort(lanes)
        return self.encoder.encode(lanes, phase, elapsed)

    def reset(self):
        if hasattr(self.env, "episode") and self.env.episode > 0:
            time.sleep(2)
        ret = self.env.reset()
        if isinstance(ret, tuple):
            ret = ret[0]
        return self._state()

    def step(self, action):
        local_action = self.mapper.map(action)
        _, reward, done, info = self.env.step(local_action)
        return self._state(), reward, done, info

    def close(self):
        self.env.close()

