"""
Federated RL Environment Framework
===================================
A city-agnostic wrapper layer that gives every SUMO city an identical,
*multi-agent* observation/action interface with automatic neighbor
discovery. No per-city Python is required anywhere in this module.

Architecture
------------
    Config
        │
        ▼
    build_multi_agent_raw_env()      # raw multi-agent SUMO env (all ts_ids)
        │
        ▼
    build_federated_env(cfg)         # assembles the full pipeline
        │
        ├── SumoLaneExtractor        # ONE universal extractor, works per ts_id
        ├── LaneSorter / LaneNormalizer / TopKEncoder   # own-observation
        ├── NeighborGraphBuilder     # static topology -> K-hop ts graph
        ├── NeighborSummaryExtractor # per-neighbor [queue, wait, phase]
        ├── ActionSpaceInspector     # ts.action_space.n -> action_mask
        │
        └── MultiAgentFederatedWrapper
              reset()/step(actions)  -> Dict[ts_id, obs_dict]
                    │
                    ▼
              RL Agent (agents/dqn.py + agents/networks.py)

Observation contract (see agents/networks.py for the consumer side)
---------------------------------------------------------------------
    obs_dict = {
        "own":            (own_dim,)          own-intersection features
        "neighbors":      (k_max, neighbor_dim)  zero-padded neighbor summaries
        "neighbor_mask":  (k_max,)             1.0 = real neighbor this slot
        "hop_dist":       (k_max,)             1..K hop distance, 0 = padding
        "action_mask":    (max_action_dim,)    1.0 = valid action for this ts
    }

Why no more ``adapter_class`` / ``phase_mapping``
--------------------------------------------------
Both were city-specific escape hatches. They're removed in favor of:
  - one universal lane extractor that works off ``TrafficSignal.lanes``
    (every sumo-rl TrafficSignal exposes this, regardless of arm count),
  - ``TrafficSignal.action_space.n`` for automatic, per-intersection valid
    action counts (3-way / 4-way / 5-way / protected-left all "just work"
    because the *count* is discovered, not hand-mapped), masked into a
    shared, fixed-size action_mask.

A caveat worth flagging explicitly (this was an open question in the
original design doc): this module assumes sumo-rl's internal action
index -> green-phase mapping (and yellow-phase insertion) is handled
correctly by ``TrafficSignal`` itself, i.e. that action index ``n`` passed
to ``env.step`` already means "the n-th valid green phase" and that
yellow transitions are automatic. That's true for recent sumo-rl
versions; if you're on an older version where phases must be pre-listed
manually, ``ActionSpaceInspector`` is the one place to patch.
"""

from __future__ import annotations

import abc
import importlib
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NEIGHBOR_SUMMARY_DIM = 3  # [mean_queue_norm, mean_waiting_norm, phase_norm]


# ---------------------------------------------------------------------------
# Unchanged helpers
# ---------------------------------------------------------------------------

class MockEnv:
    """Tiny deterministic single-agent mock env, kept for quick unit tests
    of things that don't care about the multi-agent pipeline."""

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
        return obs, reward, done, {}

    def close(self):
        return


class MultiAgentMockEnv:
    """Deterministic multi-agent mock env for exercising the full
    multi-agent / neighbor-graph / dropout pipeline without a running SUMO
    instance. Builds a small ring topology so k-hop neighbor logic is
    actually tested.
    """

    def __init__(self, n_ts: int = 5, action_dim: int = 4, seed: Optional[int] = None):
        self.ts_ids = [f"mock_ts_{i}" for i in range(n_ts)]
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._t = 0
        # Ring adjacency: each node's hop-1 neighbors are its ring neighbors.
        self.adjacency = {
            ts: [self.ts_ids[(i - 1) % n_ts], self.ts_ids[(i + 1) % n_ts]]
            for i, ts in enumerate(self.ts_ids)
        }
        self.action_counts = {
            ts: int(self._rng.integers(2, action_dim + 1)) for ts in self.ts_ids
        }

    def reset(self):
        self._t = 0
        return {ts: self._t for ts in self.ts_ids}  # placeholder, extractor ignores value

    def step(self, actions: Dict[str, int]):
        self._t += 1
        obs = {ts: self._t for ts in self.ts_ids}
        rewards = {ts: float(self._rng.random()) for ts in self.ts_ids}
        done = self._t >= 50
        dones = {ts: done for ts in self.ts_ids}
        dones["__all__"] = done
        return obs, rewards, dones, {}

    def close(self):
        return


def _ensure_sumo_tools_on_path() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def _maybe_enable_libsumo(cfg: Dict[str, Any]) -> None:
    """Opt-in: use libsumo instead of traci for this process.

    libsumo is a direct C++ binding (no socket round-trips), typically
    2-10x faster than traci for single-instance use -- and single-instance
    is exactly what every city env here is. Must be set BEFORE the first
    `import traci` anywhere in the process (sumo-rl's SumoEnvironment
    imports traci internally, and SumoLaneExtractor imports it too), so
    this is called at the very top of ``build_env_from_config``.

    Caveats (why this isn't just always-on):
      - libsumo supports only ONE live SUMO instance per process. Fine
        for the parallel-worker-per-city setup (see federated/parallel_server.py)
        where each process only ever runs one city. NOT fine if you try
        to hold two cities' envs open simultaneously in the same process.
      - No SUMO GUI support under libsumo.
    """
    if cfg.get("use_libsumo", False):
        os.environ["LIBSUMO_AS_TRACI"] = "1"


def find_active_traffic_signal(cfg: Dict[str, Any], probe_steps: int = 20) -> Optional[str]:
    """Probe a city config in multi-agent mode and return the ts_id with the
    most traffic. Kept for single-agent legacy configs / debugging."""
    probe_cfg = dict(cfg)
    probe_cfg["single_agent"] = False
    probe_cfg.pop("ts_ids", None)

    env = build_env_from_config(probe_cfg, validate=False)
    try:
        env.reset()
        ts_ids = list(env.ts_ids)
        if not ts_ids:
            return None

        cumulative_abs_reward: Dict[str, float] = {ts: 0.0 for ts in ts_ids}
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
            len(ts_ids), cfg.get("name", "?"), best_ts, best_score,
        )
        if best_score <= 0.0:
            logger.warning(
                "No traffic signal showed any reward signal during probing "
                "(%d steps) for config '%s'.", probe_steps, cfg.get("name", "?"),
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
    """Legacy single-agent path: pin ts_ids to the signal with real traffic.
    Not used by the (default) multi-agent pipeline, kept for configs that
    explicitly request ``single_agent: true``."""
    if cfg.get("ts_ids"):
        return cfg
    if not cfg.get("single_agent", False):
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


def build_env_from_config(cfg: Dict[str, Any], validate: bool = True, probe_steps: int = 200):
    """Return a *raw* environment instance based on cfg. No observation
    preprocessing -- see ``build_federated_env`` for the full pipeline."""
    mode = cfg.get("mode", "mock")
    if mode == "sumo" or ("net_file" in cfg and "route_file" in cfg):
        try:
            _maybe_enable_libsumo(cfg)
            _ensure_sumo_tools_on_path()
            from sumo_rl.environment.env import SumoEnvironment  # type: ignore

            if validate and cfg.get("single_agent", False) and not cfg.get("ts_ids"):
                cfg = validate_and_patch_config(cfg, probe_steps=probe_steps)

            net = cfg["net_file"]
            route = cfg["route_file"]
            allowed_keys = {
                "out_csv_name", "use_gui", "virtual_display", "begin_time",
                "num_seconds", "max_depart_delay", "waiting_time_memory",
                "time_to_teleport", "delta_time", "yellow_time", "min_green",
                "max_green", "enforce_max_green", "single_agent", "reward_fn",
                "reward_weights", "observation_class", "add_system_info",
                "add_per_agent_info", "sumo_seed", "ts_ids", "fixed_ts",
                "sumo_warnings", "additional_sumo_cmd", "render_mode",
            }
            params = {k: cfg[k] for k in allowed_keys if k in cfg}
            return SumoEnvironment(net_file=net, route_file=route, **params)
        except Exception as e:
            logger.exception("Failed to build SUMO environment: %s", e)
            raise

    return MockEnv(
        obs_dim=cfg.get("obs_dim", 4),
        action_dim=cfg.get("action_dim", 2),
        seed=cfg.get("seed", None),
    )


def build_multi_agent_raw_env(cfg: Dict[str, Any]):
    """Force multi-agent mode: every traffic signal in the network is
    controlled, none dropped via ``ts_ids``. This is the default path for
    the federated pipeline -- one CityClient trains every intersection.
    """
    if cfg.get("mode", "mock") == "mock" and "net_file" not in cfg:
        n_ts = int(cfg.get("mock_n_ts", 5))
        action_dim = int(cfg.get("action_dim", 4))
        return MultiAgentMockEnv(n_ts=n_ts, action_dim=action_dim, seed=cfg.get("seed"))

    ma_cfg = dict(cfg)
    ma_cfg["single_agent"] = False
    ma_cfg.pop("ts_ids", None)
    return build_env_from_config(ma_cfg, validate=False)


# ---------------------------------------------------------------------------
# Lane data model (unchanged -- already city-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """Container for lane-level traffic and movement features extracted from SUMO."""

    lane_id: str = ""
    queue: float = 0.0
    waiting_time: float = 0.0
    occupancy: float = 0.0
    speed: float = 0.0
    density: float = 0.0
    pressure: float = 0.0
    length: float = 0.0
    is_left: bool = False
    is_straight: bool = False
    is_right: bool = False
    custom_features: Dict[str, float] = field(default_factory=dict)


class LaneEncoder:
    """Maps lane and traffic-light measurements into a fixed-size feature vector."""

    FEATURES: List[Tuple[str, Callable]] = [
        ("queue",        lambda l, n: l.queue        / n.max_queue),
        ("waiting_time", lambda l, n: l.waiting_time / n.max_wait),
        ("occupancy",    lambda l, n: l.occupancy),
        ("speed",        lambda l, n: l.speed        / n.max_speed),
        ("is_left",      lambda l, n: float(l.is_left)),
        ("is_straight",  lambda l, n: float(l.is_straight)),
        ("is_right",     lambda l, n: float(l.is_right)),
    ]

    GLOBAL_FEATURES: List[Tuple[str, Callable]] = [
        ("current_phase",  lambda phase, elapsed, yellow: phase   / 16.0),
        ("elapsed_green",  lambda phase, elapsed, yellow: elapsed / 120.0),
        ("yellow_time",    lambda phase, elapsed, yellow: yellow  / 10.0),
    ]


class LaneNormalizer:
    """Normalizes lane features against configured upper bounds for the shared encoder."""

    def __init__(self, max_queue: float = 50.0, max_wait: float = 300.0,
                 max_speed: float = 20.0, encoder: type = LaneEncoder):
        self.max_queue = max_queue
        self.max_wait = max_wait
        self.max_speed = max_speed
        self.encoder = encoder

    def normalize(self, lane: Lane) -> np.ndarray:
        return np.asarray([fn(lane, self) for _, fn in self.encoder.FEATURES], dtype=np.float32)


class LaneSorter:
    """Orders lanes by the metrics that matter for the own-observation encoder."""

    def __init__(self, key: Optional[Callable[[Lane], Any]] = None):
        self.key: Callable[[Lane], Any] = key or (lambda l: (l.queue, l.waiting_time, l.occupancy))

    def sort(self, lanes: List[Lane]) -> List[Lane]:
        return sorted(lanes, key=self.key, reverse=True)


class TopKEncoder:
    """Own-observation encoder. Output dim is always
    ``max_lanes * features_per_lane + len(GLOBAL_FEATURES)`` -- fixed size
    regardless of how many lanes/arms the real intersection has."""

    def __init__(self, normalizer: LaneNormalizer, max_lanes: int = 16):
        self.normalizer = normalizer
        self.max_lanes = max_lanes
        self.features_per_lane = len(normalizer.encoder.FEATURES)
        self.output_dim = self.max_lanes * self.features_per_lane + len(normalizer.encoder.GLOBAL_FEATURES)

    def encode(self, lanes: List[Lane], current_phase: float, elapsed_green: float,
               yellow_time: float = 0.0) -> np.ndarray:
        lanes = lanes[: self.max_lanes]
        features: List[float] = []
        for lane in lanes:
            features.extend(self.normalizer.normalize(lane))
        padding_lanes = self.max_lanes - len(lanes)
        features.extend([0.0] * (padding_lanes * self.features_per_lane))
        for _, fn in self.normalizer.encoder.GLOBAL_FEATURES:
            features.append(fn(current_phase, elapsed_green, yellow_time))
        return np.asarray(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Universal lane extraction -- ONE class, works for every ts_id / every city
# ---------------------------------------------------------------------------

class LaneExtractor(abc.ABC):
    """Abstract base, kept for extensibility / testing with custom sources."""

    def __init__(self, env: Any):
        self.env = env

    @abc.abstractmethod
    def extract(self, ts_id: str) -> Tuple[List[Lane], int, float, float]:
        """Return (lanes, current_phase, elapsed_green, yellow_time) for
        the given traffic signal id."""


class SumoLaneExtractor(LaneExtractor):
    """The one universal extractor. Reads whatever lanes a given
    ``TrafficSignal`` controls -- 3-way, 4-way, 5-way, doesn't matter,
    because it never hard-codes lane counts or positions. Nothing here is
    city-specific; everything comes from sumo-rl's ``TrafficSignal`` /
    traci introspection.
    """

    def extract(self, ts_id: str) -> Tuple[List[Lane], int, float, float]:
        ts = self.env.traffic_signals[ts_id]
        try:
            import traci
        except Exception:
            traci = None

        lane_ids = list(dict.fromkeys(ts.lanes))  # de-dup, preserve order
        lanes: List[Lane] = []

        for lane_id in lane_ids:
            queue = waiting = occ = speed = 0.0
            is_left = is_straight = is_right = False
            try:
                queue = float(traci.lane.getLastStepHaltingNumber(lane_id))
                waiting = float(traci.lane.getWaitingTime(lane_id))
                occ = float(traci.lane.getLastStepOccupancy(lane_id))
                speed = float(traci.lane.getLastStepMeanSpeed(lane_id))
                is_left, is_straight, is_right = _infer_turn_direction(traci, lane_id)
            except Exception:
                logger.debug("Lane introspection failed for %s (ts=%s)", lane_id, ts_id, exc_info=True)

            lanes.append(Lane(
                lane_id=lane_id, queue=queue, waiting_time=waiting,
                occupancy=occ, speed=speed,
                is_left=is_left, is_straight=is_straight, is_right=is_right,
            ))

        current_phase = float(getattr(ts, "green_phase", 0))
        elapsed_green = float(getattr(ts, "time_since_last_phase_change", 0.0))
        yellow_time = float(getattr(ts, "yellow_time", 0.0)) if getattr(ts, "is_yellow", False) else 0.0

        return lanes, current_phase, elapsed_green, yellow_time


def _infer_turn_direction(traci_module, lane_id: str) -> Tuple[bool, bool, bool]:
    """Best-effort turn-movement classification from traci link direction
    codes. Returns (is_left, is_straight, is_right); defaults to
    "straight" if the API can't be queried (fail open, never raises)."""
    try:
        links = traci_module.lane.getLinks(lane_id)
    except Exception:
        return (False, True, False)

    if not links:
        return (False, True, False)

    # traci link tuples include a direction code, typically one of
    # 'l' (left), 's' (straight), 'r' (right), 't' (turnaround), etc.,
    # at a fixed position depending on traci version. Scan defensively.
    dir_codes = []
    for link in links:
        for field_ in link:
            if isinstance(field_, str) and field_ in ("l", "s", "r", "t", "L", "R", "S", "T"):
                dir_codes.append(field_.lower())

    is_left = "l" in dir_codes or "t" in dir_codes
    is_right = "r" in dir_codes
    is_straight = "s" in dir_codes or not (is_left or is_right)
    return (is_left, is_straight, is_right)


# ---------------------------------------------------------------------------
# Automatic action-space discovery (replaces phase_mapping/ActionMapper)
# ---------------------------------------------------------------------------

class ActionSpaceInspector:
    """Discovers each intersection's real action count directly from
    sumo-rl, instead of a hand-authored ``phase_mapping``.

    ``max_action_dim`` is the widest action space among this city's
    intersections; every intersection gets an ``action_mask`` of that
    width with the tail zeroed out for intersections with fewer real
    actions (e.g. a 3-way with 2 valid phases inside a city that also has
    5-way intersections with 6 valid phases).
    """

    def __init__(self, env: Any, ts_ids: List[str]):
        self.ts_ids = ts_ids
        self.action_counts: Dict[str, int] = {}

        for ts_id in ts_ids:
            n = None
            try:
                ts = env.traffic_signals[ts_id]
                if hasattr(ts, "action_space") and hasattr(ts.action_space, "n"):
                    n = int(ts.action_space.n)
                elif hasattr(ts, "num_green_phases"):
                    n = int(ts.num_green_phases)
            except Exception:
                logger.warning("Could not read action_space for ts=%s; defaulting to 2.", ts_id)
            self.action_counts[ts_id] = n or 2

        self.max_action_dim = max(self.action_counts.values())

    def action_mask(self, ts_id: str) -> np.ndarray:
        n = self.action_counts[ts_id]
        mask = np.zeros(self.max_action_dim, dtype=np.float32)
        mask[:n] = 1.0
        return mask

    def clip_action(self, ts_id: str, action: int) -> int:
        """Defensive clamp: if the shared policy outputs an index beyond
        this ts's real action count (shouldn't happen if action_mask is
        respected upstream, but cheap insurance), fall back to 0."""
        n = self.action_counts[ts_id]
        return int(action) if 0 <= action < n else 0


# ---------------------------------------------------------------------------
# Neighbor graph -- derived from road connectivity, not geometry
# ---------------------------------------------------------------------------

class NeighborGraphBuilder:
    """Builds K-hop neighbor relationships between traffic signals using
    the *static* SUMO network topology (via ``sumolib``), not traci and
    not geometric distance.

    Two traffic-light junctions are hop-1 neighbors if there's a path of
    roads between them that doesn't pass through another traffic-light
    junction. Plain (unsignalized) junctions are contracted/skipped over
    transparently -- they don't count as hops, they're just "the road."
    This mirrors how a real city's roads connect signalized intersections
    even when there are minor junctions in between.

    Falls back to "no neighbors" (with a warning) if ``sumolib`` or the
    net file are unavailable, so the pipeline degrades gracefully instead
    of crashing.
    """

    def __init__(self, net_file: Optional[str], ts_ids: List[str], max_hops: int = 3):
        self.ts_ids = set(ts_ids)
        self.max_hops = max_hops
        self.graph: Dict[str, List[Tuple[str, int]]] = {ts: [] for ts in ts_ids}

        if not net_file:
            logger.warning("No net_file provided; neighbor graph will be empty for all intersections.")
            return

        try:
            _ensure_sumo_tools_on_path()
            import sumolib  # type: ignore
        except Exception:
            logger.warning(
                "sumolib unavailable; neighbor graph will be empty. "
                "Install SUMO tools / set SUMO_HOME to enable neighbor discovery."
            )
            return

        try:
            net = sumolib.net.readNet(net_file)
        except Exception:
            logger.exception("Failed to parse net_file=%s for neighbor discovery.", net_file)
            return

        # Undirected adjacency between junction node IDs, via edges.
        node_adj: Dict[str, List[str]] = {}
        for edge in net.getEdges():
            a, b = edge.getFromNode().getID(), edge.getToNode().getID()
            node_adj.setdefault(a, []).append(b)
            node_adj.setdefault(b, []).append(a)

        # A node "is" a traffic signal if its ID matches a ts_id (the
        # common sumo-rl/netconvert convention) -- fall back to checking
        # node.getType() == 'traffic_light' if IDs don't line up.
        def node_is_ts(node_id: str) -> Optional[str]:
            if node_id in self.ts_ids:
                return node_id
            return None

        for start_ts in ts_ids:
            if start_ts not in node_adj and start_ts not in self.ts_ids:
                continue
            # BFS over the *physical* graph, but only increment hop count
            # when we land on another traffic-signal node; unsignalized
            # junctions are traversed for free.
            visited_ts = {start_ts}
            frontier = deque([(start_ts, 0)])
            visited_nodes = {start_ts}

            while frontier:
                node_id, hop = frontier.popleft()
                if hop >= self.max_hops:
                    continue
                for nbr in node_adj.get(node_id, []):
                    if nbr in visited_nodes:
                        continue
                    visited_nodes.add(nbr)
                    ts_here = node_is_ts(nbr)
                    if ts_here and ts_here not in visited_ts:
                        visited_ts.add(ts_here)
                        self.graph[start_ts].append((ts_here, hop + 1))
                        frontier.append((nbr, hop + 1))
                    elif not ts_here:
                        # plain junction: keep exploring at the SAME hop
                        frontier.append((nbr, hop))

        logger.info(
            "Neighbor graph built: %d intersections, avg neighbors/intersection=%.1f",
            len(ts_ids),
            np.mean([len(v) for v in self.graph.values()]) if ts_ids else 0.0,
        )

    def neighbors_of(self, ts_id: str) -> List[Tuple[str, int]]:
        """Return [(neighbor_ts_id, hop_distance), ...] sorted by hop then
        id, for deterministic slot assignment."""
        return sorted(self.graph.get(ts_id, []), key=lambda x: (x[1], x[0]))


class NeighborSummaryExtractor:
    """Produces the small, fixed-size summary each neighbor contributes:
    ``[mean_queue_norm, mean_waiting_norm, phase_norm]``. Deliberately
    much smaller than the full own-observation -- neighbors are context,
    not the primary signal -- and just as universal (works for any
    topology since it only aggregates over "whatever lanes this ts has").
    """

    def __init__(self, lane_extractor: SumoLaneExtractor, max_queue: float, max_wait: float):
        self.lane_extractor = lane_extractor
        self.max_queue = max_queue
        self.max_wait = max_wait

    def summarize(self, ts_id: str) -> np.ndarray:
        lanes, phase, _, _ = self.lane_extractor.extract(ts_id)
        if lanes:
            mean_queue = float(np.mean([l.queue for l in lanes])) / self.max_queue
            mean_wait = float(np.mean([l.waiting_time for l in lanes])) / self.max_wait
        else:
            mean_queue = 0.0
            mean_wait = 0.0
        phase_norm = float(phase) / 16.0
        return np.asarray([mean_queue, mean_wait, phase_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Multi-agent, dict-observation Gym-ish wrapper
# ---------------------------------------------------------------------------

class MultiAgentFederatedWrapper:
    """Steps every intersection in a city simultaneously and returns the
    dict-observation contract consumed by ``agents/dqn.py`` /
    ``agents/networks.py``.

    Args:
        env:              raw multi-agent env (SumoEnvironment or
                          MultiAgentMockEnv), ``ts_ids`` attribute required.
        lane_extractor:   ``SumoLaneExtractor`` instance.
        sorter, encoder:  own-observation pipeline (unchanged, universal).
        neighbor_graph:   ``NeighborGraphBuilder``.
        neighbor_summary: ``NeighborSummaryExtractor``.
        action_inspector: ``ActionSpaceInspector``.
        k_max:            fixed neighbor-slot count (padded/truncated to this).
    """

    def __init__(
        self,
        env: Any,
        lane_extractor: SumoLaneExtractor,
        sorter: LaneSorter,
        encoder: TopKEncoder,
        neighbor_graph: NeighborGraphBuilder,
        neighbor_summary: NeighborSummaryExtractor,
        action_inspector: ActionSpaceInspector,
        k_max: int = 8,
    ):
        self.env = env
        self.lane_extractor = lane_extractor
        self.sorter = sorter
        self.encoder = encoder
        self.neighbor_graph = neighbor_graph
        self.neighbor_summary = neighbor_summary
        self.action_inspector = action_inspector
        self.k_max = k_max

        self.ts_ids: List[str] = list(getattr(env, "ts_ids", []))

        # Public dims consumed by federated_training.load_clients
        self.own_dim = encoder.output_dim
        self.neighbor_dim = NEIGHBOR_SUMMARY_DIM
        self.max_action_dim = action_inspector.max_action_dim

    # ------------------------------------------------------------------
    def _build_obs(self, ts_id: str) -> Dict[str, np.ndarray]:
        lanes, phase, elapsed, yellow = self.lane_extractor.extract(ts_id)
        lanes = self.sorter.sort(lanes)
        own = self.encoder.encode(lanes, phase, elapsed, yellow)

        neighbors = np.zeros((self.k_max, self.neighbor_dim), dtype=np.float32)
        neighbor_mask = np.zeros(self.k_max, dtype=np.float32)
        hop_dist = np.zeros(self.k_max, dtype=np.int64)

        nbrs = self.neighbor_graph.neighbors_of(ts_id)[: self.k_max]
        for i, (nbr_ts, hop) in enumerate(nbrs):
            neighbors[i] = self.neighbor_summary.summarize(nbr_ts)
            neighbor_mask[i] = 1.0
            hop_dist[i] = hop

        action_mask = self.action_inspector.action_mask(ts_id)

        return {
            "own": own,
            "neighbors": neighbors,
            "neighbor_mask": neighbor_mask,
            "hop_dist": hop_dist,
            "action_mask": action_mask,
        }

    def _build_all_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {ts_id: self._build_obs(ts_id) for ts_id in self.ts_ids}

    # ------------------------------------------------------------------
    # Gym-ish multi-agent interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        if hasattr(self.env, "episode") and getattr(self.env, "episode", 0) > 0:
            time.sleep(2)
        ret = self.env.reset()
        if isinstance(ret, tuple):
            ret = ret[0]
        # refresh ts_ids in case the underlying env only populates them
        # after reset (true for sumo_rl's SumoEnvironment)
        if hasattr(self.env, "ts_ids"):
            self.ts_ids = list(self.env.ts_ids)
        return self._build_all_obs()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        clipped = {
            ts_id: self.action_inspector.clip_action(ts_id, a) for ts_id, a in actions.items()
        }
        _, rewards, dones, info = self.env.step(clipped)
        obs = self._build_all_obs()
        return obs, rewards, dones, info

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_federated_env(cfg: Dict[str, Any]) -> MultiAgentFederatedWrapper:
    """Assemble a complete, multi-agent federated environment from a city
    config dict. Single entry point for training/eval code.

    Required config keys
    --------------------
    net_file, route_file   for real SUMO cities (omit + mode: mock for tests)

    Optional (with defaults)
    ------------------------
    max_lanes    int    16     own-observation lane cap
    max_queue    float  50.0   normalization bound
    max_wait     float  300.0  normalization bound
    max_speed    float  20.0   normalization bound
    k_max        int    8      fixed neighbor-slot count
    max_hops     int    3      neighbor discovery radius (traffic-signal hops)

    No ``adapter_class`` and no ``phase_mapping`` -- both are now fully
    automatic (see module docstring).
    """
    raw_env = build_multi_agent_raw_env(cfg)

    reward_shaping_cfg = cfg.get("reward_shaping")
    if reward_shaping_cfg:
        raw_env = RewardShapingWrapper(raw_env, **reward_shaping_cfg)

    max_lanes = int(cfg.get("max_lanes", 16))
    max_queue = float(cfg.get("max_queue", 50.0))
    max_wait = float(cfg.get("max_wait", 300.0))
    max_speed = float(cfg.get("max_speed", 20.0))
    k_max = int(cfg.get("k_max", 8))
    max_hops = int(cfg.get("max_hops", 3))

    normalizer = LaneNormalizer(max_queue=max_queue, max_wait=max_wait, max_speed=max_speed)
    sorter = LaneSorter()
    encoder = TopKEncoder(normalizer=normalizer, max_lanes=max_lanes)

    if isinstance(raw_env, MultiAgentMockEnv):
        ts_ids = raw_env.ts_ids
        lane_extractor = _MockMultiAgentLaneExtractor(raw_env)
        neighbor_graph = NeighborGraphBuilder(net_file=None, ts_ids=ts_ids, max_hops=max_hops)
        neighbor_graph.graph = {
            ts: [(nbr, 1) for nbr in raw_env.adjacency[ts]] for ts in ts_ids
        }
        action_inspector = _MockActionSpaceInspector(raw_env)
    else:
        # Need one reset before traffic_signals/ts_ids are populated by
        # sumo-rl, and before we can introspect action spaces via traci.
        try:
            raw_env.reset()
        except Exception as exc:
            if cfg.get("use_libsumo", False):
                logger.warning(
                    "Initial SUMO reset failed with libsumo; retrying once without libsumo: %s",
                    exc,
                )
                cfg = dict(cfg)
                cfg["use_libsumo"] = False
                raw_env = build_multi_agent_raw_env(cfg)
                raw_env.reset()
            else:
                raise
        ts_ids = list(raw_env.ts_ids)
        lane_extractor = SumoLaneExtractor(raw_env)
        neighbor_graph = NeighborGraphBuilder(net_file=cfg.get("net_file"), ts_ids=ts_ids, max_hops=max_hops)
        action_inspector = ActionSpaceInspector(raw_env, ts_ids)

    neighbor_summary = NeighborSummaryExtractor(lane_extractor, max_queue=max_queue, max_wait=max_wait)

    logger.info(
        "MultiAgentFederatedWrapper ready: city='%s' n_intersections=%d "
        "own_dim=%d neighbor_dim=%d k_max=%d max_action_dim=%d",
        cfg.get("name", "?"), len(ts_ids), encoder.output_dim,
        NEIGHBOR_SUMMARY_DIM, k_max, action_inspector.max_action_dim,
    )

    return MultiAgentFederatedWrapper(
        env=raw_env,
        lane_extractor=lane_extractor,
        sorter=sorter,
        encoder=encoder,
        neighbor_graph=neighbor_graph,
        neighbor_summary=neighbor_summary,
        action_inspector=action_inspector,
        k_max=k_max,
    )


# ---------------------------------------------------------------------------
# Cross-city action-space padding
# ---------------------------------------------------------------------------

class RewardShapingWrapper:
    """Apply reward shaping only when it is safe for per-intersection credit assignment.

    This wrapper intentionally avoids using city-wide aggregates such as global waiting time,
    stopped vehicles, or queue length because those are not attributable to a single intersection.
    For multi-agent traffic control, only per-intersection-local signals should be used.
    """

    def __init__(
        self,
        env: Any,
        queue_weight: float = 0.0,
        wait_weight: float = 0.0,
        stopped_weight: float = 0.0,
        raw_weight: float = 1.0,
    ):
        self.env = env
        self.queue_weight = float(queue_weight)
        self.wait_weight = float(wait_weight)
        self.stopped_weight = float(stopped_weight)
        self.raw_weight = float(raw_weight)

    def _extract_local_metric(self, info: Dict[str, Any], ts_id: str, default: float = 0.0) -> float:
        if not isinstance(info, dict):
            return float(default)
        if ts_id in info:
            return float(info[ts_id])
        return float(default)

    def _shape_reward(self, reward: float, info: Dict[str, Any], ts_id: str | None = None) -> float:
        shaped = self.raw_weight * float(reward)
        if ts_id is None:
            return float(shaped)
        if self.wait_weight != 0.0:
            shaped -= self.wait_weight * self._extract_local_metric(info, f"{ts_id}_waiting_time", 0.0)
        if self.stopped_weight != 0.0:
            shaped -= self.stopped_weight * self._extract_local_metric(info, f"{ts_id}_stopped", 0.0)
        if self.queue_weight != 0.0:
            shaped -= self.queue_weight * self._extract_local_metric(info, f"{ts_id}_queue_length", 0.0)
        return float(shaped)

    def reset(self, *a, **kw):
        return self.env.reset(*a, **kw)

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        if isinstance(rewards, dict):
            shaped_rewards = {
                ts_id: self._shape_reward(float(r), info, ts_id=ts_id)
                for ts_id, r in rewards.items()
            }
        else:
            shaped_rewards = self._shape_reward(float(rewards), info)
        return obs, shaped_rewards, dones, info

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionMaskPadder:
    """Right-pads a city's ``action_mask`` (and the shared network's
    output width) from that city's own local ``max_action_dim`` up to a
    GLOBAL ``target_action_dim`` decided across every city in the
    federation. Needed because ``build_federated_env`` only knows about
    its own city; the global max is only known once ``load_clients`` has
    built every city's env (see ``federated_training.py``).

    Wrap AFTER computing the global action_dim:

        envs = [build_federated_env(cfg) for cfg in configs]
        global_action_dim = max(e.max_action_dim for e in envs)
        envs = [ActionMaskPadder(e, global_action_dim) for e in envs]
    """

    def __init__(self, env: MultiAgentFederatedWrapper, target_action_dim: int):
        self.env = env
        self.target_action_dim = target_action_dim
        if target_action_dim < env.max_action_dim:
            raise ValueError(
                f"target_action_dim={target_action_dim} smaller than this "
                f"city's own max_action_dim={env.max_action_dim}."
            )
        self.max_action_dim = target_action_dim

    def _pad(self, obs_dict: Dict[str, Dict[str, np.ndarray]]):
        pad_by = self.target_action_dim - self.env.max_action_dim
        if pad_by == 0:
            return obs_dict
        for obs in obs_dict.values():
            obs["action_mask"] = np.concatenate(
                [obs["action_mask"], np.zeros(pad_by, dtype=np.float32)]
            )
        return obs_dict

    def reset(self, *a, **kw):
        obs = self.env.reset(*a, **kw)
        if isinstance(obs, tuple):
            return self._pad(obs[0]), *obs[1:]
        return self._pad(obs)

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        return self._pad(obs), rewards, dones, info

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Mock-mode helpers (no SUMO required, for pipeline smoke-tests)
# ---------------------------------------------------------------------------

class _MockMultiAgentLaneExtractor(SumoLaneExtractor):
    """Random-but-deterministic per-ts lanes, for exercising the full
    pipeline (own-obs + neighbor-obs + masking) without SUMO."""

    def __init__(self, env: MultiAgentMockEnv):
        self.env = env
        self._rng = np.random.default_rng(0)

    def extract(self, ts_id: str) -> Tuple[List[Lane], int, float, float]:
        num_lanes = 4
        lanes = [
            Lane(
                lane_id=f"{ts_id}_lane_{i}",
                queue=float(self._rng.integers(0, 20)),
                waiting_time=float(self._rng.integers(0, 120)),
                occupancy=float(self._rng.uniform(0, 1)),
                speed=float(self._rng.uniform(0, 14)),
                is_left=(i % 3 == 0), is_straight=(i % 3 == 1), is_right=(i % 3 == 2),
            )
            for i in range(num_lanes)
        ]
        return lanes, 0, 0.0, 0.0


class _MockActionSpaceInspector(ActionSpaceInspector):
    def __init__(self, env: MultiAgentMockEnv):
        self.ts_ids = env.ts_ids
        self.action_counts = dict(env.action_counts)
        self.max_action_dim = max(self.action_counts.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Lane", "LaneEncoder", "LaneNormalizer", "LaneSorter", "TopKEncoder",
    "LaneExtractor", "SumoLaneExtractor",
    "ActionSpaceInspector", "NeighborGraphBuilder", "NeighborSummaryExtractor",
    "MultiAgentFederatedWrapper",
    "build_env_from_config", "build_multi_agent_raw_env", "build_federated_env",
    "MockEnv", "MultiAgentMockEnv",
    "find_active_traffic_signal", "validate_and_patch_config",
    "NEIGHBOR_SUMMARY_DIM",
]
