"""Per-movement pressure, the state MPLight/FRAP consumes (fidings sec 101).

RESCO's `mdp_options/states.py::mplight` is, per signal:

    [current_phase] + [ pressure(movement) for movement in lane_sets ]

where pressure(movement) = (queue on that movement's inbound lanes)
                         - (queue on the downstream lanes it discharges onto),
with `demand_shape: 1`.

This reproduces that from `configs/resco_frap/phase_pairs.json`, which holds
RESCO's own hand-authored `lane_sets` plus the `lane_sets_outbound` derived from
their `downstream` wiring (see diagnostics/build_frap_config.py). Nothing here is
re-derived from the simulator: handing FRAP its authored configuration verbatim
is the whole point of the sec 101 comparison, since that configuration is
precisely what this project's phase-relational head does NOT need.

Signal ids are globally unique across the four RESCO cities (verified), so the
lookup is keyed by ts_id alone and this class never needs to know which city it
is in -- keeping the same topology-agnostic discipline as the rest of the
observation pipeline.
"""
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "resco_frap", "phase_pairs.json")


class MovementPressureExtractor:
    """-> (movement_pressure (12,), current_union_phase, act_to_union (A,))."""

    def __init__(self, env, config_path: str = DEFAULT_CONFIG):
        self.env = env
        self.directions: List[str] = []
        self.union_pairs: List[List[str]] = []
        self.lane_sets: Dict[str, Dict[str, List[str]]] = {}
        self.lane_sets_outbound: Dict[str, Dict[str, List[str]]] = {}
        self.act_to_union: Dict[str, Dict[int, int]] = {}
        self.available = False

        if not os.path.exists(config_path):
            logger.warning(
                "MovementPressureExtractor: no config at %s -- the FRAP arm "
                "cannot run. Build it with diagnostics/build_frap_config.py.",
                config_path)
            return

        cfg = json.load(open(config_path))
        self.directions = cfg["directions"]
        self.union_pairs = cfg["union_pairs"]
        for _city, blk in cfg["scenarios"].items():
            self.lane_sets.update(blk["lane_sets"])
            self.lane_sets_outbound.update(blk.get("lane_sets_outbound", {}))
            for sid, m in blk["act_to_union"].items():
                self.act_to_union[sid] = {int(k): int(v) for k, v in m.items()}
        self.available = True

    @property
    def n_movements(self) -> int:
        return len(self.directions)

    def covers(self, ts_id: str) -> bool:
        return ts_id in self.lane_sets

    def missing(self, ts_ids) -> List[str]:
        """Signals with no authored configuration -- FRAP cannot control these."""
        return [t for t in ts_ids if not self.covers(t)]

    def _queue(self, traci_module, lane_ids) -> float:
        total = 0.0
        for lid in lane_ids:
            try:
                total += float(traci_module.lane.getLastStepHaltingNumber(lid))
            except Exception:
                pass
        return total

    def extract(self, ts_id: str, current_phase: int, action_dim: int
                ) -> Tuple[np.ndarray, int, np.ndarray]:
        pressure = np.zeros(self.n_movements, dtype=np.float32)
        act_vec = np.full(action_dim, -1, dtype=np.int64)

        amap = self.act_to_union.get(ts_id, {})
        for local_act, union_phase in amap.items():
            if 0 <= local_act < action_dim:
                act_vec[local_act] = union_phase
        current_union = int(amap.get(int(current_phase), -1))

        lane_sets = self.lane_sets.get(ts_id)
        if lane_sets is None:
            return pressure, current_union, act_vec

        try:
            import traci
        except Exception:
            return pressure, current_union, act_vec

        outbound = self.lane_sets_outbound.get(ts_id, {})
        for movement, lanes in lane_sets.items():
            if movement not in self.directions:
                continue
            idx = self.directions.index(movement)
            inbound_q = self._queue(traci, lanes or [])
            outbound_q = self._queue(traci, outbound.get(movement, []))
            # RESCO feeds raw (unnormalised) pressure into a Linear+sigmoid;
            # kept raw here so the head sees the same scale it was designed for.
            pressure[idx] = inbound_q - outbound_q

        return pressure, current_union, act_vec
