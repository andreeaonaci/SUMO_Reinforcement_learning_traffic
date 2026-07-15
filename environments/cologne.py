"""
Example: integrating a new city — Cologne
==========================================
This file demonstrates the *minimum* work required to add a new city to the
federated framework.

Checklist
---------
1. Implement ``CologneLaneExtractor(LaneExtractor)``.
2. Call ``make_adapter(env, cfg)`` — returns a ``CityAdapter`` instance.
3. Register the dotted path in the city's ``config.yaml``.

Nothing else changes: normalisation, sorting, encoding, action mapping,
and the RL training loop are all untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from federated_env import (
    CityAdapter,
    Lane,
    LaneEncoder,
    LaneExtractor,
)


# ---------------------------------------------------------------------------
# Step 1 — City-specific lane extractor
# ---------------------------------------------------------------------------

class CologneLaneExtractor(LaneExtractor):
    """Extract lane state from a Cologne SUMO environment.

    The ``env`` here is a raw ``SumoEnvironment`` from ``sumo-rl``.
    Adapt field access to match whatever API your environment exposes.
    """

    # The traffic signal ID this extractor reads from
    TS_ID = "J1"

    def extract(self) -> Tuple[List[Lane], int, float, float]:
        ts = self.env.traffic_signals[self.TS_ID]

        lanes: List[Lane] = []
        for lane_id in ts.lanes:
            sub = ts.sumo.lane.getSubscriptionResults(lane_id)

            lanes.append(
                Lane(
                    lane_id=lane_id,
                    queue=float(ts.sumo.lane.getLastStepHaltingNumber(lane_id)),
                    waiting_time=float(ts.sumo.lane.getWaitingTime(lane_id)),
                    occupancy=float(ts.sumo.lane.getLastStepOccupancy(lane_id)) / 100.0,
                    speed=float(ts.sumo.lane.getLastStepMeanSpeed(lane_id)),
                    density=float(ts.sumo.lane.getLastStepVehicleNumber(lane_id)),
                    is_left="left"     in lane_id,
                    is_straight="straight" in lane_id,
                    is_right="right"   in lane_id,
                )
            )

        current_phase = int(ts.phase)
        elapsed_green = float(ts.time_since_last_phase_change)
        yellow_time   = float(ts.yellow_time)

        return lanes, current_phase, elapsed_green, yellow_time


# ---------------------------------------------------------------------------
# Step 2 — City adapter factory
# ---------------------------------------------------------------------------

# Cologne has 4 phases (0, 2, 4, 6) exposed through a 4-action shared policy.
_COLOGNE_PHASE_MAPPING: Dict[int, int] = {
    0: 0,
    1: 2,
    2: 4,
    3: 6,
}


def make_adapter(env: Any, cfg: Dict[str, Any]) -> CityAdapter:
    """Factory function referenced in config.yaml as ``adapter_class``.

    Args:
        env: Raw SumoEnvironment (already built by ``build_env_from_config``).
        cfg: City config dict from config.yaml.

    Returns:
        A fully configured ``CityAdapter`` for Cologne.
    """
    extractor = CologneLaneExtractor(env)

    return CityAdapter(
        name="cologne",
        extractor=extractor,
        phase_mapping=_COLOGNE_PHASE_MAPPING,
        # Cologne-specific normalization bounds
        max_lanes=12,
        max_queue=40.0,
        max_wait=200.0,
        max_speed=14.0,          # 50 km/h city speed limit
        encoder=LaneEncoder,     # use the default schema; override here if needed
    )


# ---------------------------------------------------------------------------
# Step 3 — config.yaml entry (shown as a comment)
# ---------------------------------------------------------------------------
#
# name: cologne
# mode: sumo
# net_file:   nets/cologne.net.xml
# route_file: routes/cologne.rou.xml
# single_agent: true
# num_seconds: 3600
#
# # federated-specific
# adapter_class: myproject.cities.cologne.make_adapter
# max_lanes:  12
# max_queue:  40
# max_wait:   200
# max_speed:  14.0
# phase_mapping:
#   0: 0
#   1: 2
#   2: 4
#   3: 6
