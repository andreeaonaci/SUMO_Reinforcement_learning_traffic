"""Measure REALIZED per-approach-edge traffic volume by actually running
one SUMO episode -- not by statically parsing the route file.

Why this exists: route files that only specify <flow from=... to=...>
(origin/destination, no explicit route) don't record which edges a
vehicle actually drives through anywhere in the file -- SUMO's internal
router computes that at runtime. `route_traffic_balance.py`'s static
parse can't see that; this script actually runs the sim and reads real
per-edge vehicle counts via traci/libsumo instead.

Usage:
    python diagnostics/measure_approach_volume.py environments/city_5_holdout/config.yaml
    python diagnostics/measure_approach_volume.py environments/city_5_holdout/config.yaml --junctions 12 13 14 15
"""
import argparse
import os
import sys
from collections import defaultdict
from xml.etree import ElementTree as ET

import yaml

sys.path.insert(0, os.getcwd())

from environments.federated_env import build_multi_agent_raw_env  # noqa: E402


def approaches_for_junction(net_file: str, junction_id: str):
    tree = ET.parse(net_file)
    root = tree.getroot()
    for junction in root.iter("junction"):
        if junction.get("id") == junction_id:
            inc_lanes = junction.get("incLanes", "").split()
            edges = sorted({lane.rsplit("_", 1)[0] for lane in inc_lanes})
            return edges
    return []


def all_traffic_light_junctions(net_file: str):
    tree = ET.parse(net_file)
    root = tree.getroot()
    return [j.get("id") for j in root.iter("junction") if j.get("type") == "traffic_light"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml")
    parser.add_argument("--junctions", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Cap simulation ticks (default: cfg num_seconds/delta_time)")
    args = parser.parse_args()

    with open(args.config_yaml) as f:
        cfg = yaml.safe_load(f)

    net_file = cfg["net_file"]
    if not os.path.isabs(net_file):
        net_file = os.path.join('.', net_file)

    junction_ids = args.junctions or all_traffic_light_junctions(net_file)
    approach_edges = {}
    all_edges = set()
    for jid in junction_ids:
        edges = approaches_for_junction(net_file, jid)
        approach_edges[jid] = edges
        all_edges.update(edges)

    print(f"Measuring realized traffic for {len(junction_ids)} junctions, "
          f"{len(all_edges)} approach edges. Running SUMO...")

    env = build_multi_agent_raw_env(cfg)
    obs = env.reset()
    if not isinstance(obs, dict):
        obs = {"__single__": obs}

    try:
        import traci
    except Exception:
        print("ERROR: traci/libsumo not importable in this environment.")
        return

    edge_vehicle_seconds: dict = defaultdict(float)
    delta_time = cfg.get("delta_time", 5)
    num_seconds = cfg.get("num_seconds", 3600)
    max_steps = args.max_steps or (num_seconds // delta_time)

    ts_ids = list(obs.keys()) if "__single__" not in obs else []
    step = 0
    done = False
    while not done and step < max_steps:
        # Actions don't matter for this measurement -- always pick action 0.
        actions = {ts_id: 0 for ts_id in ts_ids} if ts_ids else 0
        if ts_ids:
            _, _, dones, _ = env.step(actions)
            done = dones.get("__all__", all(dones.values()) if dones else True)
        else:
            _, _, done, _ = env.step(actions)

        for e in all_edges:
            try:
                n = traci.edge.getLastStepVehicleNumber(e)
                edge_vehicle_seconds[e] += n * delta_time
            except Exception:
                pass
        step += 1

    env.close()

    print(f"\nRan {step} sim ticks ({step * delta_time}s simulated).\n")
    print(f"{'junction':<10} approach edges (vehicle-seconds | share within junction)")
    print("-" * 90)
    for jid in sorted(junction_ids, key=lambda x: (len(x), x)):
        edges = approach_edges[jid]
        if not edges:
            print(f"{jid:<10} (no incoming edges found)")
            continue
        raw = {e: edge_vehicle_seconds.get(e, 0.0) for e in edges}
        jt = sum(raw.values()) or 1.0
        parts = [f"{e}={v:.0f}veh-s|{v/jt*100:.0f}%" for e, v in raw.items()]
        max_share = max(raw.values()) / jt if jt else 0
        skew_flag = "  <-- skewed toward one approach" if max_share > 0.7 and len(raw) > 1 else ""
        print(f"{jid:<10} {', '.join(parts)}{skew_flag}")

    print(
        "\n'vehicle-seconds' = sum over sim ticks of (vehicles present on "
        "edge * tick length) -- a real, measured proxy for traffic volume "
        "through that approach, unlike the static route-file guess."
    )


if __name__ == "__main__":
    main()
