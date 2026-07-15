"""Test whether "always picks action 0" at an intersection could be a
legitimately correct policy, by checking whether traffic volume through
that intersection is actually asymmetric between the two phases.

This does NOT run SUMO -- it statically parses the route file's <flow>/
<trip>/<vehicle> 'route' or 'from'/'to' edge attributes and counts how
many vehicle-departures are assigned to each edge, then maps edges to
the intersection's approach lanes to get a rough per-phase volume split.

This is a heuristic, not a simulation -- route assignment doesn't
guarantee edge-by-edge realized volume (routing/rerouting can change it),
but a large skew here is a strong hint, and a int8 near-even split is a
strong hint the other way.

Usage:
    python diagnostics/route_traffic_balance.py environments/city_5_holdout/config.yaml
"""
import argparse
import sys
import os
from collections import Counter
from xml.etree import ElementTree as ET

import yaml


def count_edge_departures(route_file: str) -> Counter:
    """Count traversal volume through each edge (via full route edge
    lists), weighted by <flow> 'number'/'vehsPerHour' when present.

    NOTE: this counts every edge a route/flow traverses, not just where
    it departs. In a grid network, vehicles almost always originate on a
    border edge and only reach interior signalized approaches several
    hops into their route -- counting only the first edge (an earlier
    version of this script did that) would show near-zero volume on
    every interior approach regardless of actual traffic, which is a
    false signal, not evidence of balance.
    """
    counts: Counter = Counter()
    tree = ET.parse(route_file)
    root = tree.getroot()

    # Build route_id -> full edge list, for <route id=...> definitions
    route_edges = {}
    for route in root.iter("route"):
        rid = route.get("id")
        edges = route.get("edges", "").split()
        if rid and edges:
            route_edges[rid] = edges

    for tag in ("flow", "trip", "vehicle"):
        for el in root.iter(tag):
            weight = 1.0
            if el.get("number"):
                weight = float(el.get("number"))
            elif el.get("vehsPerHour"):
                weight = float(el.get("vehsPerHour"))  # relative weight, not absolute count

            edges = None
            if el.get("route") and el.get("route") in route_edges:
                edges = route_edges[el.get("route")]
            else:
                child_route = el.find("route")
                if child_route is not None and child_route.get("edges"):
                    edges = child_route.get("edges").split()
                elif el.get("from") and el.get("to"):
                    # No explicit route -- SUMO computes shortest path at
                    # runtime, which this static parse can't reproduce.
                    # Fall back to just from/to as a weak proxy.
                    edges = [el.get("from"), el.get("to")]
                elif el.get("from"):
                    edges = [el.get("from")]

            if edges:
                for e in edges:
                    counts[e] += weight

    return counts


def approaches_for_junction(net_file: str, junction_id: str):
    """Return the list of incoming edge ids for a junction, parsed
    straight from the net.xml (no sumolib dependency needed here)."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    for junction in root.iter("junction"):
        if junction.get("id") == junction_id:
            inc_lanes = junction.get("incLanes", "").split()
            edges = sorted({lane.rsplit("_", 1)[0] for lane in inc_lanes})
            return edges
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml")
    parser.add_argument("--junctions", nargs="*", default=None,
                         help="Specific junction/ts_ids to check (default: all in net)")
    args = parser.parse_args()

    with open(args.config_yaml) as f:
        cfg = yaml.safe_load(f)

    net_file = cfg["net_file"]
    route_file = cfg["route_file"]

    edge_departures = count_edge_departures(route_file)
    total = sum(edge_departures.values()) or 1.0

    tree = ET.parse(net_file)
    root = tree.getroot()
    junction_ids = args.junctions or [
        j.get("id") for j in root.iter("junction") if j.get("type") == "traffic_light"
    ]

    print(f"{'junction':<10} {'approach edges (share of total edge-traversal weight | share within junction)'}")
    print("-" * 90)
    for jid in sorted(junction_ids, key=lambda x: (len(x), x)):
        edges = approaches_for_junction(net_file, jid)
        if not edges:
            print(f"{jid:<10} (no incoming edges found)")
            continue
        raw = {e: edge_departures.get(e, 0) for e in edges}
        junction_total = sum(raw.values()) or 1.0
        parts = []
        for e, v in raw.items():
            global_share = v / total * 100
            within_share = v / junction_total * 100
            parts.append(f"{e}={global_share:.2f}%|{within_share:.0f}%")
        line = ", ".join(parts)
        within_shares = [v / junction_total for v in raw.values()]
        max_within = max(within_shares) if within_shares else 0
        skew_flag = "  <-- skewed toward one approach" if max_within > 0.7 and len(raw) > 1 else ""
        print(f"{jid:<10} {line}{skew_flag}")

    print(
        "\nNote: this counts route-file DEPARTURES assigned to each edge, not "
        "realized simulated volume -- treat a strong skew as 'worth checking "
        "further', not proof. Cross-reference against ts_ids that showed a "
        "flat action distribution in eval (see q_gap_trend.py)."
    )


if __name__ == "__main__":
    main()
