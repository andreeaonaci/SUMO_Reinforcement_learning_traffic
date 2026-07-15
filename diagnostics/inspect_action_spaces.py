"""Inspect the action space and neighbor graph of every intersection in
every city, without running any training.

Usage:
    python diagnostics/inspect_action_spaces.py [environments_dir]

Answers:
  - How many valid actions does each ts_id actually have? (should match
    what you expect from the net.xml's tlLogic phase count)
  - Which intersections have the fewest/most neighbors?
  - Does any city have an intersection whose action count looks wrong
    (e.g. 1 when you expected 2, or vice versa)?
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.getcwd())  # so `environments.*` / `agents.*` imports resolve when run from repo root

from environments.federated_env import build_federated_env  # noqa: E402


def inspect_city(name: str, cfg: dict) -> None:
    print(f"\n{'=' * 70}\nCity: {name}\n{'=' * 70}")
    try:
        env = build_federated_env(cfg)
    except Exception as e:
        print(f"  FAILED to build env: {e}")
        return

    try:
        inspector = env.action_inspector
        graph = env.neighbor_graph

        print(f"  own_dim={env.own_dim}  neighbor_dim={env.neighbor_dim}  "
              f"k_max={env.k_max}  max_action_dim={env.max_action_dim}")
        print(f"  {len(env.ts_ids)} intersections\n")

        print(f"  {'ts_id':<10} {'#actions':<10} {'#neighbors':<12} neighbor hops")
        for ts_id in sorted(env.ts_ids, key=lambda x: (len(x), x)):
            n_actions = inspector.action_counts.get(ts_id, "?")
            nbrs = graph.neighbors_of(ts_id)
            hop_summary = ", ".join(f"{nbr}(h{h})" for nbr, h in nbrs[:5])
            if len(nbrs) > 5:
                hop_summary += f", ... (+{len(nbrs) - 5} more)"
            flag = "  <-- only 1 action, check if expected" if n_actions == 1 else ""
            print(f"  {ts_id:<10} {n_actions:<10} {len(nbrs):<12} {hop_summary}{flag}")

        action_counts = list(inspector.action_counts.values())
        if len(set(action_counts)) > 1:
            print(f"\n  NOTE: action counts vary within this city: {sorted(set(action_counts))}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", nargs="?", default="environments")
    args = parser.parse_args()

    for name in sorted(os.listdir(args.base_dir)):
        cfg_path = os.path.join(args.base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        inspect_city(name, cfg)


if __name__ == "__main__":
    main()
