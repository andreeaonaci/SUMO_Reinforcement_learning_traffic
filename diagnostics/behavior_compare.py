"""Compare the trained DQN's actual control behavior against the rule-based
baselines it loses to, on the same holdout city / same episodes.

Context: fidings/divergence_investigation.md sec 51 established that
confident lock-in (sec 34/53) is only a *secondary* factor in the ~3-4 order
of magnitude baseline gap -- locked vs. not-locked rounds differ by only
~29% mean reward, both still catastrophically worse than fixed_time/
max_pressure. Sec 55 confirmed fixing the lock-in mechanism directly
(--q_entropy_weight) doesn't move the reward gap either. So something else,
still unidentified, is the primary driver.

This script doesn't assume an answer -- it just surfaces the cheapest
behavioral signal that hasn't been looked at yet: how often the trained
policy switches phases per intersection per episode (thrashing costs a
yellow-time transition every switch, unlike max_pressure, which only
switches when pressure actually favors it) and how concentrated its action
distribution is (dominant-action fraction, the same statistic sec 53 used
to characterize confident lock-in) on a checkpoint that is NOT in a locked
state (best-known-reward round, so a "clean" behavioral read). Reuses
federated/evaluator.py::HoldoutEvaluator unchanged -- it already records
per-tick action_log and per-episode q_gaps for the trained policy; this
script just does the switch-rate/dominant-fraction arithmetic and prints a
side-by-side table against fixed_time and max_pressure on the identical
holdout episodes.

Usage:
    python diagnostics/behavior_compare.py <checkpoint.pth> \
        --base_dir environments_c1_4 --episodes 10 --dueling --pad_to_true_holdout
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from agents.dqn import DQNAgent
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)


def switch_rate_and_dominant_fraction(action_log):
    """action_log: list of episodes, each a list of {ts_id: action} dicts
    (one per tick, up to 200 ticks). Returns per-ts_id switch rate (fraction
    of consecutive ticks where the action changed) and dominant-action
    fraction (share of ticks spent on the single most-used action),
    averaged across episodes."""
    per_ts_switches = {}
    per_ts_ticks = {}
    per_ts_action_counts = {}

    for ep_log in action_log:
        prev = {}
        for tick in ep_log:
            for ts_id, a in tick.items():
                per_ts_ticks[ts_id] = per_ts_ticks.get(ts_id, 0) + 1
                per_ts_action_counts.setdefault(ts_id, {})
                per_ts_action_counts[ts_id][a] = per_ts_action_counts[ts_id].get(a, 0) + 1
                if ts_id in prev and prev[ts_id] != a:
                    per_ts_switches[ts_id] = per_ts_switches.get(ts_id, 0) + 1
                prev[ts_id] = a

    switch_rate = {
        ts_id: per_ts_switches.get(ts_id, 0) / max(1, per_ts_ticks[ts_id])
        for ts_id in per_ts_ticks
    }
    dominant_fraction = {
        ts_id: max(counts.values()) / sum(counts.values())
        for ts_id, counts in per_ts_action_counts.items()
    }
    return switch_rate, dominant_fraction


def summarize(name, result):
    action_log = result.get("action_log", [])
    switch_rate, dominant_fraction = switch_rate_and_dominant_fraction(action_log)
    mean_switch = float(np.mean(list(switch_rate.values()))) if switch_rate else float("nan")
    mean_dominant = float(np.mean(list(dominant_fraction.values()))) if dominant_fraction else float("nan")
    print(f"--- {name} ---")
    print(f"  mean_reward={result.get('mean_reward'):.2f}  std_reward={result.get('std_reward'):.2f}")
    print(f"  mean_waiting_time={result.get('mean_waiting_time'):.2f}  mean_queue_length={result.get('mean_queue_length'):.4f}")
    print(f"  mean phase-switch rate per tick (per intersection, avg): {mean_switch:.4f}")
    print(f"  mean dominant-action fraction (per intersection, avg): {mean_dominant:.4f}")
    q_gaps = result.get("q_gaps", [])
    if q_gaps and any(q_gaps):
        all_gaps = [g for ep in q_gaps for g in ep.values()]
        if all_gaps:
            print(f"  mean Q(top1)-Q(top2) gap across episodes: {float(np.mean(all_gaps)):.4f}")
    return mean_switch, mean_dominant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--base_dir", default="environments_c1_4")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--disable_neighbor_attention", action="store_true")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    args = ap.parse_args()

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent = DQNAgent(
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        head_fix=not args.disable_neighbor_attention,
        dueling=args.dueling,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    agent.load_state_dict(state)

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
        episodes=args.episodes,
        eval_sumo_seed=args.eval_sumo_seed,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    print(f"checkpoint={args.checkpoint}  episodes={args.episodes}")
    trained_result = evaluator.evaluate(agent)
    summarize("trained (DQN)", trained_result)

    fixed_result = evaluator.evaluate_controller("fixed_time")
    summarize("fixed_time", fixed_result)

    mp_result = evaluator.evaluate_controller("max_pressure")
    summarize("max_pressure", mp_result)

    evaluator.close()


if __name__ == "__main__":
    main()
