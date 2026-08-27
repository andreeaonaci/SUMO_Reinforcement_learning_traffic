"""Measure whether the default `diff-waiting-time` reward saturates
`DQNAgent.reward_clip` (hardcoded to 10.0, `agents/dqn.py:116`) the same way
raw `pressure` was found to (26% of ticks, sec 37) before its fix (sec 38,
`pressure_norm`).

Context: sec 37's writeup asserted "`diff-waiting-time` is already scaled
(divided by 100 inside `_diff_waiting_time_reward`) to roughly fit this
range by design" -- but that was never actually measured, only assumed by
reading the /100 scaling in the source. `_diff_waiting_time_reward` divides
the *accumulated* waiting time by 100 before differencing, not the diff
itself -- if accumulated waiting time can jump by more than ~1000 in a
single tick (very plausible during a congestion buildup, exactly the
moments a reward signal matters most), the resulting diff can still exceed
+-10 and get clipped in `DQNAgent.remember()`. If the default reward is
ALSO getting clip-saturated -- even occasionally, even just during the
congested episodes that matter most for learning -- that would be a
training-signal-destruction mechanism with nothing to do with confident
lock-in (sec 34/51) or switching behavior (sec 56), and would apply to
every single non-pressure experiment run anywhere in this project.

Runs each configured city under random actions (matching sec 37/38's own
methodology for measuring pressure's raw-value distribution) for one
episode and reports the full per-tick, per-intersection raw reward
distribution plus the fraction of ticks exceeding +-10 in magnitude.

Usage:
    python diagnostics/measure_reward_clip_saturation.py --base_dir environments_c1_4
"""
import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from agents.dqn import DQNAgent
from environments.federated_env import build_federated_env, ActionMaskPadder
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    maybe_pad_action_dim_to_true_holdout,
    make_holdout_evaluator,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_c1_4")
    ap.add_argument("--clip", type=float, default=10.0,
                     help="DQNAgent.reward_clip's hardcoded default -- match agents/dqn.py:116.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None,
                     help="If given, act greedily from this checkpoint instead of random actions "
                          "-- tests whether a genuinely bad/congested trajectory (as opposed to "
                          "random-action moderate traffic) saturates the clip harder.")
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                     help="Measure on the TRUE holdout city (via make_holdout_evaluator) instead "
                          "of the training cities -- required to reproduce the actual conditions "
                          "(topology, scale) that produce this project's catastrophic reported "
                          "eval rewards. Requires --checkpoint.")
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    ap.add_argument("--controller", default="trained", choices=["trained", "max_pressure", "fixed_time"],
                     help="Only used with --holdout: which policy drives the rollout. 'trained' "
                          "needs --checkpoint; the rule-based ones let you contrast per-tick "
                          "reward on a policy that solves the task well.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)

    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent = None
    if args.checkpoint:
        agent = DQNAgent(own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                          action_dim=action_dim, dueling=args.dueling)
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    def report(name, rewards):
        arr = np.array(rewards)
        saturated = np.mean(np.abs(arr) >= args.clip) * 100
        print(f"--- {name} ({len(arr)} tick-intersection reward samples) ---")
        print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        print(f"  sum (matches episode total reward scale)={arr.sum():.2f}")
        print(f"  fraction with |reward| >= {args.clip}: {saturated:.2f}%")
        half = len(arr) // 2
        if half > 0:
            print(f"  first-half mean={arr[:half].mean():.4f}  second-half mean={arr[half:].mean():.4f}"
                  " (tests whether the deficit concentrates late in the episode, sec 26's claim)")

    if args.holdout:
        if args.controller == "trained" and agent is None:
            raise ValueError("--controller trained requires --checkpoint.")
        evaluator = make_holdout_evaluator(
            args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
            episodes=1, eval_sumo_seed=args.eval_sumo_seed,
        )
        if evaluator is None:
            raise RuntimeError("Could not construct holdout evaluator.")
        env = evaluator._get_env()
        if hasattr(env, "fixed_ts"):
            env.fixed_ts = (args.controller == "fixed_time")
        evaluator._set_seed(args.eval_sumo_seed)
        obs_dict = env.reset()
        if not isinstance(obs_dict, dict):
            obs_dict = {"__single__": obs_dict}
        done = False
        holdout_rewards = []
        while not done:
            if args.controller == "trained":
                actions = {ts_id: agent.act(o, explore=False) for ts_id, o in obs_dict.items()}
            else:
                actions = {
                    ts_id: evaluator._policy_action(args.controller, ts_id, o, None)
                    for ts_id, o in obs_dict.items()
                }
            if len(actions) == 1 and "__single__" in actions:
                next_obs, reward, done, info = env.step(actions["__single__"])
                rewards = {"__single__": reward}
                dones = {"__all__": done}
                obs_dict = {"__single__": next_obs}
            else:
                obs_dict, rewards, dones, info = env.step(actions)
                done = dones.get("__all__", all(dones.values()) if dones else True)
            holdout_rewards.extend(rewards.values())
        evaluator.close()
        report(f"true holdout city ({evaluator.eval_city_name}, is_true_holdout={evaluator.is_true_holdout})", holdout_rewards)
        return

    all_rewards = []
    for name, cfg in city_configs:
        env = build_federated_env(cfg)
        if agent is not None and env.max_action_dim < action_dim:
            env = ActionMaskPadder(env, action_dim)
        obs = env.reset()
        done = False
        city_rewards = []
        while not done:
            if agent is not None:
                actions = {ts_id: agent.act(o, explore=False) for ts_id, o in obs.items()}
            else:
                actions = {
                    ts_id: int(np.random.choice(np.flatnonzero(o["action_mask"] > 0.5)))
                    for ts_id, o in obs.items()
                }
            obs, rewards, dones, info = env.step(actions)
            city_rewards.extend(rewards.values())
            done = dones.get("__all__", all(dones.values()) if dones else True)
        env.close()
        report(name, city_rewards)
        all_rewards.extend(city_rewards)

    report(f"overall (across {len(city_configs)} cities)", all_rewards)


if __name__ == "__main__":
    main()
