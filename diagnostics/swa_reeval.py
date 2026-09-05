"""Item 21 (fidings/divergence_investigation.md sec 78's queue): does averaging or
ensembling several consecutive rounds' checkpoints smooth over the confident-lock-in
volatility, WITHOUT touching training at all?

Motivation: every lever tried in sec 73-77 changed something about training (algorithm,
architecture, episode count). Nothing has touched *evaluation*. Since lock-in is round-to-
round volatile (a good round is often surrounded by bad ones, sec 51-53), combining several
rounds' weights or votes at eval time is a genuinely different, training-free axis:

  --mode average  (true SWA / Stochastic Weight Averaging): average the checkpoints'
                  state_dicts elementwise into ONE set of weights, build a single agent from
                  that, evaluate once. Cheap (one forward pass per tick, same as any single
                  checkpoint) but a linear interpolation in weight space doesn't linearly
                  interpolate in *behavior* space -- a confidently-locked checkpoint's sharp,
                  high-magnitude Q-value differences could skew the average unpredictably.

  --mode ensemble (majority vote): build N separate agents (one per checkpoint), and at each
                  tick take the majority vote of their independent greedy actions (falling
                  back to the highest-summed-Q action on a tie). More principled against a
                  MINORITY of locked checkpoints specifically -- the confirmed lock-in rate is
                  ~6-7% of rounds (sec 50), so an ensemble of e.g. 5 consecutive rounds should
                  rarely have more than one locked member, and a minority vote can't dominate
                  a majority decision. Costs N forward passes per tick instead of one.

Both modes are compared against each individual checkpoint evaluated alone (same episode
count), so the report answers "did combining actually help" directly, not just "what's the
combined number."

Usage:
    # explicit checkpoint list
    python diagnostics/swa_reeval.py results/run_.../global_round_016.pth \\
        results/run_.../global_round_017.pth ... --base_dir environments_c1_4_6 \\
        --pad_to_true_holdout --dueling --episodes 30 --mode ensemble

    # or auto-pick the last N checkpoints from a run directory
    python diagnostics/swa_reeval.py --run_dir results/run_.../  --last_n 5 \\
        --base_dir environments_c1_4_6 --pad_to_true_holdout --episodes 30 --mode average
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from agents.dqn import DQNAgent
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)
from diagnostics.finetune_on_holdout import infer_arch_from_checkpoint


class EnsemblePolicy:
    """Majority vote across N independently-loaded agents sharing one
    architecture. `.act(obs, explore=False)` is the only thing
    HoldoutEvaluator calls in the pure-argmax (explore=False) eval path
    this project always uses (sec 35) -- ties broken by summed Q-value
    across members, not arbitrarily, so a tie isn't just "whichever
    member's vote happened to be counted first."""

    def __init__(self, agents):
        self.agents = agents

    def act(self, obs, explore: bool = False):
        votes = [a.act(obs, explore=False) for a in self.agents]
        tally = Counter(votes)
        top_count = max(tally.values())
        tied = [a for a, c in tally.items() if c == top_count]
        if len(tied) == 1:
            return tied[0]
        q_sums = {a: 0.0 for a in tied}
        for agent in self.agents:
            q = agent.q_values(obs)
            for a in tied:
                if not np.isnan(q[a]):
                    q_sums[a] += q[a]
        return max(tied, key=lambda a: q_sums[a])

    def q_values(self, obs):
        # Mean Q-values across members -- only used for optional diagnostic
        # logging (Q-gap), not for action selection.
        qs = [a.q_values(obs) for a in self.agents]
        return np.nanmean(np.stack(qs), axis=0)


def _resolve_checkpoints(args) -> list:
    if args.checkpoints:
        return args.checkpoints
    if not args.run_dir:
        raise ValueError("Pass either explicit checkpoint paths or --run_dir with --last_n.")
    ckpts = sorted(
        glob.glob(os.path.join(args.run_dir, "global_round_*.pth")),
        key=lambda p: int(re.search(r"global_round_(\d+)\.pth$", p).group(1)),
    )
    if not ckpts:
        raise ValueError(f"No global_round_*.pth checkpoints found in {args.run_dir}")
    return ckpts[-args.last_n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="Explicit global_round_NNN.pth paths.")
    ap.add_argument("--run_dir", default=None, help="Alternative to explicit checkpoints: "
                     "auto-pick the last --last_n rounds from this run directory.")
    ap.add_argument("--last_n", type=int, default=5)
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--mode", choices=["average", "ensemble", "both"], default="both")
    args = ap.parse_args()

    ckpt_paths = _resolve_checkpoints(args)
    print(f"Using {len(ckpt_paths)} checkpoints: {[os.path.basename(p) for p in ckpt_paths]}")

    states = [torch.load(p, map_location="cpu") for p in ckpt_paths]
    arch = infer_arch_from_checkpoint(states[0])
    for s, p in zip(states[1:], ckpt_paths[1:]):
        other = infer_arch_from_checkpoint(s)
        if other != arch:
            raise ValueError(
                f"Checkpoint {p} has architecture {other}, expected {arch} (from "
                f"{ckpt_paths[0]}) -- all checkpoints being combined must share one "
                "architecture (same run, or at least identical own_dim/neighbor_dim/"
                "action_dim/dueling/head_fix)."
            )

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)
    if action_dim != arch["action_dim"]:
        raise ValueError(
            f"Checkpoint action_dim={arch['action_dim']} doesn't match this base_dir/flags' "
            f"action_dim={action_dim} -- pass --pad_to_true_holdout if the checkpoint was "
            "trained with it."
        )

    def build_agent(state):
        agent = DQNAgent(
            own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
            action_dim=arch["action_dim"], dueling=arch["dueling"], head_fix=arch["head_fix"],
        )
        agent.load_state_dict(state)
        return agent

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
        episodes=args.episodes, eval_sumo_seed=args.eval_sumo_seed,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    print(f"\n=== Individual checkpoints ({args.episodes} episodes each) ===")
    individual_results = []
    for p, state in zip(ckpt_paths, states):
        agent = build_agent(state)
        result = evaluator.evaluate(agent)
        individual_results.append(result["mean_reward"])
        print(f"  {os.path.basename(p)}: mean_reward={result['mean_reward']:.2f}  "
              f"std_reward={result['std_reward']:.2f}")
    print(f"  (individual mean of means: {np.mean(individual_results):.2f}, "
          f"best single checkpoint: {max(individual_results):.2f})")

    if args.mode in ("average", "both"):
        avg_state = {
            k: torch.stack([s[k].float() for s in states], dim=0).mean(dim=0)
            for k in states[0]
        }
        avg_agent = build_agent(avg_state)
        result = evaluator.evaluate(avg_agent)
        print(f"\n=== SWA weight-average of {len(states)} checkpoints ({args.episodes} episodes) ===")
        print(f"  mean_reward={result['mean_reward']:.2f}  std_reward={result['std_reward']:.2f}")

    if args.mode in ("ensemble", "both"):
        agents = [build_agent(s) for s in states]
        ensemble = EnsemblePolicy(agents)
        result = evaluator.evaluate(ensemble)
        print(f"\n=== Majority-vote ensemble of {len(states)} checkpoints ({args.episodes} episodes) ===")
        print(f"  mean_reward={result['mean_reward']:.2f}  std_reward={result['std_reward']:.2f}")

    evaluator.close()


if __name__ == "__main__":
    main()
