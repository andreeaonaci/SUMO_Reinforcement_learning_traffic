"""Re-evaluate a saved global checkpoint with more episodes than the training-time default.

Built to answer §32's open question: does a round's "crashed" reward survive averaging
over many more episodes/seeds, or was the 5-episode training-time eval just an unlucky
draw? Reuses the exact same evaluator construction as a real training run
(`experiments.federated_training.make_holdout_evaluator` + `resolve_city_configs_and_dims`)
so the eval env, masking, and seeding are identical to what produced the original number --
only `episodes` (and therefore how many distinct
`HoldoutEvaluator.eval_seed_base + ep` seeds get sampled) changes.

`--temperature T` (T>0) switches action selection from pure argmax to softmax sampling
over masked Q-values (`softmax(Q/T)`), via a thin wrapper that leaves
`federated/evaluator.py`'s production eval loop untouched -- built to test §34's finding
that crashed rounds are confidently locked into a repeating bad action, and that the rare
low-Q-gap (uncertain) episodes are what escape it. T=0 (default) is unchanged pure-greedy
behavior.

Usage:
    python diagnostics/reeval_checkpoint.py results/run_.../global_round_016.pth \
        --base_dir environments_c1_4 --episodes 30 --dueling --temperature 0.3 \
        --comm_dropout_p_link 0 --comm_dropout_p_isolate 0 --comm_dropout_p_hop_cutoff 0
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


class SoftmaxPolicy:
    """Wraps a DQNAgent so `.act(obs, explore=False)` samples from
    softmax(Q/temperature) over valid actions instead of pure argmax.
    Delegates everything else (`.q_values()`) to the underlying agent
    unchanged, so it's a drop-in `model` for `HoldoutEvaluator.evaluate()`."""

    def __init__(self, agent, temperature: float):
        self.agent = agent
        self.temperature = temperature

    def act(self, obs, explore: bool = False):
        if explore:
            return self.agent.act(obs, explore=True)
        q = self.agent.q_values(obs)
        valid_mask = ~np.isnan(q)
        valid_idx = np.flatnonzero(valid_mask)
        valid_q = q[valid_mask]
        scaled = valid_q / self.temperature
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        return int(np.random.choice(valid_idx, p=probs))

    def q_values(self, obs):
        return self.agent.q_values(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to a global_round_NNN.pth (Q-network state_dict only)")
    ap.add_argument("--base_dir", default="environments_c1_4")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--disable_neighbor_attention", action="store_true")
    ap.add_argument("--comm_dropout_p_link", type=float, default=0.0)
    ap.add_argument("--comm_dropout_p_isolate", type=float, default=0.0)
    ap.add_argument("--comm_dropout_p_hop_cutoff", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0 (default) = pure argmax. >0 = softmax(Q/T) stochastic action selection.")
    ap.add_argument("--pad_to_true_holdout", action="store_true",
                     help="Widen action_dim to city_5_holdout's width before loading the "
                          "checkpoint -- required for any checkpoint produced by a "
                          "--pad_to_true_holdout training run (the standard setup since "
                          "fidings/divergence_investigation.md sec 43), otherwise load_state_dict "
                          "fails on a Q-head size mismatch.")
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
    if args.temperature > 0:
        agent = SoftmaxPolicy(agent, args.temperature)

    comm_cfg = {
        "p_link": args.comm_dropout_p_link,
        "p_isolate": args.comm_dropout_p_isolate,
        "p_hop_cutoff": args.comm_dropout_p_hop_cutoff,
    }

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
        episodes=args.episodes,
        eval_comm_dropout_cfg=comm_cfg,
        eval_sumo_seed=args.eval_sumo_seed,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    result = evaluator.evaluate(agent)
    evaluator.close()

    rewards = result.get("per_episode_reward", result.get("episode_rewards"))
    print(f"checkpoint={args.checkpoint}")
    print(f"episodes={args.episodes}")
    print(f"temperature={args.temperature} ({'softmax' if args.temperature > 0 else 'pure argmax'})")
    print(f"mean_reward={result.get('mean_reward'):.4f}  std_reward={result.get('std_reward'):.4f}")
    if rewards:
        print(f"min={min(rewards):.2f}  max={max(rewards):.2f}")
        print(f"per_episode_reward={[round(r,2) for r in rewards]}")

    # Per-episode Q(top1)-Q(top2) gap (mean and min across intersections that
    # episode) alongside that episode's reward -- tests whether "bad" episodes
    # are ones where the greedy policy was making close-call (near-tied)
    # decisions more often, i.e. whether pure argmax is fragile at small gaps.
    q_gaps = result.get("q_gaps")
    if q_gaps and rewards:
        print("episode  reward     mean_gap    min_gap   n_ts")
        for i, (r, gaps) in enumerate(zip(rewards, q_gaps)):
            vals = list(gaps.values())
            if vals:
                mean_gap = sum(vals) / len(vals)
                min_gap = min(vals)
            else:
                mean_gap = min_gap = float("nan")
            print(f"{i:7d}  {r:9.2f}  {mean_gap:9.4f}  {min_gap:9.4f}  {len(vals):4d}")


if __name__ == "__main__":
    main()
