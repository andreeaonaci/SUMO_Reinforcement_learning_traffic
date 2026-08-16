"""Re-evaluate a saved global checkpoint with more episodes than the training-time default.

Built to answer §32's open question: does a round's "crashed" reward survive averaging
over many more episodes/seeds, or was the 5-episode training-time eval just an unlucky
draw? Reuses the exact same evaluator construction as a real training run
(`experiments.federated_training.make_holdout_evaluator` + `resolve_city_configs_and_dims`)
so the eval env, masking, and seeding are identical to what produced the original number --
only `episodes` (and therefore how many distinct
`HoldoutEvaluator.eval_seed_base + ep` seeds get sampled) changes.

Usage:
    python diagnostics/reeval_checkpoint.py results/run_.../global_round_016.pth \
        --base_dir environments_c1_4 --episodes 30 --dueling \
        --comm_dropout_p_link 0 --comm_dropout_p_isolate 0 --comm_dropout_p_hop_cutoff 0
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from agents.dqn import DQNAgent
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
)


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
    args = ap.parse_args()

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)

    agent = DQNAgent(
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        head_fix=not args.disable_neighbor_attention,
        dueling=args.dueling,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    agent.load_state_dict(state)

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
    print(f"mean_reward={result.get('mean_reward'):.4f}  std_reward={result.get('std_reward'):.4f}")
    if rewards:
        print(f"min={min(rewards):.2f}  max={max(rewards):.2f}")
        print(f"per_episode_reward={[round(r,2) for r in rewards]}")


if __name__ == "__main__":
    main()
