"""Continue training from a saved checkpoint with exploration reset to full, to test
item 11(a): can a short burst of extra exploration let an already-"locked" run
(§33/§34's degenerate, confidently-wrong repeating policy) walk into the good branch
§36 showed is reachable from the same weights, without needing an eval-time patch?

Deliberately does NOT use --resume: that path computes
`init_steps_done = completed_round * local_episodes * steps_per_ep` specifically so
epsilon keeps decaying from where it left off (already ~0.05 by round 16-20, the
regime §34 found the lock-in in) -- exactly the behavior this test needs to bypass.
Instead builds a ParallelFederatedServer directly with the checkpoint's weights
pre-loaded into a fresh DQNAgent (init_steps_done=0 -> epsilon restarts at 1.0) and a
fresh eps_decay schedule sized to the short recovery run, not the original 20-round one.

Usage:
    python diagnostics/recovery_finetune.py results/run_.../global_round_020.pth \
        --base_dir environments_c1_4 --rounds 5 --local_episodes 2 --dueling --n_step 3 \
        --seed 3 --comm_dropout_p_link 0 --comm_dropout_p_isolate 0 --comm_dropout_p_hop_cutoff 0
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from agents.dqn import DQNAgent
from federated.parallel_server import ParallelFederatedServer
from federated.utils import compute_eps_decay
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    steps_per_episode_from_cfg,
    make_holdout_evaluator,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to a global_round_NNN.pth to resume weights (not schedule) from")
    ap.add_argument("--base_dir", default="environments_c1_4")
    ap.add_argument("--rounds", type=int, default=5, help="Length of the recovery burst")
    ap.add_argument("--local_episodes", type=int, default=2)
    ap.add_argument("--explore_fraction", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_decay", type=float, default=0.97)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--n_step", type=int, default=1)
    ap.add_argument("--disable_neighbor_attention", action="store_true")
    ap.add_argument("--comm_dropout_p_link", type=float, default=0.0)
    ap.add_argument("--comm_dropout_p_isolate", type=float, default=0.0)
    ap.add_argument("--comm_dropout_p_hop_cutoff", type=float, default=0.0)
    ap.add_argument("--checkpoint_dir", default=None,
                     help="Default: results/recovery_finetune_<basename of input checkpoint>")
    args = ap.parse_args()

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, steps_per_ep_first = \
        resolve_city_configs_and_dims(args.base_dir)
    steps_per_ep = steps_per_episode_from_cfg(city_configs[0][1])

    eps_decay = compute_eps_decay(
        rounds=args.rounds, local_episodes=args.local_episodes,
        steps_per_episode=steps_per_ep, explore_fraction=args.explore_fraction,
    )
    print(f"Recovery eps_decay={eps_decay:.1f} (epsilon restarts at 1.0, "
          f"reaches ~0.05 floor by ~{args.explore_fraction*100:.0f}% of the "
          f"{args.rounds}-round recovery burst)")

    global_model = DQNAgent(
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        head_fix=not args.disable_neighbor_attention,
        dueling=args.dueling, n_step=args.n_step,
        eps_decay=eps_decay,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    global_model.load_state_dict(state)
    print(f"Loaded weights from {args.checkpoint} (schedule NOT resumed -- steps_done=0, fresh epsilon)")

    comm_cfg = {
        "p_link": args.comm_dropout_p_link,
        "p_isolate": args.comm_dropout_p_isolate,
        "p_hop_cutoff": args.comm_dropout_p_hop_cutoff,
    }

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
        episodes=5,
        eval_comm_dropout_cfg=comm_cfg,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    checkpoint_dir = args.checkpoint_dir or os.path.join(
        "results", f"recovery_finetune_{os.path.basename(args.checkpoint).replace('.pth', '')}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    server = ParallelFederatedServer(
        global_model=global_model,
        city_configs=city_configs,
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        eps_decay=eps_decay,
        comm_dropout_cfg=comm_cfg,
        local_episodes=args.local_episodes,
        evaluator=evaluator,
        checkpoint_dir=checkpoint_dir,
        default_lr=args.lr, lr_decay=args.lr_decay, min_lr=args.min_lr,
        head_fix=True,
        neighbor_attention=not args.disable_neighbor_attention,
        seed=args.seed,
        dueling=args.dueling,
        n_step=args.n_step,
        init_steps_done=0,  # the whole point -- fresh exploration, not resumed schedule
    )
    try:
        history = server.run(rounds=args.rounds, eval_every=1)
    finally:
        server.close()
        evaluator.close()

    print(f"\ncheckpoint_dir={checkpoint_dir}")
    print("round  eval_reward  eval_waiting_time")
    for r, rew, wt in zip(history["round"], history["eval_reward"], history["eval_waiting_time"]):
        print(f"{r:5d}  {rew:11.2f}  {wt:17.2f}")


if __name__ == "__main__":
    main()
