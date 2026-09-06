"""Sequential (non-federated) curriculum training: instead of training every
city in parallel and averaging weights each round (FedAvg), fully train on
one city, then CONTINUE training the exact same weights on the next city, and
so on through the whole roster -- one pass, no going back, no aggregation
step at all.

Motivation: every federation-vs-no-federation comparison in this project
(fidings/divergence_investigation.md sec 49/50/64) found parallel FedAvg
makes no measurable difference over training each city independently. The
one thing that HAS reliably worked is sequential adaptation -- fine-tuning a
pretrained checkpoint on a NEW city (sec 66-69). This script asks whether
applying that same "sequentially adapt to one more city" mechanism across the
WHOLE training roster (city_1 -> city_4 -> city_6), rather than only once at
holdout-eval time, produces a model that generalizes to the true holdout
better than parallel FedAvg does -- a genuinely different training paradigm,
not a variant of anything already tried in the item-2X series.

Direct comparability with the existing environments_c1_4_6 parallel-baseline
data (items 22/23/24/TC-FedAvg's baseline arm, `--rounds 5 --local_episodes 2`
= 10 total local episodes per city): this script also gives each city
`--episodes_per_city` (default 10) episodes of training, just delivered as
one continuous block per city instead of interleaved 2-episode rounds with
inter-city averaging -- same total SUMO training volume, no averaging step,
sequential order is the only manipulated variable that matters for a fair
comparison against the existing baseline data.

Two things measured beyond the final holdout number, both cheap (reuses the
same evaluator already built):
  1. Holdout eval after EACH city finishes training -- does progressive
     sequential exposure help, hurt, or fluctuate as more cities are added?
  2. Catastrophic forgetting check: each training city's OWN in-distribution
     performance is measured right after its own training phase, then AGAIN
     at the very end (after every later city has also been trained) -- if
     later cities' training destructively overwrites earlier cities' learned
     behavior, this will show up directly as a large gap between those two
     numbers.

Usage:
    python diagnostics/sequential_training.py --base_dir environments_c1_4_6 \\
        --pad_to_true_holdout --episodes_per_city 10 --seed 3
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.dqn import DQNAgent
from environments.federated_env import ActionMaskPadder, build_federated_env
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)
from federated.utils import set_seed

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--episodes_per_city", type=int, default=10,
                     help="Total local episodes per city, delivered as one continuous "
                          "block -- matches the TOTAL per-city episode count of the "
                          "existing parallel-baseline pilots (--rounds 5 --local_episodes 2 "
                          "= 10), so the comparison isolates sequential-vs-parallel/averaged "
                          "as the only manipulated variable.")
    ap.add_argument("--q_entropy_weight", type=float, default=0.05,
                     help="Matches the current standard baseline used by items 20-25/"
                          "TC-FedAvg, for direct comparability against that existing data.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--log_loss_every_steps", type=int, default=50)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent = DQNAgent(
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        q_entropy_weight=args.q_entropy_weight,
    )

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim, episodes=args.eval_episodes,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    def eval_holdout(label):
        result = evaluator.evaluate(agent)
        logger.info(
            "%s -- HOLDOUT: mean_reward=%.2f std_reward=%.2f",
            label, result["mean_reward"], result["std_reward"],
        )
        return result

    def eval_in_distribution(name, label):
        # A fresh evaluator pointed AT this specific training city (not the holdout),
        # built directly from its own config -- HoldoutEvaluator just needs an
        # env_builder, it doesn't care whether that env is "the" holdout city or not.
        from federated.evaluator import HoldoutEvaluator
        cfg = dict(next(c for n, c in city_configs if n == name))
        def env_builder(cfg=cfg):
            e = build_federated_env(cfg)
            return ActionMaskPadder(e, action_dim)
        city_eval = HoldoutEvaluator(env_builder=env_builder, episodes=args.eval_episodes)
        result = city_eval.evaluate(agent)
        city_eval.close()
        logger.info(
            "%s -- %s (in-distribution): mean_reward=%.2f std_reward=%.2f",
            label, name, result["mean_reward"], result["std_reward"],
        )
        return result

    eval_holdout("Stage 0 (random init, before any training)")

    trained_so_far = []
    for name, cfg in city_configs:
        logger.info("=== Training on %s (%d episodes) ===", name, args.episodes_per_city)
        env = build_federated_env(cfg)
        env = ActionMaskPadder(env, action_dim)
        agent.train(env, episodes=args.episodes_per_city, log_loss_every_steps=args.log_loss_every_steps)

        trained_so_far.append(name)
        env.close()
        eval_holdout(f"After training on {'+'.join(trained_so_far)}")
        eval_in_distribution(name, f"After training on {'+'.join(trained_so_far)}")

    logger.info("=== Final forgetting check: re-evaluating every training city ===")
    for name, _ in city_configs:
        eval_in_distribution(name, "FINAL (after all cities trained)")

    evaluator.close()


if __name__ == "__main__":
    main()
