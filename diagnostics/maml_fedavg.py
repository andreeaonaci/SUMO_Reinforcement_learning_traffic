"""Item 4 of the four "significantly improve" candidates (fidings/
divergence_investigation.md sec 91): proper (second-order) MAML meta-
learning aggregation, replacing FedAvg's "average client state dicts" step
with one true MAML meta-gradient step per round (see federated/maml.py for
the mechanism and how it differs from item 24's already-null Reptile-style
`--fedavg_blend`, sec 83).

Per round, for every city:
  1. Collect --collect_episodes_per_round fresh episodes of experience into
     that city's OWN replay buffer (epsilon-greedy on the CURRENT global
     network -- same exploration convention as ordinary federated training).
  2. Sample --inner_steps support batches + 1 query batch from that city's
     buffer.
  3. Differentiably adapt a COPY of the global params through the support
     batches, then compute the query loss at the adapted params -- its
     gradient w.r.t. the ORIGINAL global params is this city's meta-gradient
     contribution (federated/maml.py::maml_client_grad).
Cities' meta-gradients are sample-count-weighted and averaged into one
meta-gradient, applied as a single Adam step on the shared global network --
this IS the aggregation step, there is no separate "average weights" pass.

Usage:
    python diagnostics/maml_fedavg.py --base_dir environments_c1_4_6 \\
        --pad_to_true_holdout --rounds 5 --seed 3
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from agents.dqn import DQNAgent, ReplayBuffer
from environments.federated_env import ActionMaskPadder, build_federated_env
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)
from federated.maml import maml_client_grad
from federated.utils import set_seed

logger = logging.getLogger(__name__)


def collect_episode(agent: DQNAgent, env, buffer: ReplayBuffer) -> None:
    """One epsilon-greedy episode against ``env``, using ``agent``'s CURRENT
    network purely as a policy (no gradient step, no touching agent.replay --
    transitions go straight into the per-city ``buffer`` this function is
    given), so multiple cities can share one agent/network object safely.
    Epsilon is recomputed fresh each tick from ``agent.steps_done``, which
    this function advances itself -- the same convention as
    ``DQNAgent.train()``'s own loop, so the exploration schedule decays
    naturally across rounds exactly as it would in ordinary training."""
    obs_dict = env.reset()
    done = False
    while not done:
        eps = agent._current_epsilon()
        actions = agent.act_batch(obs_dict, eps=eps, explore=True)
        agent.steps_done += 1
        next_obs_dict, rewards, dones, _ = env.step(actions)
        for ts_id, o in obs_dict.items():
            r = rewards.get(ts_id, 0.0)
            no = next_obs_dict.get(ts_id, o)
            d = dones.get(ts_id, dones.get("__all__", False))
            buffer.add(o, actions[ts_id], r, no, d, 1)
        obs_dict = next_obs_dict
        done = dones.get("__all__", all(dones.values()) if dones else True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--collect_episodes_per_round", type=int, default=2)
    ap.add_argument("--inner_steps", type=int, default=3,
                     help="Number of differentiable support-batch SGD steps per city per round.")
    ap.add_argument("--inner_lr", type=float, default=3e-4)
    ap.add_argument("--meta_lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--target_update_every", type=int, default=1,
                     help="Sync target network from online network every N rounds.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent = DQNAgent(own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim)
    device = agent.device
    network = agent.q

    envs = {}
    buffers = {}
    for name, cfg in city_configs:
        env = build_federated_env(cfg)
        env = ActionMaskPadder(env, action_dim)
        envs[name] = env
        buffers[name] = ReplayBuffer(capacity=10000)

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

    eval_holdout("Round 0 (random init, before any training)")

    meta_optimizer = torch.optim.Adam(network.parameters(), lr=args.meta_lr)
    min_transitions = args.batch_size * (args.inner_steps + 1)

    for round_idx in range(1, args.rounds + 1):
        grads_accum = None
        total_weight = 0
        query_losses = []

        for name, cfg in city_configs:
            env = envs[name]
            buf = buffers[name]
            for _ in range(args.collect_episodes_per_round):
                collect_episode(agent, env, buf)

            if len(buf) < min_transitions:
                logger.info(
                    "Round %d | %s: only %d transitions (<%d needed), skipping this city this round.",
                    round_idx, name, len(buf), min_transitions,
                )
                continue

            support_batches = [buf.sample(args.batch_size) for _ in range(args.inner_steps)]
            query_batch = buf.sample(args.batch_size)
            global_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            target_state = {k: v.detach().clone() for k, v in agent.q_target.state_dict().items()}

            grad_dict, q_loss = maml_client_grad(
                network, global_state, target_state, support_batches, query_batch,
                inner_lr=args.inner_lr, gamma=agent.gamma, device=device,
            )
            query_losses.append(q_loss)
            weight = args.batch_size  # sample-count weighting, same convention as plain FedAvg
            if grads_accum is None:
                grads_accum = {k: g.clone() * weight for k, g in grad_dict.items()}
            else:
                for k, g in grad_dict.items():
                    grads_accum[k] += g * weight
            total_weight += weight

        if grads_accum is None or total_weight == 0:
            logger.info("Round %d: no city had enough data for a meta-update yet, skipping.", round_idx)
            continue

        meta_optimizer.zero_grad()
        for name, p in network.named_parameters():
            p.grad = (grads_accum[name] / total_weight).clone()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
        meta_optimizer.step()

        if round_idx % args.target_update_every == 0:
            agent.q_target.load_state_dict(network.state_dict())
            agent.q_target.eval()

        mean_q_loss = sum(query_losses) / len(query_losses) if query_losses else float("nan")
        logger.info("Round %d | cities_updated=%d | mean query loss=%.4f", round_idx, len(query_losses), mean_q_loss)
        eval_holdout(f"After round {round_idx}")

    for env in envs.values():
        env.close()
    evaluator.close()


if __name__ == "__main__":
    main()
