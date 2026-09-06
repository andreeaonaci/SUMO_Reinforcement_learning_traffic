"""Item 25 (fidings/divergence_investigation.md queue, the last of the six
"genuinely different paradigm" items): evolution strategies -- a population-
based, gradient-free alternative to TD-learning. No replay buffer, no Bellman
backup, no bootstrapped value estimate to become overconfident about --
sidesteps the confident-lock-in mechanism class (sec 32-34/51-57) entirely
rather than patching around it, the most radical departure of the six.

Design: reuses NeighborAttentionQNetwork/DQNAgent purely as a STATELESS POLICY
CONTAINER -- no .train()/.optimize() call anywhere in this script. A policy is
just a state_dict; "acting" is DQNAgent.act(obs, explore=False), this
project's standard pure-argmax eval convention (sec 35), against whatever
weights are loaded. Optimization is OpenAI-ES (Salimans et al. 2017): each
generation, sample N perturbations of the current mean weights, evaluate each
on one training city for one full episode, and move the mean a step in the
direction of the reward-weighted (rank-normalized, Wierstra et al. 2014 style
fitness shaping for robustness to reward scale/outliers) perturbations -- no
gradients through the network at all.

Scope of this first pilot (deliberately small, matching this project's own
"short pilot before multi-seed investment" pattern used for the PPO/
Munchausen algorithm swap, sec 73): small population/generation count, one
training-city rollout per individual (not all three per individual, to keep
wall-clock bounded), single seed. This is a screen for "is this worth
pursuing further," not a claim of a finding either way.

Usage:
    python diagnostics/evolution_strategies.py --base_dir environments_c1_4_6 \\
        --pad_to_true_holdout --population 4 --generations 3 --seed 3
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from agents.dqn import DQNAgent
from environments.federated_env import ActionMaskPadder, build_federated_env
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)

logger = logging.getLogger(__name__)


def _flatten(state_dict):
    keys = list(state_dict.keys())
    flat = torch.cat([state_dict[k].reshape(-1).float() for k in keys])
    shapes = [(k, tuple(state_dict[k].shape)) for k in keys]
    return flat, shapes


def _unflatten(flat: torch.Tensor, shapes):
    out = {}
    i = 0
    for k, shape in shapes:
        n = 1
        for d in shape:
            n *= d
        out[k] = flat[i:i + n].reshape(shape)
        i += n
    return out


def rollout_reward(state_dict, env, agent_kwargs) -> float:
    """One full episode, deterministic argmax policy (this project's standard
    eval convention, sec 35), total reward summed across every
    intersection/tick. No training, no replay buffer -- pure inference."""
    agent = DQNAgent(**agent_kwargs)
    agent.load_state_dict(state_dict)
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]
    done = False
    total_reward = 0.0
    while not done:
        actions = {ts_id: agent.act(o, explore=False) for ts_id, o in obs_dict.items()}
        obs_dict, rewards, dones, _ = env.step(actions)
        total_reward += float(sum(rewards.values()))
        done = dones.get("__all__", all(dones.values()) if dones else True)
    return total_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.02,
                     help="Perturbation std in parameter space.")
    ap.add_argument("--lr", type=float, default=0.02,
                     help="ES step size (OpenAI-ES convention: update magnitude ~ lr/sigma).")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent_kwargs = dict(own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim)

    city_envs = []
    for name, cfg in city_configs:
        env = build_federated_env(cfg)
        env = ActionMaskPadder(env, action_dim)
        city_envs.append((name, env))

    base_agent = DQNAgent(**agent_kwargs)
    theta_flat, shapes = _flatten(base_agent.state_dict())
    n_params = theta_flat.numel()
    logger.info(
        "Policy has %d parameters, %d training cities: %s",
        n_params, len(city_envs), [n for n, _ in city_envs],
    )

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim, episodes=args.eval_episodes,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    def eval_theta(flat, label):
        sd = _unflatten(flat, shapes)
        agent = DQNAgent(**agent_kwargs)
        agent.load_state_dict(sd)
        result = evaluator.evaluate(agent)
        logger.info(
            "%s: mean_reward=%.2f std_reward=%.2f", label, result["mean_reward"], result["std_reward"],
        )
        return result

    eval_theta(theta_flat, "Generation 0 (initial random weights)")

    for gen in range(1, args.generations + 1):
        noise = rng.standard_normal((args.population, n_params)).astype(np.float32)
        rewards = np.zeros(args.population, dtype=np.float64)
        for i in range(args.population):
            perturbed_flat = theta_flat + args.sigma * torch.from_numpy(noise[i])
            sd = _unflatten(perturbed_flat, shapes)
            city_name, env = city_envs[i % len(city_envs)]
            r = rollout_reward(sd, env, agent_kwargs)
            rewards[i] = r
            logger.info("  gen=%d individual=%d city=%s reward=%.2f", gen, i, city_name, r)

        if args.population > 1:
            ranks = np.argsort(np.argsort(rewards))
            fitness = (ranks / (args.population - 1)) - 0.5
        else:
            fitness = np.zeros(1)

        update = (args.lr / (args.population * args.sigma)) * (noise.T @ fitness)
        theta_flat = theta_flat + torch.from_numpy(update.astype(np.float32))

        logger.info(
            "Generation %d done: rewards=%s mean=%.2f",
            gen, [round(r, 1) for r in rewards], rewards.mean(),
        )
        eval_theta(theta_flat, f"Generation {gen} (updated mean)")

    for _, env in city_envs:
        env.close()
    evaluator.close()


if __name__ == "__main__":
    main()
