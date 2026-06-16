"""
Centralized training baseline:
Trains one DQN agent over all cities sequentially (no federation).
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
import torch

# ensure repo root is on path FIRST (critical)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environments.common import build_env_from_config
from configs.loader import load_cfg
from agents.dqn import DQNAgent


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_city_cfgs(base_dir="environments"):
    cfgs = []
    city_names = sorted(os.listdir(base_dir))

    for name in city_names:
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if os.path.exists(cfg_path):
            cfgs.append(load_cfg(cfg_path))

    if len(cfgs) == 0:
        raise RuntimeError("No city configs found in environments/*/config.yaml")

    return cfgs


def main(args):
    set_seed(args.seed)

    cfgs = load_city_cfgs()

    obs_dim = cfgs[0].get("obs_dim", cfgs[0].get("state_dim", 21))
    action_dim = cfgs[0].get("action_dim", 4)

    agent = DQNAgent(obs_dim=obs_dim, action_dim=action_dim)

    for ep in range(args.episodes):
        total_reward = 0

        for cfg in cfgs:
            env = build_env_from_config(cfg)

            state, _ = env.reset()
            done = False

            while not done:
                action = agent.act(state, explore=True)
                next_state, reward, done, _, info = env.step(action)

                agent.remember(state, action, reward, next_state, done)
                agent.train_step()

                state = next_state
                total_reward += reward

            env.close()

        logger.info(f"[Centralized] Episode {ep} | Reward: {total_reward}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent.save(args.out)

    logger.info("Centralized training finished. Model saved to %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/centralized.pth")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)