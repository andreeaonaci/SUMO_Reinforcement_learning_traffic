"""Evaluate saved models on holdout city (city_5_holdout)."""
import argparse
import logging
from configs.loader import load_cfg
import os
from environments.common import build_env_from_config
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def load_holdout():
    cfg_path = os.path.join("environments", "city_5_holdout", "config.yaml")
    return load_cfg(cfg_path)


def evaluate_model(model_path: str, episodes: int = 5):
    cfg = load_holdout()
    env = build_env_from_config(cfg)
    agent = DQNAgent(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2))
    agent.load(model_path)
    # run episodes and collect rewards
    rewards = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.select_action(s)
            s, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
    logger.info("Evaluation rewards: %s", rewards)
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate_model(args.model, args.episodes)
