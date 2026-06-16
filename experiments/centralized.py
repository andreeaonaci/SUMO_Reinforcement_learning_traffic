"""Centralized training baseline: pool all cities and train one agent."""
import os
import sys
# ensure repo root is on sys.path for local package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import logging
from environments.common import build_env_from_config
from configs.loader import load_cfg
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def load_city_cfgs(base_dir="environments"):
    cfgs = []
    for name in os.listdir(base_dir):
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if os.path.exists(cfg_path):
            cfgs.append(load_cfg(cfg_path))
    return cfgs


def main(args):
    cfgs = load_city_cfgs()
    envs = [lambda c=cfg: build_env_from_config(c) for cfg in cfgs]
    # simple pooled training: cycle through envs
    agent = DQNAgent(obs_dim=cfgs[0].get("obs_dim", 4), action_dim=cfgs[0].get("action_dim", 2))
    episodes = args.episodes
    for ep in range(episodes):
        env = envs[ep % len(envs)]()
        agent.train(env, episodes=1)
        env.close()
    agent.save(args.out)
    logger.info("Centralized training finished, model saved to %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/centralized.pth")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
