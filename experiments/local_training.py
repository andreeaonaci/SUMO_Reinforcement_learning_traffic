"""Train local models independently per city (Scenario A)."""
import argparse
import logging
from configs.loader import load_cfg
import os
from environments.common import build_env_from_config
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def main(args):
    base = os.path.join("environments")
    for name in os.listdir(base):
        cfg_path = os.path.join(base, name, "config.yaml")
        if not os.path.exists(cfg_path) or name == "city_5_holdout":
            continue
        cfg = load_cfg(cfg_path)
        env = build_env_from_config(cfg)
        agent = DQNAgent(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2))
        agent.train(env, episodes=args.episodes)
        out = os.path.join("results", f"local_{name}.pth")
        os.makedirs("results", exist_ok=True)
        agent.save(out)
        logger.info("Saved local model for %s -> %s", name, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
