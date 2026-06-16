"""Run federated training across multiple city clients."""
import argparse
import logging
import os
import sys
# ensure repo root is on sys.path for local package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.loader import load_cfg

from federated.server import FederatedServer
from federated.client import FederatedClient
from environments.common import build_env_from_config
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def make_agent_builder(cfg):
    def builder():
        return DQNAgent(obs_dim=cfg.get("obs_dim", 4), action_dim=cfg.get("action_dim", 2))

    return builder


def make_env_builder(cfg):
    return lambda: build_env_from_config(cfg)


def load_clients(base_dir: str, local_episodes: int):
    clients = []
    for name in os.listdir(base_dir):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = load_cfg(cfg_path)
        clients.append(FederatedClient(name=name, env_builder=make_env_builder(cfg), agent_builder=make_agent_builder(cfg), local_episodes=local_episodes))
    return clients


def main(args):
    base = os.path.join("environments")
    clients = load_clients(base, args.local_episodes)
    # initialize global model from first client's agent
    cfg0 = load_cfg(os.path.join(base, "city_1", "config.yaml"))
    global_model = DQNAgent(obs_dim=cfg0.get("obs_dim", 4), action_dim=cfg0.get("action_dim", 2))
    server = FederatedServer(global_model=global_model, clients=clients)
    history = server.run(rounds=args.rounds, eval_every=args.eval_every)
    os.makedirs("results", exist_ok=True)
    global_model.save(os.path.join("results", "global_fed.pth"))
    logger.info("Federated training finished. History: %s", history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_episodes", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
