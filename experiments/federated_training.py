"""Run federated training across multiple city clients."""
import argparse
import logging
import os
import yaml
import json
import gymnasium as gym

from federated.server import FederatedServer
from federated.client import FederatedClient
from federated.evaluator import HoldoutEvaluator
from environments.common import build_env_from_config, PaddingWrapper
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def load_clients(base_dir: str, local_episodes: int, target_obs: int, target_action: int):
    clients = []
    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        def make_env(c=cfg):
            def _build():
                e = build_env_from_config(c)
                e = PaddingWrapper(e, target_obs, target_action)
                e.action_space = gym.spaces.Discrete(target_action)
                return e
            return _build

        def make_agent():
            return DQNAgent(obs_dim=target_obs, action_dim=target_action)

        clients.append(FederatedClient(
            name=name,
            env_builder=make_env(),
            agent_builder=make_agent,
            local_episodes=local_episodes,
        ))

    return clients


def infer_dims(base_dir: str):
    """Infer target obs/action dims by instantiating each city env once."""
    import time
    import numpy as np

    obs_sizes = []
    action_sizes = []

    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        env = build_env_from_config(cfg)
        try:
            reset_ret = env.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            obs_arr = np.array(obs, dtype=object)
            obs_len = int(np.concatenate([np.atleast_1d(x).ravel() for x in obs_arr]).shape[0]) if obs_arr.size > 0 else 0
            obs_sizes.append(obs_len)
            try:
                action_sizes.append(env.action_space.n)
            except Exception:
                action_sizes.append(2)
        finally:
            try:
                env.close()
            except Exception:
                pass
            time.sleep(2)

    return max(obs_sizes) if obs_sizes else 4, max(action_sizes) if action_sizes else 2


def make_holdout_evaluator(base_dir: str, target_obs: int, target_action: int, episodes: int = 1):
    cfg_path = os.path.join(base_dir, "city_5_holdout", "config.yaml")
    if not os.path.exists(cfg_path):
        logger.warning("No holdout city found at %s, skipping evaluator", cfg_path)
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    def build_holdout_env():
        e = build_env_from_config(cfg)
        e = PaddingWrapper(e, target_obs, target_action)
        e.action_space = gym.spaces.Discrete(target_action)
        return e

    return HoldoutEvaluator(env_builder=build_holdout_env, episodes=episodes)


def main(args):
    base = "environments"

    logger.info("Inferring obs/action dims from training cities...")
    target_obs, target_action = infer_dims(base)
    logger.info("target_obs=%d, target_action=%d", target_obs, target_action)

    clients = load_clients(base, args.local_episodes, target_obs, target_action)
    if not clients:
        raise RuntimeError("No clients found for federated training")

    global_model = DQNAgent(obs_dim=target_obs, action_dim=target_action)

    evaluator = make_holdout_evaluator(base, target_obs, target_action, episodes=args.eval_episodes)

    server = FederatedServer(global_model=global_model, clients=clients, evaluator=evaluator)
    history = server.run(rounds=args.rounds, eval_every=args.eval_every)

    # cleanup
    for c in clients:
        c.close()
    if evaluator:
        evaluator.close()

    os.makedirs("results", exist_ok=True)
    global_model.save(os.path.join("results", "global_fed.pth"))

    history_path = os.path.join("results", "federated_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Federated training finished.")
    logger.info("History saved to %s", history_path)
    logger.info("History: %s", history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_episodes", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)