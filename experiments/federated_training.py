"""Run federated training across multiple city clients."""
import argparse
import logging
import os
import yaml
import gymnasium as gym

from federated.server import FederatedServer
from federated.client import FederatedClient
from environments.common import build_env_from_config, PaddingWrapper
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def make_agent_builder(cfg):
    # agent builder will be created later with unified dims
    return None


def make_env_builder(cfg):
    return lambda: build_env_from_config(cfg)


def load_clients(base_dir: str, local_episodes: int):
    # load client configs
    clients = []
    cfgs = []
    names = []
    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfgs.append(cfg)
        names.append(name)

    # compute target obs/action by instantiating each env once
    obs_sizes = []
    action_sizes = []
    for cfg in cfgs:
        env = build_env_from_config(cfg)
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs = reset_ret[0]
        else:
            obs = reset_ret
        import numpy as _np

        obs_arr = _np.array(obs, dtype=object)
        obs_len = int(_np.concatenate([_np.atleast_1d(x).ravel() for x in obs_arr]).shape[0]) if obs_arr.size > 0 else 0
        obs_sizes.append(obs_len)
        try:
            action_n = env.action_space.n
        except Exception:
            action_n = 2
        action_sizes.append(action_n)
        env.close()
        import time
        time.sleep(2)  # WSL are nevoie de timp sa elibereze portul

    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2

    # build clients with wrapped env builders and agent builders
    for name, cfg in zip(names, cfgs):
        def make_env(name_cfg=cfg):
            def _build():
                e = build_env_from_config(name_cfg)
                e = PaddingWrapper(e, target_obs, target_action)
                e.action_space = gym.spaces.Discrete(target_action)
                return e

            return _build

        def make_agent():
            return DQNAgent(obs_dim=target_obs, action_dim=target_action)

        clients.append(FederatedClient(name=name, env_builder=make_env(), agent_builder=make_agent, local_episodes=local_episodes))

    return clients


def main(args):
    base = os.path.join("environments")
    
    # load_clients apelat O SINGURA DATA
    clients = load_clients(base, args.local_episodes)
    
    if not clients:
        raise RuntimeError("No clients found for federated training")
    
    # global model construit direct din primul client, fara al doilea load_clients
    global_model = clients[0].agent_builder()
    
    server = FederatedServer(global_model=global_model, clients=clients)
    history = server.run(rounds=args.rounds, eval_every=args.eval_every)
        # cleanup clienti
    for c in clients:
        if hasattr(c, 'close'):
            c.close()
    
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
