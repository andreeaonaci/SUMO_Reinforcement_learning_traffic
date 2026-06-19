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

from pyfiglet import Figlet

# Creează bannerul
f = Figlet(font="slant", width=200)
text = f.renderText("FederatedTraining")

# Afișează cu culoare cyan
print("\033[96m")
print("=" * 100)
print(text)
print("=" * 100)
print("\033[0m")

logger = logging.getLogger(__name__)


def load_clients(base_dir: str, local_episodes: int):
    """Construiește clienții și deduce dimensiunile obs/action din ei.

    Returnează (clients, target_obs, target_action). Fiecare environment e
    construit o singură dată aici — nu mai există o trecere separată de
    probing pentru deducerea dimensiunilor.
    """
    import numpy as np

    raw_envs = {}  # name -> (env, cfg)
    for name in sorted(os.listdir(base_dir)):
        if name != "city_1":
            continue
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        env = build_env_from_config(cfg)
        raw_envs[name] = (env, cfg)

    # Deduce dimensiunile facand reset pe fiecare env deja construit
    obs_sizes = []
    action_sizes = []
    obs_cache = {}
    for name, (env, cfg) in raw_envs.items():
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        obs_arr = np.array(obs, dtype=object)
        obs_len = int(np.concatenate([np.atleast_1d(x).ravel() for x in obs_arr]).shape[0]) if obs_arr.size > 0 else 0
        obs_sizes.append(obs_len)
        try:
            action_sizes.append(env.action_space.n)
        except Exception:
            action_sizes.append(2)
        obs_cache[name] = obs

    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2

    clients = []
    for name, (env, cfg) in raw_envs.items():
        wrapped = PaddingWrapper(env, target_obs, target_action)
        wrapped.action_space = gym.spaces.Discrete(target_action)
        # primul reset deja s-a facut mai sus pe env-ul "gol";
        # PaddingWrapper.reset() va re-apela reset() la prima rundă de training,
        # ceea ce e ok (SumoEnvironment suportă reset multiplu).

        def make_agent():
            return DQNAgent(obs_dim=target_obs, action_dim=target_action)

        clients.append(FederatedClient(
            name=name,
            env_builder=(lambda w=wrapped: w),
            agent_builder=make_agent,
            local_episodes=local_episodes,
        ))

    return clients, target_obs, target_action

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

    clients, target_obs, target_action = load_clients(base, args.local_episodes)
    if not clients:
        raise RuntimeError("No clients found for federated training")
    logger.info("target_obs=%d, target_action=%d", target_obs, target_action)

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