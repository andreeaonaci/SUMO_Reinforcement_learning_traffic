"""Run minimal federated flow: wrap envs to uniform dims, train 1 episode per city, aggregate."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

if "SUMO_HOME" not in os.environ:
    for candidate in ("/usr/share/sumo", "/usr/local/share/sumo"):
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            break

if "SUMO_HOME" in os.environ:
    sumo_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if sumo_tools not in sys.path:
        sys.path.insert(0, sumo_tools)
from configs.loader import load_cfg
from environments.common import build_env_from_config, PaddingWrapper
from agents.dqn import DQNAgent
from federated.aggregation import fed_avg


def main():
    base = os.path.join("environments")
    cities = [f for f in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, f)) and f.startswith("city_")][:4]
    envs = {}
    infos = {}
    for name in cities:
        cfg = load_cfg(os.path.join(base, name, "config.yaml"))
        env = build_env_from_config(cfg)
        obs, _ = env.reset()
        obs_dim = None
        try:
            import numpy as np

            obs_arr = np.array(obs)
            obs_dim = obs_arr.shape[0]
        except Exception:
            obs_dim = len(obs)
        try:
            action_n = env.action_space.n
        except Exception:
            action_n = 2
        infos[name] = (obs_dim, action_n)
        env.close()

    # compute targets
    target_obs = max(v[0] for v in infos.values())
    target_action = max(v[1] for v in infos.values())

    print("Target obs dim:", target_obs, "target action_n:", target_action)

    agents_state = []
    for name in cities:
        cfg = load_cfg(os.path.join(base, name, "config.yaml"))
        env = build_env_from_config(cfg)
        # wrap
        env = PaddingWrapper(env, target_obs, target_action)
        # expose action_space
        try:
            import gymnasium as gym
            env.action_space = gym.spaces.Discrete(target_action)
        except Exception:
            class _D:
                def __init__(self, n):
                    self.n = n
            env.action_space = _D(target_action)

        agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
        state, steps = agent.train(env, episodes=1)
        print(f"Trained {name}: steps={steps}, state keys={list(state.keys())}")
        agents_state.append((state, steps))
        env.close()

    agg = fed_avg(agents_state)
    print("Aggregated state keys:", list(agg.keys()))


if __name__ == '__main__':
    main()
