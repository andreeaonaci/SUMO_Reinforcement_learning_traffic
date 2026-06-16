"""Train local models independently per city (Scenario A)."""
import argparse
import logging
import os
import sys
# ensure repo root is on sys.path for local package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.loader import load_cfg
from environments.common import build_env_from_config, PaddingWrapper
from agents.dqn import DQNAgent
import gymnasium as gym
import csv
import numpy as np

logger = logging.getLogger(__name__)


def main(args):
    base = os.path.join("environments")
    # compute target dims across clients
    obs_sizes = []
    action_sizes = []
    cities = []
    for name in sorted(os.listdir(base)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = load_cfg(cfg_path)
        env = build_env_from_config(cfg)
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        obs_arr = np.array(obs, dtype=object)
        obs_len = int(np.concatenate([np.atleast_1d(x).ravel() for x in obs_arr]).shape[0]) if obs_arr.size > 0 else 0
        obs_sizes.append(obs_len)
        try:
            action_n = env.action_space.n
        except Exception:
            action_n = 2
        action_sizes.append(action_n)
        env.close()
        cities.append((name, cfg))

    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2

    for name, cfg in cities:
        env = build_env_from_config(cfg)
        env = PaddingWrapper(env, target_obs, target_action)
        env.action_space = gym.spaces.Discrete(target_action)
        agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
        # metrics csv
        os.makedirs("results", exist_ok=True)
        csv_path = os.path.join("results", f"local_{name}.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["episode", "cumulative_reward", "system_mean_waiting_time", "agents_total_stopped"])
            for ep in range(args.episodes):
                # train one episode and capture metrics
                reset_ret = env.reset()
                state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
                done = False
                ep_r = 0.0
                last_info = {}
                while not done:
                    a = agent.select_action(state)
                    state, r, done, info = env.step(a)
                    ep_r += float(r)
                    last_info = info
                    # step training internally (agent maintains replay)
                    agent.replay.add(state, int(a), float(r), state, float(done))
                    agent.optimize()
                mean_wait = last_info.get("system_mean_waiting_time", 0.0)
                stopped = last_info.get("agents_total_stopped", 0)
                writer.writerow([ep, ep_r, mean_wait, stopped])
        out = os.path.join("results", f"local_{name}.pth")
        agent.save(out)
        logger.info("Saved local model for %s -> %s", name, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
