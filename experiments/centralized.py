"""Centralized training baseline: pool all cities and train one agent."""
import os
import sys
# ensure repo root is on sys.path for local package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import logging
from environments.common import build_env_from_config, PaddingWrapper
from configs.loader import load_cfg
from agents.dqn import DQNAgent
import gymnasium as gym
import numpy as np
import csv

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
    # compute target dims
    obs_sizes = []
    action_sizes = []
    for cfg in cfgs:
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

    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2

    agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
    episodes = args.episodes
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "centralized.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["episode", "cumulative_reward", "system_mean_waiting_time", "agents_total_stopped"])
        for ep in range(episodes):
            cfg = cfgs[ep % len(cfgs)]
            env = build_env_from_config(cfg)
            env = PaddingWrapper(env, target_obs, target_action)
            env.action_space = gym.spaces.Discrete(target_action)
            # train one episode
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
                agent.replay.add(state, int(a), float(r), state, float(done))
                agent.optimize()
            mean_wait = last_info.get("system_mean_waiting_time", 0.0)
            stopped = last_info.get("agents_total_stopped", 0)
            writer.writerow([ep, ep_r, mean_wait, stopped])
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
