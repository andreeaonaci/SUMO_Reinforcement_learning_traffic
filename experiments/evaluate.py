"""Evaluate saved models on holdout city (city_5_holdout)."""
import argparse
import logging
import os
import sys
# ensure repo root is on sys.path for local package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.loader import load_cfg
from environments.common import build_env_from_config, PaddingWrapper
from agents.dqn import DQNAgent
import numpy as np
import gymnasium as gym
import logging
import csv

logger = logging.getLogger(__name__)


def load_holdout():
    cfg_path = os.path.join("environments", "city_5_holdout", "config.yaml")
    return load_cfg(cfg_path)


def evaluate_model(model_path: str, episodes: int = 5):
    cfg = load_holdout()
    # infer target dims from training cities (city_1..city_4)
    base = "environments"
    obs_sizes = []
    action_sizes = []
    for name in [f for f in sorted(__import__("os").listdir(base)) if f.startswith("city_") and f != "city_5_holdout"]:
        ccfg = load_cfg(os.path.join(base, name, "config.yaml"))
        e = build_env_from_config(ccfg)
        reset_ret = e.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        obs_arr = np.array(obs, dtype=object)
        obs_len = int(np.concatenate([np.atleast_1d(x).ravel() for x in obs_arr]).shape[0]) if obs_arr.size > 0 else 0
        obs_sizes.append(obs_len)
        try:
            action_n = e.action_space.n
        except Exception:
            action_n = 2
        action_sizes.append(action_n)
        e.close()

    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2

    env = build_env_from_config(cfg)
    env = PaddingWrapper(env, target_obs, target_action)
    env.action_space = gym.spaces.Discrete(target_action)
    agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
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
    # save CSV
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "evaluation.csv"), "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate_model(args.model, args.episodes)
