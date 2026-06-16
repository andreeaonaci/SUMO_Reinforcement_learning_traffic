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
import csv

logger = logging.getLogger(__name__)


def _unwrap_reset(reset_ret):
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        return reset_ret[0]
    return reset_ret


def _infer_target_dims(cfgs):
    obs_sizes = []
    action_sizes = []
    for cfg in cfgs:
        env = build_env_from_config(cfg)
        try:
            reset_ret = env.reset()
            obs = _unwrap_reset(reset_ret)
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
    target_obs = max(obs_sizes) if obs_sizes else 4
    target_action = max(action_sizes) if action_sizes else 2
    return target_obs, target_action


def load_holdout():
    cfg_path = os.path.join("environments", "city_5_holdout", "config.yaml")
    return load_cfg(cfg_path)


def evaluate_model(model_path: str, episodes: int = 5):
    train_cfgs = []
    base = "environments"
    for name in sorted(os.listdir(base)):
        if not name.startswith("city_"):
            continue
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base, name, "config.yaml")
        if os.path.exists(cfg_path):
            train_cfgs.append(load_cfg(cfg_path))

    if not train_cfgs:
        raise RuntimeError("No training city configs found")

    target_obs, target_action = _infer_target_dims(train_cfgs)

    holdout_cfg = load_holdout()
    env = build_env_from_config(holdout_cfg)
    env = PaddingWrapper(env, target_obs, target_action)
    env.action_space = gym.spaces.Discrete(target_action)
    agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
    agent.load(model_path)

    results = []
    try:
        for _ in range(episodes):
            reset_ret = env.reset()
            state = _unwrap_reset(reset_ret)
            done = False
            ep_r = 0.0
            while not done:
                action = agent.act(state, explore=False)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                ep_r += float(reward)
            results.append(ep_r)
    except Exception:
        logger.exception("Error during evaluation")
    finally:
        try:
            env.close()
        except Exception:
            pass

    logger.info("Evaluation rewards: %s", results)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "evaluation.csv"), "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(results):
            writer.writerow([i, r])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate_model(args.model, args.episodes)
