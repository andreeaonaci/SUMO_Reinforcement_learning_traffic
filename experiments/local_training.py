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


def main(args):
    base = os.path.join("environments")
    cities = []
    for name in sorted(os.listdir(base)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base, name, "config.yaml")
        if os.path.exists(cfg_path):
            cities.append((name, load_cfg(cfg_path)))

    if not cities:
        raise RuntimeError("No training cities found")

    target_obs, target_action = _infer_target_dims([cfg for _, cfg in cities])

    os.makedirs("results", exist_ok=True)
    for name, cfg in cities:
        agent = DQNAgent(obs_dim=target_obs, action_dim=target_action)
        csv_path = os.path.join("results", f"local_{name}.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["episode", "cumulative_reward", "system_mean_waiting_time", "agents_total_stopped"])
            env = build_env_from_config(cfg)
            env = PaddingWrapper(env, target_obs, target_action)
            env.action_space = gym.spaces.Discrete(target_action)
            try:
                for ep in range(args.episodes):
                    reset_ret = env.reset()
                    state = _unwrap_reset(reset_ret)
                    done = False
                    ep_r = 0.0
                    last_info = {}
                    while not done:
                        action = agent.act(state, explore=True)
                        next_state, reward, done, info = env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        agent.train_step()
                        state = next_state
                        ep_r += float(reward)
                        last_info = info
                    mean_wait = last_info.get("system_mean_waiting_time", 0.0)
                    stopped = last_info.get("agents_total_stopped", 0)
                    writer.writerow([ep, ep_r, mean_wait, stopped])
            except Exception:
                logger.exception("Training failed for city %s", name)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        out = os.path.join("results", f"local_{name}.pth")
        agent.save(out)
        logger.info("Saved local model for %s -> %s", name, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
