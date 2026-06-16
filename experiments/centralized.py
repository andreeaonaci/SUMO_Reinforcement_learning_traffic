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


def _unwrap_reset(reset_ret):
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        return reset_ret[0]
    return reset_ret


def _infer_target_dims(cfgs):
    """Infer obs/action dims from configs without starting SUMO."""
    from sumo_rl.environment.env import SumoEnvironment
    import sumolib
    import traci

    obs_sizes = []
    action_sizes = []

    for cfg in cfgs:
        net_file = cfg.get("net_file")
        if net_file is None:
            obs_sizes.append(cfg.get("obs_dim", 4))
            action_sizes.append(cfg.get("action_dim", 2))
            continue

        # citeste ts_ids si fazele direct din fisierul .net.xml, fara SUMO
        try:
            net = sumolib.net.readNet(net_file, withInternal=False)
            tls_list = net.getTrafficLights()
            
            # numara fazele verzi per intersectie
            max_green_phases = 0
            for tls in tls_list:
                programs = tls.getPrograms()
                for prog in programs.values():
                    phases = prog.getPhases()
                    green = sum(1 for p in phases if 'G' in p.state or 'g' in p.state)
                    max_green_phases = max(max_green_phases, green)
            
            n_tls = len(tls_list)
            # obs default: (num_green_phases + 1 + num_lanes) per ts
            # folosim o aproximare conservatoare
            obs_sizes.append(cfg.get("obs_dim", max(4, n_tls * 10)))
            action_sizes.append(cfg.get("action_dim", max(2, max_green_phases)))
        except Exception as e:
            logger.warning("Could not read net file %s: %s", net_file, e)
            obs_sizes.append(cfg.get("obs_dim", 4))
            action_sizes.append(cfg.get("action_dim", 2))

    return max(obs_sizes) if obs_sizes else 4, max(action_sizes) if action_sizes else 2

def load_city_cfgs(base_dir="environments"):
    cfgs = []
    for name in sorted(os.listdir(base_dir)):
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if os.path.exists(cfg_path):
            cfgs.append(load_cfg(cfg_path))
    return cfgs


def main(args):
    cfgs = load_city_cfgs()
    if not cfgs:
        raise RuntimeError("No city configs found in environments/*/config.yaml")

    target_obs, target_action = _infer_target_dims(cfgs)

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
            try:
                reset_ret = env.reset()
                state = _unwrap_reset(reset_ret)
                done = False
                ep_r = 0.0
                last_info = {}
                while not done:
                    action = agent.act(state, explore=True)
                    state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, state, done)
                    agent.train_step()
                    ep_r += float(reward)
                    last_info = info
                mean_wait = last_info.get("system_mean_waiting_time", 0.0)
                stopped = last_info.get("agents_total_stopped", 0)
                writer.writerow([ep, ep_r, mean_wait, stopped])
            except Exception:
                logger.exception("Error during centralized training episode %s", ep)
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    agent.save(args.out)
    logger.info("Centralized training finished, model saved to %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/centralized.pth")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
