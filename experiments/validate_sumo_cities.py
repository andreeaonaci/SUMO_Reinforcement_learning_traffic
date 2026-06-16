"""Validate SUMO environments: reset, 10 random actions, record shapes and counts."""
import os
import sys
import random
import traceback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.loader import load_cfg
from environments.common import build_env_from_config


def main():
    base = os.path.join("environments")
    rows = []
    for name in sorted(os.listdir(base)):
        cfg_path = os.path.join(base, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        cfg = load_cfg(cfg_path)
        print(f"\nTesting {name} -> mode={cfg.get('mode')}\n")
        try:
            env = build_env_from_config(cfg)
        except Exception as e:
            print("Failed to build env:", e)
            traceback.print_exc()
            continue

        try:
            reset_ret = env.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            # determine observation shape
            import numpy as np

            obs_arr = np.array(obs)
            obs_shape = obs_arr.shape
            # action space size
            try:
                action_n = env.action_space.n
            except Exception:
                action_n = None
            # traffic light count
            try:
                tl_count = len(env.ts_ids)
            except Exception:
                try:
                    tl_count = len(env.traffic_signals)
                except Exception:
                    tl_count = None

            rewards = []
            for i in range(10):
                if action_n is None:
                    a = random.randint(0, 1)
                else:
                    a = env.action_space.sample()
                next_obs, r, done, info = env.step(a)
                rewards.append(float(r))
                if done:
                    env.reset()

            print(f"Observation shape: {obs_shape}")
            print(f"Action space: {action_n}")
            print(f"Rewards (10): {rewards}")
            print(f"Traffic lights: {tl_count}")

            rows.append((name, obs_shape, action_n, tl_count))
        except Exception as e:
            print("Error during stepping:", e)
            traceback.print_exc()
        finally:
            try:
                env.close()
            except Exception:
                pass

    # Print summary table
    print("\nSummary")
    print("| City | Observation Shape | Action Space | Traffic Lights |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")


if __name__ == '__main__':
    main()
