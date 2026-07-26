"""Evaluator for federated learning - runs the global (shared) model on
every intersection of a holdout city simultaneously."""
import csv
import json
import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


class HoldoutEvaluator:
    """Evaluates the shared policy on a holdout city's intersections."""

    def __init__(
        self,
        env_builder,
        episodes: int = 1,
        output_dir: str | None = None,
        eval_seeds: int = 3,
        include_baselines: bool = False,
    ):
        self.env_builder = env_builder
        self.episodes = episodes
        self.output_dir = output_dir
        self.eval_seeds = max(1, eval_seeds)
        self.include_baselines = include_baselines
        self._env = None
        self._round_robin_offset = {}
        self.last_summary = {}

    def _get_env(self):
        if self._env is None:
            self._env = self.env_builder()
        return self._env

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _extract_metric(self, info: dict, candidates: list[str], default: float = 0.0) -> float:
        for key in candidates:
            if key in info and info[key] is not None:
                return float(info[key])
        return float(default)

    def _policy_action(self, policy_name: str, ts_id: str | None, obs: dict, model) -> int:
        if policy_name == "trained":
            return int(model.act(obs, explore=False))

        valid = np.flatnonzero(obs["action_mask"] > 0.5)
        if len(valid) == 0:
            return 0

        if policy_name == "always_zero":
            return int(valid[0])

        if policy_name == "random":
            return int(np.random.choice(valid))

        if policy_name == "round_robin":
            key = ts_id or "__default__"
            idx = self._round_robin_offset.get(key, 0)
            choice = int(valid[idx % len(valid)])
            self._round_robin_offset[key] = (idx + 1) % len(valid)
            return choice

        if policy_name == "fixed_time":
            return int(valid[0])

        return int(valid[0])

    def _evaluate_policy(self, policy_name: str, model=None, seed_offset: int = 0, episode_count: int | None = None) -> dict:
        env = self._get_env()
        if hasattr(env, "fixed_ts"):
            env.fixed_ts = (policy_name == "fixed_time")
        self._round_robin_offset.clear()
        ep_rewards = []
        ep_waiting_times = []
        ep_stopped = []
        ep_arrived = []
        ep_queue_lengths = []
        ep_action_counts = []
        ep_action_logs = []
        ep_q_gaps = []

        episode_count = self.episodes if episode_count is None else max(1, episode_count)
        for ep in range(episode_count):
            try:
                seed = seed_offset + ep
                self._set_seed(seed)
                obs_dict = env.reset()
                if not isinstance(obs_dict, dict):
                    obs_dict = {"__single__": obs_dict}

                done = False
                ep_r = 0.0
                last_info = {}
                action_counts: dict = {}
                action_log: list = []
                q_gaps: dict = {}

                while not done:
                    actions = {
                        ts_id: self._policy_action(policy_name, ts_id, o, model)
                        for ts_id, o in obs_dict.items()
                    }

                    if len(action_log) % 10 == 0:
                        for ts_id, o in obs_dict.items():
                            try:
                                qvals = model.q_values(o) if policy_name == "trained" else None
                                if qvals is not None:
                                    valid = qvals[~np.isnan(qvals)]
                                    if len(valid) >= 2:
                                        top2 = np.sort(valid)[-2:]
                                        gap = float(top2[1] - top2[0])
                                        q_gaps.setdefault(ts_id, []).append(gap)
                            except Exception:
                                pass

                    if len(action_log) < 200:
                        action_log.append(dict(actions))
                    for ts_id, a in actions.items():
                        action_counts.setdefault(ts_id, {})
                        action_counts[ts_id][a] = action_counts[ts_id].get(a, 0) + 1

                    if len(actions) == 1 and "__single__" in actions:
                        next_obs, reward, done, info = env.step(actions["__single__"])
                        next_obs_dict = {"__single__": next_obs}
                        rewards = {"__single__": reward}
                        dones = {"__all__": done}
                    else:
                        next_obs_dict, rewards, dones, info = env.step(actions)

                    ep_r += float(sum(rewards.values()))
                    obs_dict = next_obs_dict
                    done = dones.get("__all__", all(dones.values()) if dones else True)
                    last_info = info or {}

                mean_q_gaps = {ts_id: float(np.mean(gaps)) for ts_id, gaps in q_gaps.items()}
                logger.info("Eval[%s] ep=%d action distribution per ts: %s", policy_name, ep, action_counts)
                logger.info("Eval[%s] ep=%d mean |Q(top1)-Q(top2)| per ts: %s", policy_name, ep, mean_q_gaps)

                ep_rewards.append(ep_r)
                ep_waiting_times.append(self._extract_metric(last_info, ["system_mean_waiting_time", "mean_waiting_time", "waiting_time"], 0.0))
                ep_stopped.append(int(self._extract_metric(last_info, ["agents_total_stopped", "stopped"], 0.0)))
                ep_arrived.append(int(self._extract_metric(last_info, ["system_total_arrived", "total_arrived"], 0.0)))
                ep_queue_lengths.append(self._extract_metric(last_info, ["system_mean_queue_length", "mean_queue_length", "queue_length"], 0.0))
                ep_action_counts.append(action_counts)
                ep_action_logs.append(action_log)
                ep_q_gaps.append(mean_q_gaps)
            except Exception as e:
                logger.warning("Evaluation episode %d for policy '%s' failed: %s", ep, policy_name, e)

        if not ep_rewards:
            return {
                "policy": policy_name,
                "mean_reward": 0.0,
                "mean_waiting_time": 0.0,
                "mean_stopped": 0,
                "mean_arrived": 0,
                "mean_queue_length": 0.0,
                "episodes": 0,
                "action_counts": [],
                "action_log": [],
                "q_gaps": [],
            }

        result = {
            "policy": policy_name,
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_waiting_time": float(np.mean(ep_waiting_times)),
            "mean_stopped": float(np.mean(ep_stopped)),
            "mean_arrived": float(np.mean(ep_arrived)),
            "mean_queue_length": float(np.mean(ep_queue_lengths)),
            "episodes": len(ep_rewards),
            "action_counts": ep_action_counts,
            "action_log": ep_action_logs,
            "q_gaps": ep_q_gaps,
        }
        logger.info(
            "Holdout eval[%s]: reward=%.4f, waiting_time=%.2fs, stopped=%.1f, arrived=%.1f, queue=%.2f",
            policy_name,
            result["mean_reward"],
            result["mean_waiting_time"],
            result["mean_stopped"],
            result["mean_arrived"],
            result["mean_queue_length"],
        )
        return result

    def _persist_summary(self, summary: dict) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        summary_path = os.path.join(self.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(self.output_dir, "eval_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "episodes", "mean_reward", "mean_waiting_time", "mean_stopped", "mean_arrived", "mean_queue_length"])
            for policy_name, metrics in summary.items():
                writer.writerow([
                    policy_name,
                    metrics.get("episodes", 0),
                    metrics.get("mean_reward", 0.0),
                    metrics.get("mean_waiting_time", 0.0),
                    metrics.get("mean_stopped", 0),
                    metrics.get("mean_arrived", 0),
                    metrics.get("mean_queue_length", 0.0),
                ])

    def evaluate(self, model) -> dict:
        summary = {}
        trained_result = self._evaluate_policy("trained", model=model, seed_offset=0, episode_count=self.episodes)
        summary["trained"] = trained_result

        if self.include_baselines:
            baseline_count = max(1, self.eval_seeds)
            for policy_name in ["always_zero", "random", "round_robin", "fixed_time"]:
                summary[policy_name] = self._evaluate_policy(
                    policy_name,
                    model=None,
                    seed_offset=self.eval_seeds,
                    episode_count=baseline_count,
                )

        self.last_summary = summary
        self._persist_summary(summary)
        return trained_result

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
