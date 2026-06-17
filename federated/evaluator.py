"""Evaluator for federated learning - runs global model on holdout city."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class HoldoutEvaluator:
    def __init__(self, env_builder, episodes: int = 1):
        self.env_builder = env_builder
        self.episodes = episodes
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = self.env_builder()
        return self._env

    def evaluate(self, model) -> dict:
        env = self._get_env()
        ep_rewards = []
        ep_waiting_times = []
        ep_stopped = []

        for ep in range(self.episodes):
            try:
                reset_ret = env.reset()
                state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
                done = False
                ep_r = 0.0
                last_info = {}
                while not done:
                    action = model.act(state, explore=False)
                    next_state, reward, done, info = env.step(action)
                    ep_r += float(reward)
                    state = next_state
                    last_info = info
                ep_rewards.append(ep_r)
                ep_waiting_times.append(last_info.get("system_mean_waiting_time", 0.0))
                ep_stopped.append(last_info.get("agents_total_stopped", 0))
            except Exception as e:
                logger.warning("Evaluation episode %d failed: %s", ep, e)

        if not ep_rewards:
            return {"mean_reward": 0.0, "mean_waiting_time": 0.0, "mean_stopped": 0, "episodes": 0}

        result = {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_waiting_time": float(np.mean(ep_waiting_times)),
            "mean_stopped": float(np.mean(ep_stopped)),
            "episodes": len(ep_rewards),
        }
        logger.info("Holdout eval: reward=%.4f, waiting_time=%.2fs, stopped=%.1f",
                    result["mean_reward"], result["mean_waiting_time"], result["mean_stopped"])
        return result

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None