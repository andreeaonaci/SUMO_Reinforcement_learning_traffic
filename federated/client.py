from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, name: str, env_builder, agent_builder, local_episodes: int = 5):
        self.name = name
        self.agent_builder = agent_builder
        self.local_episodes = local_episodes
        # construieste env O SINGURA DATA, nu la fiecare local_train
        self._env = env_builder()

    def local_train(self, global_state):
        logger.info("Client %s starting local training for %d episodes", self.name, self.local_episodes)
        agent = self.agent_builder()
        agent.load_state_dict(global_state)
        # refoloseste acelasi env intre runde
        try:
            state_dict, n_samples = agent.train(self._env, episodes=self.local_episodes)
        except Exception:
            # daca env-ul s-a stricat, incearca sa-l inchida si ridica eroarea
            try:
                self._env.close()
            except Exception:
                pass
            raise

        logger.info("Client %s finished local training: samples=%d", self.name, n_samples)
        return state_dict, n_samples

    def close(self):
        if hasattr(self, '_env'):
            try:
                self._env.close()
            except Exception:
                pass
