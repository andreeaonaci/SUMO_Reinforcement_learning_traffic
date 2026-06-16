from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FederatedClient:
    """Client that performs local RL training and returns model updates.

    The client expects an `agent_factory` that given a model_state_dict returns
    an object with `load_state_dict`, `train_on_env(state_dict, episodes)`
    or a `train` method. For simplicity, `train_on_env` returns (state_dict, n_steps).
    """

    def __init__(self, name: str, env_builder, agent_builder, local_episodes: int = 5):
        self.name = name
        self.env_builder = env_builder
        self.agent_builder = agent_builder
        self.local_episodes = local_episodes

    def local_train(self, global_state: Dict) -> Tuple[Dict, int]:
        logger.info("Client %s starting local training for %d episodes", self.name, self.local_episodes)
        agent = self.agent_builder()
        agent.load_state_dict(global_state)
        env = self.env_builder()
        # agent.train should return number of samples/steps used for weighting
        try:
            state_dict, n_samples = agent.train(env, episodes=self.local_episodes)
        finally:
            # allow env cleanup
            if hasattr(env, "close"):
                env.close()

        logger.info("Client %s finished local training: samples=%d", self.name, n_samples)
        return state_dict, n_samples
