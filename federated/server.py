from typing import List, Dict, Any
import logging

from federated.aggregation import fed_avg

logger = logging.getLogger(__name__)


class FederatedServer:
    """Simple federated server orchestrating rounds of FedAvg.

    Attributes:
        global_model: a model instance exposing `state_dict`/`load_state_dict`
        clients: list of client objects with `local_train` API
    """

    def __init__(self, global_model, clients: List[Any], evaluator=None):
        self.global_model = global_model
        self.clients = clients
        self.evaluator = evaluator

    def run(self, rounds: int, eval_every: int = 1, device: str = "cpu") -> Dict[str, Any]:
        history = {"round": [], "eval_reward": []}
        for r in range(1, rounds + 1):
            logger.info("Starting round %d/%d", r, rounds)
            # broadcast
            global_state = self.global_model.state_dict()
            updates = []
            for c in self.clients:
                state_dict, n_samples = c.local_train(global_state)
                updates.append((state_dict, n_samples))

            # aggregate
            agg_state = fed_avg(updates)
            self.global_model.load_state_dict(agg_state)

            # evaluate
            if self.evaluator and (r % eval_every == 0):
                reward = self.evaluator.evaluate(self.global_model)
                history["round"].append(r)
                history["eval_reward"].append(reward)
                logger.info("Round %d evaluation reward: %.4f", r, reward)

        return history
