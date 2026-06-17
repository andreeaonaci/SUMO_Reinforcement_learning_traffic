"""Federated server orchestrating rounds of FedAvg."""
from typing import List, Dict, Any
import logging

from federated.aggregation import fed_avg

logger = logging.getLogger(__name__)


class FederatedServer:
    def __init__(self, global_model, clients: List[Any], evaluator=None):
        self.global_model = global_model
        self.clients = clients
        self.evaluator = evaluator

    def run(self, rounds: int, eval_every: int = 1, device: str = "cpu") -> Dict[str, Any]:
        history = {
            "round": [],
            "client_samples": [],
            "eval_reward": [],
            "eval_waiting_time": [],
            "eval_stopped": [],
        }

        for r in range(1, rounds + 1):
            logger.info("Starting round %d/%d", r, rounds)
            global_state = self.global_model.state_dict()

            updates = []
            total_samples = 0
            for c in self.clients:
                state_dict, n_samples = c.local_train(global_state)
                updates.append((state_dict, n_samples))
                total_samples += n_samples

            agg_state = fed_avg(updates)
            self.global_model.load_state_dict(agg_state)

            if r % eval_every == 0:
                history["round"].append(r)
                history["client_samples"].append(total_samples)

                if self.evaluator:
                    metrics = self.evaluator.evaluate(self.global_model)
                    history["eval_reward"].append(metrics["mean_reward"])
                    history["eval_waiting_time"].append(metrics["mean_waiting_time"])
                    history["eval_stopped"].append(metrics["mean_stopped"])
                    logger.info(
                        "Round %d | reward=%.4f | waiting_time=%.2fs | stopped=%.1f",
                        r, metrics["mean_reward"], metrics["mean_waiting_time"], metrics["mean_stopped"]
                    )
                else:
                    history["eval_reward"].append(None)
                    history["eval_waiting_time"].append(None)
                    history["eval_stopped"].append(None)
                    logger.info("Round %d done. Total samples: %d", r, total_samples)

        return history