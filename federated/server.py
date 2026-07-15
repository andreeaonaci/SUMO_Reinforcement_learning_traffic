"""Federated server — orchestrates rounds of FedAvg."""
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from federated.aggregation import fed_avg

logger = logging.getLogger(__name__)


class FederatedServer:
    """Coordinate clients through multiple rounds of Federated Averaging.

    Each round:
      1. Broadcast the current global weights to every client.
      2. Each client trains locally; its weights are checkpointed to disk
         IMMEDIATELY (see ``client_checkpoint_every``) rather than only
         after every client in the round has finished -- useful when a
         city with many intersections (e.g. 16) takes far longer per
         round than a 1-intersection city, and you don't want to wait on
         the slowest client just to see the fastest one's progress.
      3. Aggregate via FedAvg (weighted by step count).
      4. Optionally evaluate on the holdout city and checkpoint the
         aggregated global model to disk.

    Args:
        global_model:     A ``DQNAgent`` instance that acts as the global model.
        clients:          List of ``FederatedClient`` instances.
        evaluator:        Optional ``HoldoutEvaluator``.  When None, evaluation
                          metrics are recorded as ``None`` in the history.
        checkpoint_dir:   Directory where per-round / per-client ``.pth``
                          files are saved.
        client_checkpoint_every: Save a client's local weights every this
                          many ROUNDS (1 = every round, the default). Set
                          to 0 to disable per-client checkpointing.
    """

    def __init__(
        self,
        global_model,
        clients: List[Any],
        evaluator: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        client_checkpoint_every: int = 1,
    ):
        self.global_model   = global_model
        self.clients        = clients
        self.evaluator      = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.client_checkpoint_every = client_checkpoint_every
        os.makedirs(os.path.join(checkpoint_dir, "clients"), exist_ok=True)

    def run(self, rounds: int, eval_every: int = 1) -> Dict[str, Any]:
        """Execute ``rounds`` rounds of FedAvg.

        Args:
            rounds:     Total number of federated rounds.
            eval_every: Evaluate and checkpoint every this many rounds.

        Returns:
            history dict with keys:
              round, client_samples, eval_reward,
              eval_waiting_time, eval_stopped, eval_action_counts
        """
        history: Dict[str, list] = {
            "round":              [],
            "client_samples":     [],
            "eval_reward":        [],
            "eval_waiting_time":  [],
            "eval_stopped":       [],
            "eval_action_counts": [],
            "eval_q_gaps":        [],
        }

        for r in range(1, rounds + 1):
            logger.info("=== Federated round %d / %d ===", r, rounds)

            global_state  = self.global_model.state_dict()
            updates       = []
            total_samples = 0

            for client in self.clients:
                state_dict, n_samples = client.local_train(global_state)
                updates.append((state_dict, n_samples))
                total_samples += n_samples

                # Save THIS client's local weights right away -- don't
                # wait for every other client in the round to finish.
                # This is what fixes "we're training city_6 but still
                # don't have city_1's weights": city_1's checkpoint is
                # written the moment city_1 finishes, every round.
                if self.client_checkpoint_every and (r % self.client_checkpoint_every == 0):
                    client_ckpt_path = os.path.join(
                        self.checkpoint_dir, "clients", f"{client.name}_round_{r:03d}.pth"
                    )
                    torch.save(state_dict, client_ckpt_path)
                    logger.info(
                        "Client '%s' local checkpoint saved: %s (samples=%d)",
                        client.name, client_ckpt_path, n_samples,
                    )

            # Weighted aggregation
            agg_state = fed_avg(updates)
            self.global_model.load_state_dict(agg_state)

            if r % eval_every == 0:
                total_norm = sum(
                    p.data.norm(2).item() ** 2
                    for p in self.global_model.q.parameters()
                ) ** 0.5
                logger.info(
                    "Round %d | global weight norm=%.6f | total_samples=%d",
                    r, total_norm, total_samples,
                )

                history["round"].append(r)
                history["client_samples"].append(total_samples)

                if self.evaluator:
                    metrics = self.evaluator.evaluate(self.global_model)
                    history["eval_reward"].append(metrics["mean_reward"])
                    history["eval_waiting_time"].append(metrics["mean_waiting_time"])
                    history["eval_stopped"].append(metrics["mean_stopped"])
                    history["eval_action_counts"].append(metrics.get("action_counts"))
                    history["eval_q_gaps"].append(metrics.get("q_gaps"))
                    logger.info(
                        "Round %d | reward=%.4f | waiting_time=%.2fs | stopped=%.1f",
                        r,
                        metrics["mean_reward"],
                        metrics["mean_waiting_time"],
                        metrics["mean_stopped"],
                    )
                else:
                    history["eval_reward"].append(None)
                    history["eval_waiting_time"].append(None)
                    history["eval_stopped"].append(None)
                    history["eval_action_counts"].append(None)
                    history["eval_q_gaps"].append(None)

                # Aggregated global model checkpoint
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"global_round_{r:03d}.pth"
                )
                torch.save(self.global_model.q.state_dict(), ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

        return history
