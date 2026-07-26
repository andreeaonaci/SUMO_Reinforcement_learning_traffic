"""Federated server -- orchestrates rounds of pluggable-strategy aggregation.

No aggregation-specific logic lives here beyond: build a ClientRoundInfo per
client, hand the batch to the configured strategy, aggregate the
result. Swapping strategies is a config change (`aggregation_strategy: ...`),
never a code change in this file.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from federated.aggregation import aggregate_round
from federated.aggregation_strategies import (
    BaseAggregationStrategy,
    ClientRoundInfo,
    GradientSurvivalStrategy,
    build_aggregation_strategy,
)

logger = logging.getLogger(__name__)


class FederatedServer:
    """Coordinate clients through multiple rounds of pluggable-strategy FedAvg."""

    def __init__(
        self,
        global_model,
        clients: List[Any],
        evaluator: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        aggregation_strategy: str = "fedavg",
        aggregation_config: Optional[Dict[str, Any]] = None,
        use_masked_head: bool = True,
    ):
        self.global_model = global_model
        self.clients = clients
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.use_masked_head = use_masked_head

        self.strategy: BaseAggregationStrategy = build_aggregation_strategy(
            aggregation_strategy, aggregation_config
        )

        logger.info(
            "Aggregation strategy: %s  config=%s",
            type(self.strategy).__name__,
            aggregation_config or {},
        )
        logger.info(
            "Masked head aggregation: %s",
            self.use_masked_head,
        )

        # Per-client history needed to compute deltas each round.
        self._previous_client_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self._previous_loss: Dict[str, float] = {}
        self._previous_global_state: Optional[Dict[str, torch.Tensor]] = None
        self._global_gradient: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Client call compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def _call_local_train(client: Any, global_state: Dict[str, torch.Tensor]):
        """Accept clients whose local_train returns either a 2-, 3-, or 4-tuple."""
        result = client.local_train(global_state)

        if len(result) == 4:
            state_dict, n_samples, mean_loss, action_counts = result
        elif len(result) == 3:
            state_dict, n_samples, mean_loss = result
            action_counts = None
        else:
            state_dict, n_samples = result
            mean_loss = None
            action_counts = None

        return state_dict, n_samples, mean_loss, action_counts

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, rounds: int, eval_every: int = 1) -> Dict[str, Any]:
        history: Dict[str, list] = {
            "round": [],
            "client_samples": [],
            "eval_reward": [],
            "eval_waiting_time": [],
            "eval_stopped": [],
        }

        for r in range(1, rounds + 1):
            logger.info("=== Federated round %d / %d ===", r, rounds)

            global_state_before = self.global_model.state_dict()

            client_states: Dict[str, Dict[str, torch.Tensor]] = {}
            infos: List[ClientRoundInfo] = []
            total_samples = 0

            client_action_counts: Dict[str, Optional[Dict[int, int]]] = {}

            for client in self.clients:
                cid = getattr(client, "name", repr(client))

                (
                    state_dict,
                    n_samples,
                    mean_loss,
                    action_counts,
                ) = self._call_local_train(client, global_state_before)

                client_states[cid] = state_dict
                client_action_counts[cid] = action_counts
                total_samples += n_samples

                client_gradient = self.strategy.compute_pseudo_gradient(
                    state_dict,
                    global_state_before,
                )

                infos.append(
                    ClientRoundInfo(
                        client_id=cid,
                        num_samples=n_samples,
                        client_state=state_dict,
                        previous_client_state=self._previous_client_state.get(cid),
                        global_state=global_state_before,
                        previous_global_state=self._previous_global_state,
                        client_gradient=client_gradient,
                        previous_gradient=self.strategy.get_state(cid).get(
                            "_last_client_gradient"
                        ),
                        global_gradient=self._global_gradient,
                        local_loss=mean_loss,
                        previous_loss=self._previous_loss.get(cid),
                        round_num=r,
                    )
                )

                # Persist this round's values for next round.
                self._previous_client_state[cid] = state_dict

                if mean_loss is not None:
                    self._previous_loss[cid] = mean_loss

                if client_gradient is not None:
                    self.strategy.get_state(cid)[
                        "_last_client_gradient"
                    ] = client_gradient

            # ----------------------------------------------------------
            # Compute aggregation weights
            # ----------------------------------------------------------

            weights = self.strategy.compute_weights(infos)
            ordered_ids = [info.client_id for info in infos]

            agg_state = aggregate_round(
                state_dicts=[client_states[cid] for cid in ordered_ids],
                base_weights=[weights[cid] for cid in ordered_ids],
                action_counts=[client_action_counts.get(cid) for cid in ordered_ids],
                use_masked_head=self.use_masked_head,
            )

            self.global_model.load_state_dict(agg_state)

            # Update the federation's own trajectory.
            new_global_gradient = self.strategy.compute_pseudo_gradient(
                agg_state,
                global_state_before,
            )

            self._global_gradient = new_global_gradient
            self._previous_global_state = global_state_before

            if isinstance(self.strategy, GradientSurvivalStrategy):
                self.strategy.record_global_gradient(new_global_gradient)

            if r % eval_every == 0:
                total_norm = (
                    sum(
                        p.data.norm(2).item() ** 2
                        for p in self.global_model.q.parameters()
                    )
                    ** 0.5
                )

                logger.info(
                    "Round %d | global weight norm=%.6f | total_samples=%d",
                    r,
                    total_norm,
                    total_samples,
                )

                history["round"].append(r)
                history["client_samples"].append(total_samples)

                if self.evaluator:
                    metrics = self.evaluator.evaluate(self.global_model)

                    history["eval_reward"].append(metrics["mean_reward"])
                    history["eval_waiting_time"].append(
                        metrics["mean_waiting_time"]
                    )
                    history["eval_stopped"].append(metrics["mean_stopped"])

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

                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"global_round_{r:03d}.pth",
                )

                torch.save(self.global_model.q.state_dict(), ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

        return history