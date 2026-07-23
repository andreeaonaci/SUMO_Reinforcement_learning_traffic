"""FederatedClient — one participant per city in the sequential FedAvg loop.

Design notes
------------
One DQNAgent per city (not per intersection).  The same network weights
act for every intersection each tick, and every intersection's transitions
land in the same replay buffer -- this is what trains a topology-agnostic
policy (see agents/dqn.py and agents/networks.py for details).

The env handed in here is already fully wrapped:
    raw SUMO env
        -> MultiAgentFederatedWrapper   (own + neighbor obs, action_mask)
        -> ActionMaskPadder             (pads to global action_dim)
        -> CommDropoutWrapper           (simulates unreliable comms)
All of that is assembled in main.py load_clients(); this class just
drives the training loop.

Relationship to parallel_server.py
------------------------------------
``_client_worker`` in parallel_server.py does the same job as
``local_train`` here, but runs in its own OS process. Both call
``agent.train(env, episodes=..., log_loss_every_steps=...)`` and return
the result to the server. The logic is intentionally kept identical so
sequential and parallel modes produce the same training behavior.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


class FederatedClient:
    """One city participant in the sequential FedAvg training loop.

    Args:
        name:                 City name (used in log messages).
        env_builder:          Zero-argument callable that returns the fully
                              wrapped city environment.  Called once at
                              construction time so the env (and SUMO) persists
                              across every federated round.
        agent_builder:        Zero-argument callable that returns a fresh
                              DQNAgent.  Called once at construction time.
                              The agent's replay buffer and epsilon schedule
                              also persist across rounds -- only the network
                              weights are synced at the start of each round.
        local_episodes:       Episodes to run per federated round.
        log_loss_every_steps: Passed through to ``DQNAgent.train()``.
                              Print a mid-episode loss line every N steps.
                              0 = end-of-episode summaries only.
    """

    def __init__(
        self,
        name: str,
        env_builder: Callable,
        agent_builder: Callable,
        local_episodes: int = 5,
        log_loss_every_steps: int = 50,
    ):
        self.name                 = name
        self.local_episodes       = local_episodes
        self.log_loss_every_steps = log_loss_every_steps

        # Build once; reuse across all federated rounds.
        self._env   = env_builder()
        self._agent: DQNAgent = agent_builder()

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------

    def local_train(self, global_state: Dict) -> Tuple[Dict, int]:
        """Sync network weights with the global model, then train locally.

        The replay buffer and epsilon schedule are NOT reset -- they
        accumulate warm experience across rounds, which is the main
        advantage of persistent clients over spawn-fresh-each-round.

        Args:
            global_state: ``state_dict`` from the global DQNAgent.

        Returns:
            state_dict:  Updated local weights after training.
            n_samples:   Environment steps taken (used for weighted FedAvg).
        """
        logger.info(
            "Client '%s' starting local training (%d episodes).",
            self.name, self.local_episodes,
        )

        # Sync weights only -- keep replay buffer + epsilon schedule alive.
        self._agent.load_state_dict(global_state)

        try:
            # DQNAgent.train() handles the full multi-agent loop:
            #   - same shared epsilon for all intersections each tick
            #   - one replay buffer entry per intersection per tick
            #   - one gradient step per tick (all intersections pooled)
            #   - mid-episode and end-of-episode loss logging
            state_dict, n_samples, mean_loss, action_counts = self._agent.train(
                self._env,
                episodes=self.local_episodes,
                log_loss_every_steps=self.log_loss_every_steps,
            )
        except Exception:
            logger.exception("Client '%s' local training failed.", self.name)
            try:
                self._env.close()
            except Exception:
                pass
            raise

        new_lr = self._agent.decay_lr()
        logger.info(
            "Client '%s' finished local training: n_samples=%d  mean_loss=%s  lr=%.2e  "
            "action_counts=%s.",
            self.name, n_samples,
            f"{mean_loss:.6f}" if mean_loss is not None else "n/a",
            new_lr, action_counts,
        )
        return state_dict, n_samples, mean_loss, action_counts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self, "_env"):
            try:
                self._env.close()
            except Exception:
                pass
