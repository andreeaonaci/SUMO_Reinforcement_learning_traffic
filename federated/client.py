"""CityClient — one federated client per city, one DQN agent per intersection.

Each city runs in multi-agent mode.  All intersections step the shared SUMO
simulation simultaneously, so the training loop collects experience from every
active intersection in lockstep.

Dead intersections (zero traffic during the probe phase) are excluded by
``build_multi_agent_city`` before this client is created, so every agent
here is guaranteed to receive a non-trivial reward signal.

Action selection respects each intersection's actual phase count: a 3-way
junction with 2 green phases only explores and evaluates actions {0, 1} even
though the shared DQN head outputs ``global_action_dim`` Q-values.  This
prevents the model from wasting capacity on impossible actions.

At the end of ``local_train`` the per-intersection weights are averaged locally
before being returned to the server (local FedAvg), which is then combined
with other cities' updates by the global FedAvg in the server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from agents.dqn import DQNAgent
from environments.federated_env import MultiAgentFederatedWrapper

logger = logging.getLogger(__name__)


class CityClient:
    """One federated participant per city.

    Args:
        name:                 City name (used in log messages).
        env:                  ``MultiAgentFederatedWrapper`` for this city.
        global_action_dim:    Output size of the shared DQN — must match
                              across all cities.  Passed explicitly so all
                              agents use the same architecture regardless of
                              how many green phases this city's intersections
                              happen to have.
        local_episodes:       Episodes to run per federated round.
        log_loss_every_steps: Print a mid-episode loss line every N steps.
                              0 = end-of-episode summaries only.
    """

    def __init__(
        self,
        name: str,
        env: MultiAgentFederatedWrapper,
        global_action_dim: int,
        local_episodes: int = 5,
        log_loss_every_steps: int = 50,
    ):
        self.name                 = name
        self.local_episodes       = local_episodes
        self.log_loss_every_steps = log_loss_every_steps
        self._env                 = env
        self._global_action_dim   = global_action_dim

        # One agent per active intersection, all using the global action dim
        self._agents: Dict[str, DQNAgent] = {
            ts_id: DQNAgent(
                obs_dim=env.observation_dim,
                action_dim=global_action_dim,
            )
            for ts_id in env.ts_ids
        }
        logger.info(
            "CityClient '%s': %d active intersections, obs_dim=%d, "
            "global_action_dim=%d, valid phases: %s",
            name,
            len(self._agents),
            env.observation_dim,
            global_action_dim,
            {ts: env.n_valid(ts) for ts in env.ts_ids},
        )

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------

    def local_train(self, global_state: Dict) -> Tuple[Dict, int]:
        """Sync with global model, run local episodes, return averaged weights.

        Args:
            global_state: ``state_dict`` from the global DQNAgent.

        Returns:
            state_dict:  Local FedAvg of all intersection agents' weights.
            total_steps: Total environment steps taken.
        """
        logger.info(
            "CityClient '%s' starting local training (%d episodes, "
            "%d intersections).",
            self.name, self.local_episodes, len(self._agents),
        )

        # Sync weights; preserve replay buffers and epsilon schedules
        for agent in self._agents.values():
            agent.load_state_dict(global_state)

        total_steps = 0

        for ep in range(1, self.local_episodes + 1):
            obs        = self._env.reset()    # {ts_id: np.ndarray}
            done       = False
            ep_steps   = 0
            ep_losses: List[float] = []

            while not done:
                # Each intersection acts within its own valid phase count,
                # not the full global action space
                actions = {
                    ts_id: agent.act(
                        obs[ts_id],
                        explore=True,
                        n_valid=self._env.n_valid(ts_id),
                    )
                    for ts_id, agent in self._agents.items()
                }

                next_obs, rewards, done, _ = self._env.step(actions)

                for ts_id, agent in self._agents.items():
                    agent.remember(
                        obs[ts_id],
                        actions[ts_id],
                        float(rewards.get(ts_id, 0.0)),
                        next_obs[ts_id],
                        done,
                    )
                    loss = agent.train_step()
                    if loss is not None:
                        ep_losses.append(loss)

                obs       = next_obs
                ep_steps += 1
                total_steps += 1

                # Mid-episode loss log
                if (
                    self.log_loss_every_steps > 0
                    and ep_losses
                    and ep_steps % self.log_loss_every_steps == 0
                ):
                    n = self.log_loss_every_steps * len(self._agents)
                    recent = ep_losses[-n:]
                    msg = (
                        f"  [{self.name}] ep={ep}/{self.local_episodes}"
                        f"  step={ep_steps}"
                        f"  loss(last {len(recent)})={np.mean(recent):.6f}"
                    )
                    logger.info(msg)
                    print(msg, flush=True)

            # End-of-episode summary
            if ep_losses:
                msg = (
                    f"[{self.name}] ep={ep}/{self.local_episodes}"
                    f"  steps={ep_steps}"
                    f"  intersections={len(self._agents)}"
                    f"  loss  mean={np.mean(ep_losses):.6f}"
                    f"  min={np.min(ep_losses):.6f}"
                    f"  max={np.max(ep_losses):.6f}"
                    f"  updates={len(ep_losses)}"
                )
            else:
                msg = (
                    f"[{self.name}] ep={ep}/{self.local_episodes}"
                    f"  steps={ep_steps}"
                    f"  loss=n/a (buffer not yet full)"
                )
            logger.info(msg)
            print(msg, flush=True)

        local_avg = self._local_fedavg()
        logger.info(
            "CityClient '%s' done: total_steps=%d, intersections=%d.",
            self.name, total_steps, len(self._agents),
        )
        return local_avg, total_steps

    def close(self) -> None:
        if hasattr(self, "_env"):
            try:
                self._env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _local_fedavg(self) -> Dict[str, torch.Tensor]:
        """Average state_dicts across all intersection agents."""
        weights = [agent.state_dict() for agent in self._agents.values()]
        return {
            key: torch.mean(torch.stack([w[key].float() for w in weights]), dim=0)
            for key in weights[0]
        }
