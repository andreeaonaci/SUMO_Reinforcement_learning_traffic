"""Parallel federated training: one persistent OS process per city.

Why processes, not threads
---------------------------
Each city's env is a SUMO simulation reached through TraCI/libsumo -- an
external subprocess talking over a socket (or an embedded C++ sim under
libsumo). Stepping it is what actually costs the wall-clock time in this
project, not the DQN math. Python threads wouldn't help even for the
DQN math (GIL), and can't help with SUMO stepping at all (a single
simulation is inherently sequential). What CAN run concurrently is
DIFFERENT cities' simulations, since they're fully independent until
FedAvg -- that's what this module parallelizes, with real OS processes.

Why persistent workers, not "spawn fresh each round"
------------------------------------------------------
`federated/client.py`'s whole design point is a warm start: build the env
and agent ONCE, keep the replay buffer and epsilon schedule alive across
every round. Spawning a fresh process (and fresh SUMO instance) every
round would throw that away and pay SUMO startup cost every round too.
Instead, each worker process is started ONCE at the beginning of training,
loops receiving the current global weights over a Queue, trains locally
(keeping its own persistent env + agent + replay buffer across the whole
run), and sends back the updated weights + sample count + mean loss --
repeating until told to stop.

Aggregation
-----------
Uses the same pluggable ``BaseAggregationStrategy`` as the sequential
``FederatedServer`` (see ``federated/aggregation_strategies.py``). No
aggregation-specific logic lives in this file beyond building a
``ClientRoundInfo`` per worker result and handing the batch to the
configured strategy -- identical pattern to the sequential server, just
fed by queue messages instead of direct method calls.
"""
import logging
import multiprocessing as mp
import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

import torch

from federated.aggregation import masked_head_weighted_average, weighted_average
from federated.aggregation_strategies import (
    BaseAggregationStrategy,
    ClientRoundInfo,
    GradientSurvivalStrategy,
    build_aggregation_strategy,
)
from federated.comm_dropout import CommDropoutWrapper
from environments.federated_env import build_federated_env, ActionMaskPadder
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def _client_worker(
    name: str,
    cfg: Dict[str, Any],
    own_dim: int,
    neighbor_dim: int,
    k_max: int,
    action_dim: int,
    comm_dropout_cfg: Dict[str, float],
    local_episodes: int,
    log_loss_every_steps: int,
    eps_decay: float,
    lr: float,
    lr_decay: float,
    min_lr: float,
    head_fix: bool,
    log_file: str,
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
):
    """Runs inside its own process for the ENTIRE training run.

    Builds its env/agent once (so SUMO only starts once, and the replay
    buffer/epsilon schedule persist across rounds -- same warm-start
    behavior as the single-process ``FederatedClient``), then loops on
    ``in_queue`` for work until it receives a stop sentinel.

    ``agent.train()`` returns ``(state_dict, n_samples, mean_loss)`` --
    all three are sent back so the server can build a ``ClientRoundInfo``
    for strategies that need a loss signal (ema_loss, ema_alignment, ...).
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(levelname)s | [{name}] %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
        force=True,
    )
    try:
        env = build_federated_env(cfg)
        env = CommDropoutWrapper(ActionMaskPadder(env, action_dim), **comm_dropout_cfg)
        agent = DQNAgent(
            own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
            action_dim=action_dim, eps_decay=eps_decay,
            lr=lr, lr_decay=lr_decay, min_lr=min_lr,
            head_fix=head_fix,
        )

        while True:
            msg = in_queue.get()  # blocks until the main process sends something
            if msg is None or msg[0] == "stop":
                break

            _, global_state = msg
            agent.load_state_dict(global_state)
            try:
                eps_start = agent.current_epsilon()
                state_dict, n_samples, mean_loss, action_counts = agent.train(
                    env, episodes=local_episodes, log_loss_every_steps=log_loss_every_steps
                )
                new_lr = agent.decay_lr()
                eps_end = agent.current_epsilon()
                logger.info(
                    "Worker '%s' round done: mean_loss=%s  lr=%.2e  eps_start=%.4f eps_end=%.4f  action_counts=%s",
                    name, f"{mean_loss:.6f}" if mean_loss is not None else "n/a",
                    new_lr, eps_start, eps_end, action_counts,
                )
                out_queue.put(("ok", name, state_dict, n_samples, mean_loss, action_counts, eps_start, eps_end))
            except Exception as e:
                logger.exception("Worker '%s' local training failed.", name)
                out_queue.put(("error", name, str(e), 0, None, None, None, None))
    finally:
        try:
            env.close()
        except Exception:
            pass


class ParallelFederatedServer:
    """Drop-in alternative to ``FederatedServer`` that trains every city
    concurrently instead of one-after-another. Same round/eval/checkpoint/
    aggregation semantics; the only difference is HOW local_train gets
    called (queue round-trip to a persistent worker process instead of a
    direct method call).

    Args:
        city_configs: list of (name, cfg) tuples -- the raw config dicts,
                      NOT pre-built envs (workers build their own envs
                      inside their own process; a SUMO env holding an open
                      subprocess/socket generally can't be handed across a
                      process boundary).
        own_dim, neighbor_dim, k_max, action_dim: shared network dims,
                      already resolved across all cities (see
                      ``federated_training.load_clients``).
        eps_decay:    Computed once by the caller from the training
                      schedule (see ``federated/utils.compute_eps_decay``)
                      and forwarded to every worker so all cities share the
                      same exploration schedule.
        aggregation_strategy / aggregation_config: same meaning as on
                      ``FederatedServer`` -- see
                      ``federated/aggregation_strategies.py``.
    """

    def __init__(
        self,
        global_model: DQNAgent,
        city_configs: List[Tuple[str, Dict[str, Any]]],
        own_dim: int,
        neighbor_dim: int,
        k_max: int,
        action_dim: int,
        comm_dropout_cfg: Dict[str, float],
        local_episodes: int,
        eps_decay: float,
        log_loss_every_steps: int = 50,
        evaluator: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        log_file: Optional[str] = None,
        client_checkpoint_every: int = 1,
        aggregation_strategy: str = "fedavg",
        aggregation_config: Optional[Dict[str, Any]] = None,
        default_lr: float = 3e-4,
        lr_decay: float = 1.0,
        min_lr: float = 1e-6,
        per_city_lr: Optional[Dict[str, float]] = None,
        head_fix: bool = True,
    ):
        self.global_model = global_model
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file or os.path.join(checkpoint_dir, "training.log")
        self.client_checkpoint_every = client_checkpoint_every
        os.makedirs(os.path.join(checkpoint_dir, "clients"), exist_ok=True)

        self.strategy: BaseAggregationStrategy = build_aggregation_strategy(
            aggregation_strategy, aggregation_config
        )
        logger.info(
            "[parallel] Aggregation strategy: %s  config=%s",
            type(self.strategy).__name__, aggregation_config or {},
        )

        # Per-client history for computing deltas each round (same fields
        # as the sequential FederatedServer).
        self._previous_client_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self._previous_loss: Dict[str, float] = {}
        self._previous_global_state: Optional[Dict[str, torch.Tensor]] = None
        self._global_gradient: Optional[Dict[str, torch.Tensor]] = None

        ctx = mp.get_context("spawn")  # 'spawn' is safer than 'fork' across platforms
        # for processes that will themselves launch subprocesses (SUMO).
        self.names = [name for name, _ in city_configs]
        self.in_queues = {name: ctx.Queue() for name in self.names}
        self.out_queue = ctx.Queue()  # shared: workers tag their own name on results

        per_city_lr = per_city_lr or {}
        self.processes = []
        for name, cfg in city_configs:
            city_lr = per_city_lr.get(name, default_lr)
            p = ctx.Process(
                target=_client_worker,
                args=(
                    name, cfg, own_dim, neighbor_dim, k_max, action_dim,
                    comm_dropout_cfg, local_episodes, log_loss_every_steps, eps_decay,
                    city_lr, lr_decay, min_lr,
                    head_fix,
                    self.log_file,
                    self.in_queues[name], self.out_queue,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            logger.info("[parallel] city='%s' lr=%.2e lr_decay=%.4f", name, city_lr, lr_decay)

        logger.info("Started %d parallel city worker processes.", len(self.processes))

    def _atomic_save_history(self, history: Dict[str, list]) -> None:
        history_path = os.path.join(self.checkpoint_dir, "federated_history.json")
        tmp_path = history_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_path, history_path)

    def run(self, rounds: int, eval_every: int = 1) -> Dict[str, Any]:
        history: Dict[str, list] = {
            "round": [], "client_samples": [], "eval_reward": [],
            "round_eps_start": [], "round_eps_end": [],
            "eval_reward_std": [], "eval_reward_episodes": [],
            "eval_waiting_time": [], "eval_waiting_time_std": [], "eval_waiting_time_episodes": [],
            "eval_stopped": [], "eval_stopped_std": [], "eval_stopped_episodes": [],
            "eval_action_counts": [], "eval_q_gaps": [],
        }

        try:
            for r in range(1, rounds + 1):
                logger.info("=== Federated round %d / %d (parallel) ===", r, rounds)
                global_state_before = self.global_model.state_dict()

                # Dispatch to every city at once -- they now train concurrently.
                for name in self.names:
                    self.in_queues[name].put(("train", global_state_before))

                # Collect results as they arrive (order doesn't matter).
                client_states: Dict[str, Dict[str, torch.Tensor]] = {}
                infos: List[ClientRoundInfo] = []
                total_samples = 0
                pending = set(self.names)

                client_action_counts: Dict[str, Optional[Dict[int, int]]] = {}
                eps_start_by_client: Dict[str, Optional[float]] = {}
                eps_end_by_client: Dict[str, Optional[float]] = {}

                while pending:
                    status, name, payload, n_samples, mean_loss, action_counts, eps_start, eps_end = self.out_queue.get()
                    pending.discard(name)
                    if status == "error":
                        raise RuntimeError(f"Client '{name}' failed: {payload}")

                    state_dict = payload
                    client_states[name] = state_dict
                    client_action_counts[name] = action_counts
                    eps_start_by_client[name] = eps_start
                    eps_end_by_client[name] = eps_end
                    total_samples += n_samples

                    client_gradient = self.strategy.compute_pseudo_gradient(
                        state_dict, global_state_before
                    )

                    infos.append(ClientRoundInfo(
                        client_id=name,
                        num_samples=n_samples,
                        client_state=state_dict,
                        previous_client_state=self._previous_client_state.get(name),
                        global_state=global_state_before,
                        previous_global_state=self._previous_global_state,
                        client_gradient=client_gradient,
                        previous_gradient=self.strategy.get_state(name).get("_last_client_gradient"),
                        global_gradient=self._global_gradient,
                        local_loss=mean_loss,
                        previous_loss=self._previous_loss.get(name),
                        round_num=r,
                    ))

                    self._previous_client_state[name] = state_dict
                    if mean_loss is not None:
                        self._previous_loss[name] = mean_loss
                    if client_gradient is not None:
                        self.strategy.get_state(name)["_last_client_gradient"] = client_gradient

                    if eps_start is not None and eps_end is not None:
                        logger.info(
                            "Round %d | client=%s epsilon start=%.4f end=%.4f",
                            r,
                            name,
                            eps_start,
                            eps_end,
                        )

                    if self.client_checkpoint_every and (r % self.client_checkpoint_every == 0):
                        ckpt_path = os.path.join(self.checkpoint_dir, "clients", f"{name}_round_{r:03d}.pth")
                        torch.save(state_dict, ckpt_path)
                        logger.info(
                            "Client '%s' local checkpoint saved: %s (samples=%d, mean_loss=%s)",
                            name, ckpt_path, n_samples,
                            f"{mean_loss:.6f}" if mean_loss is not None else "n/a",
                        )

                # ---- weight + aggregate ----------------------------------
                weights = self.strategy.compute_weights(infos)
                ordered_ids = [info.client_id for info in infos]
                agg_state = masked_head_weighted_average(
                    [client_states[cid] for cid in ordered_ids],
                    [weights[cid] for cid in ordered_ids],
                    [client_action_counts.get(cid) for cid in ordered_ids],
                )
                self.global_model.load_state_dict(agg_state)

                new_global_gradient = self.strategy.compute_pseudo_gradient(
                    agg_state, global_state_before
                )
                self._global_gradient = new_global_gradient
                self._previous_global_state = global_state_before
                if isinstance(self.strategy, GradientSurvivalStrategy):
                    self.strategy.record_global_gradient(new_global_gradient)

                if r % eval_every == 0:
                    history["round"].append(r)
                    history["client_samples"].append(total_samples)
                    history["round_eps_start"].append(eps_start_by_client)
                    history["round_eps_end"].append(eps_end_by_client)

                    if self.evaluator:
                        metrics = self.evaluator.evaluate(self.global_model)
                        history["eval_reward"].append(metrics["mean_reward"])
                        history["eval_reward_std"].append(metrics.get("std_reward"))
                        history["eval_reward_episodes"].append(metrics.get("per_episode_reward"))
                        history["eval_waiting_time"].append(metrics["mean_waiting_time"])
                        history["eval_waiting_time_std"].append(metrics.get("std_waiting_time"))
                        history["eval_waiting_time_episodes"].append(metrics.get("per_episode_waiting_time"))
                        history["eval_stopped"].append(metrics["mean_stopped"])
                        history["eval_stopped_std"].append(metrics.get("std_stopped"))
                        history["eval_stopped_episodes"].append(metrics.get("per_episode_stopped"))
                        history["eval_action_counts"].append(metrics.get("action_counts"))
                        history["eval_q_gaps"].append(metrics.get("q_gaps"))
                        logger.info(
                            "Round %d | reward mean=%.4f std=%.4f | waiting_time mean=%.2fs std=%.2f | stopped mean=%.1f std=%.1f",
                            r,
                            metrics["mean_reward"],
                            metrics.get("std_reward", 0.0),
                            metrics["mean_waiting_time"],
                            metrics.get("std_waiting_time", 0.0),
                            metrics["mean_stopped"],
                            metrics.get("std_stopped", 0.0),
                        )
                    else:
                        history["eval_reward"].append(None)
                        history["eval_reward_std"].append(None)
                        history["eval_reward_episodes"].append(None)
                        history["eval_waiting_time"].append(None)
                        history["eval_waiting_time_std"].append(None)
                        history["eval_waiting_time_episodes"].append(None)
                        history["eval_stopped"].append(None)
                        history["eval_stopped_std"].append(None)
                        history["eval_stopped_episodes"].append(None)
                        history["eval_action_counts"].append(None)
                        history["eval_q_gaps"].append(None)

                    self._atomic_save_history(history)
                    logger.info(
                        "Round %d | partial history saved to %s",
                        r,
                        os.path.join(self.checkpoint_dir, "federated_history.json"),
                    )

                    ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{r:03d}.pth")
                    torch.save(self.global_model.q.state_dict(), ckpt_path)
                    logger.info("Checkpoint saved: %s", ckpt_path)
        finally:
            self.close()

        return history

    def close(self):
        for name in self.names:
            try:
                self.in_queues[name].put(("stop", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
