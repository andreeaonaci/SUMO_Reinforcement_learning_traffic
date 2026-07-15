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
run), and sends back the updated weights + sample count -- repeating
until told to stop.
"""
import logging
import multiprocessing as mp
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import torch

from federated.aggregation import fed_avg
from federated.comm_dropout import CommDropoutWrapper
from environments.federated_env import build_federated_env, ActionMaskPadder
from agents.dqn import DQNAgent

logger = logging.getLogger(__name__)


def seed_everything(seed: int | None = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


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
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
):
    """Runs inside its own process for the ENTIRE training run.

    Builds its env/agent once (so SUMO only starts once, and the replay
    buffer/epsilon schedule persist across rounds -- same warm-start
    behavior as the single-process ``FederatedClient``), then loops on
    ``in_queue`` for work until it receives a stop sentinel.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | %(levelname)s | [{name}] %(name)s | %(message)s",
    )
    worker_seed = None
    if cfg.get("seed") is not None:
        worker_seed = int(cfg["seed"]) + sum(ord(ch) for ch in name)
    seed_everything(worker_seed)
    try:
        env = build_federated_env(cfg)
        env = CommDropoutWrapper(ActionMaskPadder(env, action_dim), **comm_dropout_cfg)
        agent = DQNAgent(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            k_max=k_max,
            action_dim=action_dim,
            eps_decay=eps_decay,
        )

        while True:
            msg = in_queue.get()  # blocks until the main process sends something
            if msg is None or msg[0] == "stop":
                break

            _, global_state = msg
            agent.load_state_dict(global_state)
            try:
                state_dict, n_samples = agent.train(
                    env, episodes=local_episodes, log_loss_every_steps=log_loss_every_steps
                )
                out_queue.put(("ok", name, state_dict, n_samples))
            except Exception as e:
                logger.exception("Worker '%s' local training failed.", name)
                out_queue.put(("error", name, str(e), 0))
    finally:
        try:
            env.close()
        except Exception:
            pass


class ParallelFederatedServer:
    """Drop-in alternative to ``FederatedServer`` that trains every city
    concurrently instead of one-after-another. Same round/eval/checkpoint
    semantics; the only difference is HOW local_train gets called.

    Args:
        city_configs: list of (name, cfg) tuples -- the raw config dicts,
                      NOT pre-built envs (workers build their own envs
                      inside their own process; a SUMO env holding an open
                      subprocess/socket generally can't be handed across a
                      process boundary).
        own_dim, neighbor_dim, k_max, action_dim: shared network dims,
                      already resolved across all cities (see
                      ``federated_training.load_clients``).
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
        log_loss_every_steps: int = 50,
        eps_decay: float = 20000.0,
        evaluator: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        client_checkpoint_every: int = 1,
    ):
        self.global_model = global_model
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.client_checkpoint_every = client_checkpoint_every
        os.makedirs(os.path.join(checkpoint_dir, "clients"), exist_ok=True)

        ctx = mp.get_context("spawn")  # 'spawn' is safer than 'fork' across platforms
        # for processes that will themselves launch subprocesses (SUMO).
        self.names = [name for name, _ in city_configs]
        self.in_queues = {name: ctx.Queue() for name in self.names}
        self.out_queue = ctx.Queue()  # shared: workers tag their own name on results

        self.processes = []
        for name, cfg in city_configs:
            p = ctx.Process(
                target=_client_worker,
                args=(
                    name, cfg, own_dim, neighbor_dim, k_max, action_dim,
                    comm_dropout_cfg, local_episodes, log_loss_every_steps,
                    eps_decay,
                    self.in_queues[name], self.out_queue,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        logger.info("Started %d parallel city worker processes.", len(self.processes))

    def run(self, rounds: int, eval_every: int = 1) -> Dict[str, Any]:
        history: Dict[str, list] = {
            "round": [], "client_samples": [], "eval_reward": [],
            "eval_waiting_time": [], "eval_stopped": [], "eval_action_counts": [], "eval_q_gaps": [],
        }

        try:
            for r in range(1, rounds + 1):
                logger.info("=== Federated round %d / %d (parallel) ===", r, rounds)
                global_state = self.global_model.state_dict()

                # Dispatch to every city at once -- they now train concurrently.
                for name in self.names:
                    self.in_queues[name].put(("train", global_state))

                # Collect results as they arrive (order doesn't matter).
                updates = []
                total_samples = 0
                pending = set(self.names)
                while pending:
                    status, name, payload, n_samples = self.out_queue.get()
                    pending.discard(name)
                    if status == "error":
                        raise RuntimeError(f"Client '{name}' failed: {payload}")
                    state_dict = payload
                    updates.append((state_dict, n_samples))
                    total_samples += n_samples

                    if self.client_checkpoint_every and (r % self.client_checkpoint_every == 0):
                        ckpt_path = os.path.join(self.checkpoint_dir, "clients", f"{name}_round_{r:03d}.pth")
                        torch.save(state_dict, ckpt_path)
                        logger.info("Client '%s' local checkpoint saved: %s (samples=%d)", name, ckpt_path, n_samples)

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
                        history["eval_action_counts"].append(metrics.get("action_counts"))
                        history["eval_q_gaps"].append(metrics.get("q_gaps"))
                        logger.info(
                            "Round %d | reward=%.4f | waiting_time=%.2fs | stopped=%.1f",
                            r, metrics["mean_reward"], metrics["mean_waiting_time"], metrics["mean_stopped"],
                        )
                    else:
                        history["eval_reward"].append(None)
                        history["eval_waiting_time"].append(None)
                        history["eval_stopped"].append(None)
                        history["eval_action_counts"].append(None)
                        history["eval_q_gaps"].append(None)

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
