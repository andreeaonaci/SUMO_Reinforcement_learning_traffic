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
import random
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from federated.aggregation import (
    aggregate_round,
    evaluate_with_optional_ema,
    head_key_names,
    shape_server_update,
    update_eval_ema,
    weighted_average,
)
from federated.aggregation_strategies import (
    BaseAggregationStrategy,
    ClusteredFedAvgStrategy,
    ClientRoundInfo,
    GradientSurvivalStrategy,
    build_aggregation_strategy,
)
from federated.comm_dropout import CommDropoutWrapper
from environments.federated_env import build_federated_env, ActionMaskPadder
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.munchausen_dqn import MunchausenDQNAgent
from agents.recurrent_dqn import RecurrentDQNAgent
from agents.topology_conditioned_dqn import TopologyConditionedDQNAgent
from agents.qrdqn import QRDQNAgent
from federated.utils import set_seed

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
    neighbor_attention: bool,
    tau: float,
    target_update: int,
    mu: float,
    dueling: bool,
    n_step: int,
    q_entropy_weight: float,
    log_file: str,
    in_queue: "mp.Queue",
    out_queue: "mp.Queue",
    seed: Optional[int] = None,
    init_steps_done: int = 0,
    epsilon_reset_every: int = 0,
    algo: str = "dqn",
    d_model: int = 128,
    n_heads: int = 4,
    munchausen_temp: float = 0.03,
    munchausen_alpha: float = 0.9,
    use_batchnorm: bool = False,
    activation: str = "relu",
    encoder_depth: int = 2,
    n_attn_layers: int = 1,
    anchor_revert: bool = False,
    anchor_warmup_calls: int = 100,
    anchor_check_every: int = 50,
    anchor_qgap_growth_threshold: float = 3.0,
    anchor_pullback_beta: float = 0.5,
    cql_weight: float = 0.0,
    n_quantiles: int = 21,
):
    """Runs inside its own process for the ENTIRE training run.

    Builds its env/agent once (so SUMO only starts once, and the replay
    buffer/epsilon schedule persist across rounds -- same warm-start
    behavior as the single-process ``FederatedClient``), then loops on
    ``in_queue`` for work until it receives a stop sentinel.

    ``agent.train()`` returns ``(state_dict, n_samples, mean_loss)`` --
    all three are sent back so the server can build a ``ClientRoundInfo``
    for strategies that need a loss signal (ema_loss, ema_alignment, ...).

    Seeding note: ``mp.get_context("spawn")`` starts a brand-new
    interpreter per worker -- it does NOT inherit the main process's
    ``random``/``numpy``/``torch`` global RNG state, so the top-level
    ``set_seed(args.seed)`` call in ``experiments/federated_training.py``
    never reaches here. Without the explicit seeding below, every worker's
    epsilon-greedy exploration (``agents/dqn.py``'s ``random.random()``/
    ``random.choice()``), replay-buffer minibatch sampling
    (``random.sample()``), and comm-dropout pattern are seeded from OS
    entropy at process start -- different every run regardless of
    ``--seed``. This was the primary source of the run-to-run
    non-determinism documented in ``fidings/divergence_investigation.md``.
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
    # Each city already gets its own OS process for parallelism -- torch's
    # default intra-op thread pool (one per process, sized to all visible
    # cores) is redundant on top of that and actively harmful: N cities
    # each spawning e.g. 6 BLAS threads on a 12-core box means 6x more
    # threads than cores the moment more than ~2 workers train at once,
    # thrashing on context switches instead of doing useful work. Pin each
    # worker to a single thread so the OS scheduler just runs N processes
    # on N cores directly.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(seed)
    try:
        env = build_federated_env(cfg)
        # Captured BEFORE ActionMaskPadder widens this city's own env to the
        # shared global action_dim -- every client's live Q-head ends up the
        # same padded width by design (that's the whole point of one shared
        # architecture across topologies), so ClusteredFedAvgStrategy's
        # "cluster by action_dim" (federated/aggregation_strategies.py)
        # would silently see identical widths for every city and degenerate
        # to an arbitrary alphabetical tie-break if this weren't threaded
        # through separately. See fidings/divergence_investigation.md sec 65.
        native_action_dim = env.max_action_dim
        dropout_cfg = dict(comm_dropout_cfg)
        # Offset by a large prime rather than reusing `seed` verbatim: the
        # comm-dropout RNG (its own random.Random/np.random.RandomState
        # instance, see CommDropoutWrapper) and the global random/np.random
        # state seeded by set_seed() above are otherwise handed the exact
        # same integer, so they'd produce identical -- not just
        # independent-looking -- pseudorandom sequences.
        dropout_cfg.setdefault("seed", None if seed is None else seed + 1_000_003)
        env = CommDropoutWrapper(ActionMaskPadder(env, action_dim), **dropout_cfg)
        if algo == "ppo":
            agent = PPOAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                d_model=d_model, n_heads=n_heads,
            )
        elif algo == "munchausen":
            agent = MunchausenDQNAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                tau=tau, dueling=dueling, n_step=n_step,
                init_steps_done=init_steps_done,
                d_model=d_model, n_heads=n_heads,
                munchausen_temp=munchausen_temp, munchausen_alpha=munchausen_alpha,
            )
        elif algo == "recurrent":
            agent = RecurrentDQNAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, eps_decay=eps_decay,
                lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                tau=tau, target_update=target_update,
                mu=mu, dueling=dueling, n_step=n_step,
                init_steps_done=init_steps_done,
                q_entropy_weight=q_entropy_weight,
                d_model=d_model, n_heads=n_heads,
            )
        elif algo == "topo":
            agent = TopologyConditionedDQNAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, eps_decay=eps_decay,
                lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                tau=tau, target_update=target_update,
                mu=mu, dueling=dueling, n_step=n_step,
                init_steps_done=init_steps_done,
                q_entropy_weight=q_entropy_weight,
                d_model=d_model, n_heads=n_heads,
            )
        elif algo == "qrdqn":
            agent = QRDQNAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, eps_decay=eps_decay,
                lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                tau=tau, target_update=target_update,
                mu=mu, n_step=n_step,
                init_steps_done=init_steps_done,
                d_model=d_model, n_heads=n_heads,
                n_quantiles=n_quantiles,
            )
        else:
            agent = DQNAgent(
                own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
                action_dim=action_dim, eps_decay=eps_decay,
                lr=lr, lr_decay=lr_decay, min_lr=min_lr,
                head_fix=neighbor_attention,
                tau=tau, target_update=target_update,
                mu=mu, dueling=dueling, n_step=n_step,
                init_steps_done=init_steps_done,
                q_entropy_weight=q_entropy_weight,
                d_model=d_model, n_heads=n_heads,
                use_batchnorm=use_batchnorm, activation=activation,
                encoder_depth=encoder_depth, n_attn_layers=n_attn_layers,
                anchor_revert=anchor_revert, anchor_warmup_calls=anchor_warmup_calls,
                anchor_check_every=anchor_check_every,
                anchor_qgap_growth_threshold=anchor_qgap_growth_threshold,
                anchor_pullback_beta=anchor_pullback_beta,
                cql_weight=cql_weight,
            )

        while True:
            msg = in_queue.get()  # blocks until the main process sends something
            if msg is None or msg[0] == "stop":
                break

            _, global_state, round_num, clear_replay = msg
            agent.start_round(global_state)
            if clear_replay and algo != "ppo":
                # item 20 (fidings sec 78): server detected a confident
                # lock-in on last round's eval and asked every worker to
                # drop its replay buffer before training this round. PPO
                # has no persistent replay buffer (on-policy) -- nothing to
                # clear, and clear_replay() doesn't exist on PPOAgent.
                agent.clear_replay()
                logger.info("Worker '%s' round %d: replay buffer cleared (lock-in detected last round)",
                            name, round_num)
            # Periodic exploration reset (item 11(b), fidings/divergence_
            # investigation.md §40): monotonic epsilon decay means a client
            # that's locked onto a confidently-wrong repeating action by
            # round ~16-20 (epsilon already ~0.05, §34) almost never samples
            # its way back out on its own -- §39 showed a one-off exploration
            # reset can walk a locked checkpoint into a good policy, and §40
            # showed a single reset doesn't durably fix a severely-locked
            # one. This makes that reset periodic instead of a one-shot
            # post-hoc repair: every `epsilon_reset_every` rounds, restart
            # this client's epsilon schedule at eps_start (steps_done=0)
            # rather than letting it keep decaying from where it left off.
            # Reuses the run's own eps_decay, not a separately-tuned one --
            # each reset cycle decays over the same eps_decay steps_done
            # is measured in, which was already fast enough (§39/§40's short
            # recovery bursts used a similar-order eps_decay) to reach the
            # floor again well before the next reset at typical round
            # lengths.
            if algo != "ppo" and epsilon_reset_every > 0 and round_num % epsilon_reset_every == 0:
                agent.steps_done = 0
                logger.info("Worker '%s' round %d: periodic epsilon reset (steps_done -> 0)",
                            name, round_num)
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
                out_queue.put(("ok", name, state_dict, n_samples, mean_loss, action_counts, eps_start, eps_end, native_action_dim))
            except Exception as e:
                logger.exception("Worker '%s' local training failed.", name)
                out_queue.put(("error", name, str(e), 0, None, None, None, None, None))
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
        neighbor_attention: bool = True,
        no_federation: bool = False,
        fedavg_blend: float = 1.0,
        tau: float = 0.005,
        target_update: int = 200,
        seed: Optional[int] = None,
        mu: float = 0.0,
        dueling: bool = False,
        server_momentum: float = 0.0,
        n_step: int = 1,
        q_entropy_weight: float = 0.0,
        pseudo_grad_clip: float = 0.0,
        eval_ema_decay: float = 0.0,
        init_steps_done: int = 0,
        epsilon_reset_every: int = 0,
        algo: str = "dqn",
        d_model: int = 128,
        n_heads: int = 4,
        munchausen_temp: float = 0.03,
        munchausen_alpha: float = 0.9,
        use_batchnorm: bool = False,
        activation: str = "relu",
        encoder_depth: int = 2,
        n_attn_layers: int = 1,
        lockin_reset_std_threshold: float = 0.0,
        anchor_revert: bool = False,
        anchor_warmup_calls: int = 100,
        anchor_check_every: int = 50,
        anchor_qgap_growth_threshold: float = 3.0,
        anchor_pullback_beta: float = 0.5,
        cql_weight: float = 0.0,
        n_quantiles: int = 21,
    ):
        # item 20 (fidings sec 78): if >0, a round whose eval std_reward
        # falls below this threshold (the same std<50 screen already used
        # throughout fidings/divergence_investigation.md to flag confident
        # lock-in, sec 49/50) triggers a replay-buffer clear on every
        # worker before the NEXT round trains. 0.0 (default) is an exact
        # no-op -- self._pending_clear_replay never becomes True.
        self.lockin_reset_std_threshold = float(lockin_reset_std_threshold)
        self.algo = algo
        self.d_model = d_model
        self.n_heads = n_heads
        self.munchausen_temp = munchausen_temp
        self.munchausen_alpha = munchausen_alpha
        self.encoder_depth = encoder_depth
        self.n_attn_layers = n_attn_layers
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.anchor_revert = anchor_revert
        self.anchor_warmup_calls = anchor_warmup_calls
        self.anchor_check_every = anchor_check_every
        self.anchor_qgap_growth_threshold = anchor_qgap_growth_threshold
        self.anchor_pullback_beta = anchor_pullback_beta
        self.cql_weight = cql_weight
        self.n_quantiles = n_quantiles
        self.global_model = global_model
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file or os.path.join(checkpoint_dir, "training.log")
        self.client_checkpoint_every = client_checkpoint_every
        self.no_federation = bool(no_federation)
        # self.head_fix: aggregation-time only (masked-head weighted average
        # across heterogeneous action-space widths, see use_masked_head
        # below). self.neighbor_attention: network-forward-time only (each
        # worker's Q-network attends over neighbor obs vs. mean-pools them,
        # see NeighborAttentionQNetwork.forward). These used to be the same
        # flag (a naming collision, not a deliberate coupling) -- every past
        # "masked-head ablation" result was silently also an
        # attention-vs-pooling comparison. Now independent so each can be
        # tested in isolation.
        # Forced off under PPO regardless of the requested flag: masked-head
        # aggregation's key names (head_key_names(dueling) below) are DQN
        # Q-head-specific ("advantage_head.*"/"head.4.*") and don't exist in
        # PPOAgent's state dict (policy_head/ac_value_head) -- plain full-
        # state FedAvg is used instead, same as the sequential path's
        # matching guard in experiments/federated_training.py.
        #
        # ALSO forced off under --batchnorm (found 2026-09-05, after §74's
        # comparison had already run): NeighborAttentionQNetwork's
        # _mlp_block inserts a _FlattenBatchNorm1d module between every
        # Linear and its activation when use_batchnorm=True, which shifts
        # the plain (non-dueling) head's appended output Linear from index
        # 4 to index 6 ("head.4.weight" -> "head.6.weight"). head_key_names()
        # still hardcodes "head.4.*", so with batchnorm on, masked-head
        # aggregation was silently finding no matching key and falling back
        # to plain full-state FedAvg -- an uncontrolled confound in every
        # --batchnorm run so far (the baseline it was compared against DID
        # get real masked-head aggregation). See fidings sec 74's
        # correction note and sec 75 for the batchnorm-off depth
        # experiments run instead once this was caught.
        # Forced off under --algo qrdqn too: the distributional head's final
        # Linear is action_dim*n_quantiles wide, not action_dim -- masked-head
        # aggregation's per-action row indexing (head_key_names below) doesn't
        # apply to that shape at all. Plain full-state FedAvg is used instead,
        # same fallback as ppo/batchnorm above.
        self.head_fix = bool(head_fix) and algo not in ("ppo", "qrdqn") and not use_batchnorm
        self.neighbor_attention = bool(neighbor_attention)
        self.fedavg_blend = float(max(0.0, min(1.0, fedavg_blend)))
        self._head_weight_key, self._head_bias_key = head_key_names(
            dueling and algo not in ("ppo", "qrdqn") and not use_batchnorm
        )
        self.server_momentum = float(server_momentum)
        self._momentum_buffer: Optional[Dict[str, torch.Tensor]] = None
        self.pseudo_grad_clip = float(pseudo_grad_clip)
        # Eval-only EMA snapshot: smooths what gets evaluated/reported each
        # round without touching what's actually broadcast to clients next
        # round (that always stays the raw aggregated state -- see the
        # eval-time swap in run() below). 0 = disabled, exact no-op.
        self.eval_ema_decay = float(eval_ema_decay)
        self._eval_ema_state: Optional[Dict[str, torch.Tensor]] = None
        self.epsilon_reset_every = int(epsilon_reset_every)
        os.makedirs(os.path.join(checkpoint_dir, "clients"), exist_ok=True)
        self.strategy = build_aggregation_strategy(
            aggregation_strategy, aggregation_config
        )
        
        logger.info(
            "[parallel] Aggregation strategy: %s  config=%s",
            type(self.strategy).__name__, aggregation_config or {},
        )
        logger.info("[parallel] No federation mode: %s", self.no_federation)
        logger.info("[parallel] FedAvg blend: %.3f  tau: %.4f  target_update: %d",
                    self.fedavg_blend, tau, target_update)

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
        for idx, (name, cfg) in enumerate(city_configs):
            city_lr = per_city_lr.get(name, default_lr)
            # Distinct-but-deterministic per-city seed: same --seed always
            # reproduces the same run, but cities don't all explore/sample
            # identically (see _client_worker's docstring for why this is
            # needed at all under spawn-based multiprocessing).
            city_seed = None if seed is None else seed + idx
            p = ctx.Process(
                target=_client_worker,
                args=(
                    name, cfg, own_dim, neighbor_dim, k_max, action_dim,
                    comm_dropout_cfg, local_episodes, log_loss_every_steps, eps_decay,
                    city_lr, lr_decay, min_lr,
                    self.neighbor_attention,
                    tau, target_update,
                    mu, dueling, n_step, q_entropy_weight,
                    self.log_file,
                    self.in_queues[name], self.out_queue,
                    city_seed, init_steps_done,
                    self.epsilon_reset_every,
                    self.algo,
                    self.d_model, self.n_heads,
                    self.munchausen_temp, self.munchausen_alpha,
                    self.use_batchnorm, self.activation,
                    self.encoder_depth, self.n_attn_layers,
                    self.anchor_revert, self.anchor_warmup_calls,
                    self.anchor_check_every, self.anchor_qgap_growth_threshold,
                    self.anchor_pullback_beta, self.cql_weight, self.n_quantiles,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            logger.info("[parallel] city='%s' lr=%.2e lr_decay=%.4f seed=%s", name, city_lr, lr_decay, city_seed)

        logger.info("Started %d parallel city worker processes.", len(self.processes))

    @staticmethod
    def _clone_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in state.items()}

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        t = torch.tensor(values, dtype=torch.float32)
        return float(torch.std(t, unbiased=True).item())

    def _evaluate_multiple_models(self, named_states: Dict[str, Dict[str, torch.Tensor]]) -> tuple[dict, dict]:
        per_model = {}
        rewards = []
        waits = []
        stops = []
        arrived = []

        for name, state in named_states.items():
            self.global_model.load_state_dict(state)
            m = self.evaluator.evaluate(self.global_model)
            per_model[name] = m
            rewards.append(float(m.get("mean_reward", 0.0)))
            waits.append(float(m.get("mean_waiting_time", 0.0)))
            stops.append(float(m.get("mean_stopped", 0.0)))
            arrived.append(float(m.get("mean_arrived", 0.0)))

        aggregate = {
            "mean_reward": self._mean(rewards),
            "std_reward": self._std(rewards),
            "per_episode_reward": None,
            "mean_waiting_time": self._mean(waits),
            "std_waiting_time": self._std(waits),
            "per_episode_waiting_time": None,
            "mean_stopped": self._mean(stops),
            "std_stopped": self._std(stops),
            "per_episode_stopped": None,
            "mean_arrived": self._mean(arrived),
            "action_counts": None,
            "q_gaps": None,
            "eval_city_name": self.evaluator.eval_city_name,
            "is_true_holdout": self.evaluator.is_true_holdout,
        }
        aggregate["eval_per_model"] = per_model
        return aggregate, per_model

    def _atomic_save_history(self, history: Dict[str, list]) -> None:
        history_path = os.path.join(self.checkpoint_dir, "federated_history.json")
        tmp_path = history_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_path, history_path)

    def run(
        self,
        rounds: int,
        eval_every: int = 1,
        start_round: int = 1,
        initial_history: Optional[Dict[str, list]] = None,
    ) -> Dict[str, Any]:
        # On a fresh run start_round=1 and initial_history=None -- identical
        # to the old always-empty-dict behavior. On --resume, the caller
        # passes the previous run's federated_history.json (already loaded)
        # so newly-completed rounds get appended to the same lists instead
        # of the history starting over from round 1.
        history: Dict[str, list] = initial_history if initial_history is not None else {
            "round": [], "client_samples": [], "eval_reward": [],
            "round_eps_start": [], "round_eps_end": [],
            "eval_reward_std": [], "eval_reward_episodes": [],
            "eval_waiting_time": [], "eval_waiting_time_std": [], "eval_waiting_time_episodes": [],
            "eval_stopped": [], "eval_stopped_std": [], "eval_stopped_episodes": [],
            "eval_arrived": [],
            "eval_action_counts": [], "eval_q_gaps": [],
            "eval_mode": [], "cluster_assignments": [],
            "eval_city_name": [], "is_true_holdout": [],
        }

        # self.global_model already carries the resumed checkpoint's weights
        # when start_round > 1 (loaded by the caller before constructing this
        # server), so cloning it here into every client's starting state is
        # correct in both the fresh-run and resumed-run case.
        base_global_state = self._clone_state_dict(self.global_model.state_dict())
        per_client_state: Dict[str, Dict[str, torch.Tensor]] = {
            name: self._clone_state_dict(base_global_state) for name in self.names
        }
        # Set from the PREVIOUS round's eval (see the bottom of this loop) --
        # False for start_round since there's no prior eval yet this run.
        pending_clear_replay = False

        try:
            for r in range(start_round, rounds + 1):
                logger.info("=== Federated round %d / %d (parallel) ===", r, rounds)
                global_state_before = self.global_model.state_dict()

                # Dispatch to every city at once -- they now train concurrently.
                for name in self.names:
                    if self.no_federation or isinstance(self.strategy, ClusteredFedAvgStrategy):
                        send_state = per_client_state[name]
                    else:
                        send_state = global_state_before
                    self.in_queues[name].put(("train", send_state, r, pending_clear_replay))
                pending_clear_replay = False  # consumed; next round's value set after this round's eval below

                # Collect results as they arrive (order doesn't matter).
                client_states: Dict[str, Dict[str, torch.Tensor]] = {}
                infos: List[ClientRoundInfo] = []
                total_samples = 0
                pending = set(self.names)

                client_action_counts: Dict[str, Optional[Dict[int, int]]] = {}
                eps_start_by_client: Dict[str, Optional[float]] = {}
                eps_end_by_client: Dict[str, Optional[float]] = {}

                while pending:
                    status, name, payload, n_samples, mean_loss, action_counts, eps_start, eps_end, native_action_dim = self.out_queue.get()
                    pending.discard(name)
                    if status == "error":
                        raise RuntimeError(f"Client '{name}' failed: {payload}")

                    state_dict = payload
                    client_states[name] = state_dict
                    per_client_state[name] = self._clone_state_dict(state_dict)
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
                        metadata={"action_dim": native_action_dim},
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
                cluster_assignments = None
                eval_named_states: Dict[str, Dict[str, torch.Tensor]] = {}
                if self.no_federation:
                    logger.info("[parallel] No-federation mode: skipping aggregation for round %d.", r)
                    first_cid = self.names[0]
                    self.global_model.load_state_dict(per_client_state[first_cid])
                    agg_state = self.global_model.state_dict()
                    eval_named_states = {cid: sd for cid, sd in per_client_state.items()}
                elif isinstance(self.strategy, ClusteredFedAvgStrategy):
                    cluster_assignments = self.strategy.assign_clusters(infos)
                    logger.info("[parallel] Round %d | clustered_fedavg assignments: %s", r, cluster_assignments)
                    cluster_models = self.strategy.aggregate_by_cluster(infos, cluster_assignments)

                    for cid in self.names:
                        cluster_id = cluster_assignments.get(cid, 0)
                        if cluster_id in cluster_models:
                            per_client_state[cid] = self._clone_state_dict(cluster_models[cluster_id])

                    if cluster_models:
                        cluster_states = list(cluster_models.values())
                        cluster_weights = []
                        for cluster_id, _state in cluster_models.items():
                            cluster_weights.append(
                                float(
                                    sum(
                                        info.num_samples
                                        for info in infos
                                        if cluster_assignments.get(info.client_id, 0) == cluster_id
                                    )
                                )
                            )
                        agg_state = weighted_average(cluster_states, cluster_weights)
                        self.global_model.load_state_dict(agg_state)
                    else:
                        agg_state = global_state_before

                    eval_named_states = {
                        f"cluster_{cluster_id}": state
                        for cluster_id, state in cluster_models.items()
                    }
                else:
                    weights = self.strategy.compute_weights(infos)
                    ordered_ids = [info.client_id for info in infos]
                    # `use_masked_head=self.head_fix` is what makes
                    # `--disable_head_fix` actually disable masked-head
                    # aggregation -- this branch used to call
                    # masked_head_weighted_average unconditionally, so the
                    # flag never worked under `--parallel` (it only changed
                    # the local network's neighbor-processing architecture,
                    # via what was then a shared `head_fix` value also
                    # threaded into _client_worker). `self.head_fix` is now
                    # aggregation-only and `self.neighbor_attention` (see
                    # __init__) is the independent flag threaded into
                    # _client_worker for the network-level choice -- the two
                    # were accidentally the same variable until 2026-08-15,
                    # confounding every past masked-head ablation with an
                    # attention-vs-pooling comparison. Reusing
                    # aggregate_round() here instead of reimplementing the
                    # dispatch also keeps this in sync with the sequential
                    # path (federated/server.py), which already calls it the
                    # same way via `use_masked_head=self.use_masked_head`.
                    agg_state = aggregate_round(
                        state_dicts=[client_states[cid] for cid in ordered_ids],
                        base_weights=[weights[cid] for cid in ordered_ids],
                        action_counts=[client_action_counts.get(cid) for cid in ordered_ids],
                        use_masked_head=self.head_fix,
                        head_weight_key=self._head_weight_key,
                        head_bias_key=self._head_bias_key,
                        previous_global_state=global_state_before,
                    )
                    if self.fedavg_blend < 1.0:
                        prev_state = self.global_model.state_dict()
                        b = self.fedavg_blend
                        agg_state = {
                            k: b * agg_state[k].float() + (1.0 - b) * prev_state[k].float()
                            for k in agg_state
                        }
                    agg_state, self._momentum_buffer = shape_server_update(
                        agg_state, global_state_before,
                        self.pseudo_grad_clip, self.server_momentum, self._momentum_buffer,
                    )
                    self.global_model.load_state_dict(agg_state)
                    if self.eval_ema_decay > 0.0:
                        self._eval_ema_state = update_eval_ema(
                            self._eval_ema_state, agg_state, self.eval_ema_decay
                        )

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
                    history["cluster_assignments"].append(cluster_assignments)
                    history["eval_mode"].append(
                        "no_federation" if self.no_federation else (
                            "clustered_fedavg" if isinstance(self.strategy, ClusteredFedAvgStrategy) else "federated"
                        )
                    )

                    if self.evaluator:
                        if self.no_federation or isinstance(self.strategy, ClusteredFedAvgStrategy):
                            if not eval_named_states:
                                eval_named_states = {"model": self.global_model.state_dict()}
                            metrics, per_model = self._evaluate_multiple_models(eval_named_states)
                            history.setdefault("eval_per_model", []).append(per_model)
                        else:
                            metrics = evaluate_with_optional_ema(
                                self.evaluator, self.global_model,
                                self.eval_ema_decay, self._eval_ema_state,
                            )
                        history["eval_reward"].append(metrics["mean_reward"])
                        history["eval_reward_std"].append(metrics.get("std_reward"))
                        history["eval_reward_episodes"].append(metrics.get("per_episode_reward"))
                        history["eval_waiting_time"].append(metrics["mean_waiting_time"])
                        history["eval_waiting_time_std"].append(metrics.get("std_waiting_time"))
                        history["eval_waiting_time_episodes"].append(metrics.get("per_episode_waiting_time"))
                        history["eval_stopped"].append(metrics["mean_stopped"])
                        history["eval_stopped_std"].append(metrics.get("std_stopped"))
                        history["eval_stopped_episodes"].append(metrics.get("per_episode_stopped"))
                        history["eval_arrived"].append(metrics.get("mean_arrived"))
                        history["eval_action_counts"].append(metrics.get("action_counts"))
                        history["eval_q_gaps"].append(metrics.get("q_gaps"))
                        history.setdefault("eval_city_name", []).append(metrics.get("eval_city_name"))
                        history.setdefault("is_true_holdout", []).append(metrics.get("is_true_holdout"))
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
                        # item 20 (fidings sec 78): flag a replay-buffer
                        # clear for every worker's NEXT round if this
                        # round's eval looks confidently locked. Uses the
                        # cheap 5-episode std screen already established
                        # throughout fidings/divergence_investigation.md
                        # (sec 49/50) -- a real but imperfect proxy (sec 56
                        # found false negatives near the threshold), fine
                        # for triggering an inexpensive corrective action
                        # rather than a load-bearing measurement.
                        std_reward = metrics.get("std_reward")
                        if self.lockin_reset_std_threshold > 0.0 and std_reward is not None \
                                and std_reward < self.lockin_reset_std_threshold:
                            pending_clear_replay = True
                            logger.info(
                                "Round %d | eval std=%.4f < threshold=%.1f -- clearing every "
                                "worker's replay buffer before round %d",
                                r, std_reward, self.lockin_reset_std_threshold, r + 1,
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
                        history["eval_arrived"].append(None)
                        history["eval_action_counts"].append(None)
                        history["eval_q_gaps"].append(None)
                        history.setdefault("eval_city_name", []).append(None)
                        history.setdefault("is_true_holdout", []).append(None)

                    self._atomic_save_history(history)
                    logger.info(
                        "Round %d | partial history saved to %s",
                        r,
                        os.path.join(self.checkpoint_dir, "federated_history.json"),
                    )

                    ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{r:03d}.pth")
                    # state_dict(), not .q.state_dict() -- agent-agnostic
                    # (see federated/server.py's matching fix); identical
                    # behavior for DQNAgent today, and needed if/when a
                    # non-DQN agent (agents/ppo.py) is wired into this
                    # (parallel) path.
                    torch.save(self.global_model.state_dict(), ckpt_path)
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
