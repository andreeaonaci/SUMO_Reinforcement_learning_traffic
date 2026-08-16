"""Run federated training across multiple city clients.

Each city runs ONE shared DQNAgent (the "foundation model") against ALL of
its intersections simultaneously -- the same weights act for every
intersection, and every intersection's transitions land in the same replay
buffer. This is what lets a single architecture generalize across 3-way,
4-way, 5-way, etc. intersections: topology differences are expressed
through `action_mask` (see agents/networks.py), never through separate
per-topology code or models.
"""
import argparse
from datetime import datetime
import glob
import json
import logging
import os
import pprint
import re
import sys
import yaml

try:
    from pyfiglet import Figlet
except ImportError:  # pragma: no cover - optional dependency fallback
    class Figlet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def renderText(self, text: str) -> str:
            return text

from federated.server import FederatedServer
from federated.parallel_server import ParallelFederatedServer
from federated.client import FederatedClient
from federated.evaluator import HoldoutEvaluator
from federated.comm_dropout import CommDropoutWrapper
from environments.federated_env import build_federated_env, ActionMaskPadder
from agents.dqn import DQNAgent
from federated.utils import compute_eps_decay, set_seed

run_dir = None
logger = logging.getLogger(__name__)

DEFAULT_COMM_DROPOUT = dict(
    p_link=0.10,
    p_isolate=0.05,
    p_hop_cutoff=0.10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def steps_per_episode_from_cfg(cfg: dict) -> int:
    """Ticks per episode = num_seconds / delta_time."""
    return int(cfg.get("num_seconds", 3600) // cfg.get("delta_time", 5))


def resolve_resume(resume_arg: str) -> tuple:
    """Resolve --resume into (run_dir, checkpoint_path, completed_round).

    Accepts either a run directory (picks its latest global_round_*.pth) or
    a direct path to one checkpoint file. Only the global model weights and
    the round number are recoverable from a checkpoint -- each worker's
    replay buffer, optimizer momentum, and epsilon step counter live only
    in that worker's now-gone process memory and are NOT restored; the
    resumed run's workers rebuild those from scratch (epsilon's step
    counter is approximated instead of reset, see init_steps_done below).
    """
    if os.path.isdir(resume_arg):
        run_dir = resume_arg
        ckpts = sorted(glob.glob(os.path.join(run_dir, "global_round_*.pth")))
        if not ckpts:
            raise ValueError(f"--resume: no global_round_*.pth checkpoints found in {resume_arg}")
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = resume_arg
        run_dir = os.path.dirname(ckpt_path) or "."

    m = re.search(r"global_round_(\d+)\.pth$", os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"--resume: could not parse a round number out of checkpoint path {ckpt_path}")
    completed_round = int(m.group(1))
    return run_dir, ckpt_path, completed_round


def _make_agent(own_dim, neighbor_dim, k_max, action_dim, eps_decay, head_fix: bool = True,
                tau: float = 0.005, target_update: int = 200, mu: float = 0.0,
                dueling: bool = False, n_step: int = 1):
    """Single place that constructs a DQNAgent with the computed eps_decay."""
    return DQNAgent(
        own_dim=own_dim,
        neighbor_dim=neighbor_dim,
        k_max=k_max,
        action_dim=action_dim,
        eps_decay=eps_decay,
        head_fix=head_fix,
        tau=tau,
        target_update=target_update,
        mu=mu,
        dueling=dueling,
        n_step=n_step,
    )


# ---------------------------------------------------------------------------
# Client loading  (sequential path)
# ---------------------------------------------------------------------------

def load_clients(
    base_dir: str,
    rounds: int,
    local_episodes: int,
    explore_fraction: float,
    log_loss_every_steps: int = 50,
    comm_dropout_cfg: dict | None = None,
    head_fix: bool = True,
    tau: float = 0.005,
    target_update: int = 200,
    mu: float = 0.0,
    dueling: bool = False,
    n_step: int = 1,
) -> tuple:
    """Build one FederatedClient per city directory.

    Returns:
        clients     list of FederatedClient
        obs_dims    (own_dim, neighbor_dim, k_max)
        action_dim  global Q-output size
        eps_decay   computed from training schedule (logged for reference)
    """
    comm_dropout_cfg = comm_dropout_cfg or DEFAULT_COMM_DROPOUT

    # ── Pass 1: build envs, collect dims ────────────────────────────────
    city_envs = []   # (name, env, cfg)
    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        env = build_federated_env(cfg)
        city_envs.append((name, env, cfg))

    if not city_envs:
        raise RuntimeError(f"No city configs found under '{base_dir}'.")

    own_dims      = {env.own_dim      for _, env, _ in city_envs}
    neighbor_dims = {env.neighbor_dim for _, env, _ in city_envs}
    k_maxs        = {env.k_max        for _, env, _ in city_envs}

    if len(own_dims) != 1:
        raise RuntimeError(f"Cities produce different own_dim values: {own_dims}.")
    if len(neighbor_dims) != 1:
        raise RuntimeError(f"Cities produce different neighbor_dim values: {neighbor_dims}.")
    if len(k_maxs) != 1:
        raise RuntimeError(f"Cities use different k_max values: {k_maxs}.")

    own_dim, neighbor_dim, k_max = own_dims.pop(), neighbor_dims.pop(), k_maxs.pop()
    action_dim = max(env.max_action_dim for _, env, _ in city_envs)

    # ── Compute eps_decay from actual training schedule ──────────────────
    # Use the first city's config to derive steps_per_episode.  All cities
    # must share num_seconds/delta_time (enforced implicitly by the dim
    # checks above; a city with a different episode length would produce a
    # different own_dim via the encoder).
    first_cfg       = city_envs[0][2]
    steps_per_ep    = steps_per_episode_from_cfg(first_cfg)
    eps_decay       = compute_eps_decay(
        rounds=rounds,
        local_episodes=local_episodes,
        steps_per_episode=steps_per_ep,
        explore_fraction=explore_fraction,
    )
    logger.info(
        "Training schedule: rounds=%d  local_episodes=%d  "
        "steps_per_episode=%d  explore_fraction=%.2f  → eps_decay=%.1f",
        rounds, local_episodes, steps_per_ep, explore_fraction, eps_decay,
    )

    # ── Pass 2: wrap envs, build clients ────────────────────────────────
    clients = []
    for name, env, _ in city_envs:
        wrapped = CommDropoutWrapper(ActionMaskPadder(env, action_dim), **comm_dropout_cfg)

        # Default-arg capture avoids late-binding closure bug.
        def make_agent(
            _own=own_dim, _nbr=neighbor_dim, _k=k_max,
            _act=action_dim, _eps=eps_decay,
            _head_fix=head_fix,
            _tau=tau, _tu=target_update, _mu=mu, _dueling=dueling, _n_step=n_step,
        ):
            return _make_agent(_own, _nbr, _k, _act, _eps, head_fix=_head_fix,
                               tau=_tau, target_update=_tu, mu=_mu, dueling=_dueling,
                               n_step=_n_step)

        clients.append(
            FederatedClient(
                name=name,
                env_builder=lambda e=wrapped: e,
                agent_builder=make_agent,
                local_episodes=local_episodes,
                log_loss_every_steps=log_loss_every_steps,
            )
        )

    return clients, (own_dim, neighbor_dim, k_max), action_dim, eps_decay


# ---------------------------------------------------------------------------
# Dim resolution  (parallel path — workers build their own envs)
# ---------------------------------------------------------------------------

def resolve_city_configs_and_dims(base_dir: str) -> tuple:
    """Read dims and raw configs for the parallel path.

    Workers receive the raw cfg dict and build their own SUMO env inside
    their own process — a live SUMO env can't cross a process boundary.

    Returns:
        city_configs      list of (name, cfg)
        obs_dims          (own_dim, neighbor_dim, k_max)
        action_dim        global Q-output size
        steps_per_episode ticks per episode (from first city's config)
    """
    city_configs = []
    dims         = []
    first_cfg    = None

    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue
        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if first_cfg is None:
            first_cfg = cfg
        env = build_federated_env(cfg)
        dims.append((env.own_dim, env.neighbor_dim, env.k_max, env.max_action_dim))
        env.close()
        city_configs.append((name, cfg))

    if not city_configs:
        raise RuntimeError(f"No city configs found under '{base_dir}'.")

    own_dims      = {d[0] for d in dims}
    neighbor_dims = {d[1] for d in dims}
    k_maxs        = {d[2] for d in dims}
    if len(own_dims) != 1 or len(neighbor_dims) != 1 or len(k_maxs) != 1:
        raise RuntimeError(
            f"Mismatched dims across cities: own_dims={own_dims} "
            f"neighbor_dims={neighbor_dims} k_maxs={k_maxs}"
        )

    own_dim, neighbor_dim, k_max = own_dims.pop(), neighbor_dims.pop(), k_maxs.pop()
    action_dim       = max(d[3] for d in dims)
    steps_per_episode = steps_per_episode_from_cfg(first_cfg)

    return city_configs, (own_dim, neighbor_dim, k_max), action_dim, steps_per_episode


# ---------------------------------------------------------------------------
# Holdout evaluator
# ---------------------------------------------------------------------------

def make_holdout_evaluator(
    base_dir: str,
    obs_dims: tuple,
    action_dim: int,
    episodes: int = 5,
    eval_comm_dropout_cfg: dict | None = None,
    holdout_base_dir: str | None = None,
    eval_sumo_seed: int = 12345,
) -> "HoldoutEvaluator | None":
    candidate_base_dirs = [base_dir]
    if holdout_base_dir and holdout_base_dir not in candidate_base_dirs:
        candidate_base_dirs.append(holdout_base_dir)
    if "environments" not in candidate_base_dirs:
        candidate_base_dirs.append("environments")

    own_dim, neighbor_dim, k_max = obs_dims
    dropout_cfg = eval_comm_dropout_cfg if eval_comm_dropout_cfg is not None else DEFAULT_COMM_DROPOUT

    preferred_cfg = None
    preferred_cfg_path = None
    fallback_candidates = []

    for candidate in candidate_base_dirs:
        preferred_path = os.path.join(candidate, "city_5_holdout", "config.yaml")
        if os.path.exists(preferred_path) and preferred_cfg is None:
            with open(preferred_path) as f:
                preferred_cfg = yaml.safe_load(f)
            preferred_cfg_path = preferred_path

        if not os.path.isdir(candidate):
            continue
        for city_name in sorted(os.listdir(candidate)):
            cfg_path = os.path.join(candidate, city_name, "config.yaml")
            if os.path.exists(cfg_path):
                fallback_candidates.append((city_name, cfg_path))

    if preferred_cfg is None and not fallback_candidates:
        logger.warning(
            "No evaluation city found. Searched base dirs: %s",
            candidate_base_dirs,
        )
        return None

    selected_cfg = None
    selected_name = None

    # First preference: true holdout city if compatible.
    if preferred_cfg is not None:
        env = build_federated_env(preferred_cfg)
        try:
            dims_match = (
                env.own_dim == own_dim
                and env.neighbor_dim == neighbor_dim
                and env.k_max == k_max
            )
            action_ok = env.max_action_dim <= action_dim
            if dims_match and action_ok:
                selected_cfg = preferred_cfg
                selected_name = "city_5_holdout"
            else:
                logger.warning(
                    "Holdout evaluator city is incompatible (dims_match=%s, holdout_action_dim=%d, global_action_dim=%d). "
                    "Falling back to a compatible city from base dirs.",
                    dims_match,
                    env.max_action_dim,
                    action_dim,
                )
        finally:
            env.close()

    # Fallback: any compatible city config (useful for subset smoke tests).
    if selected_cfg is None:
        seen = set()
        for city_name, cfg_path in fallback_candidates:
            if cfg_path in seen:
                continue
            seen.add(cfg_path)
            with open(cfg_path) as f:
                candidate_cfg = yaml.safe_load(f)
            env = build_federated_env(candidate_cfg)
            try:
                dims_match = (
                    env.own_dim == own_dim
                    and env.neighbor_dim == neighbor_dim
                    and env.k_max == k_max
                )
                action_ok = env.max_action_dim <= action_dim
                if dims_match and action_ok:
                    selected_cfg = candidate_cfg
                    selected_name = city_name
                    break
            finally:
                env.close()

    if selected_cfg is None:
        logger.warning(
            "No compatible evaluation city found for obs_dims=%s and action_dim=%d.",
            obs_dims,
            action_dim,
        )
        return None

    if selected_name != "city_5_holdout":
        logger.info(
            "Using '%s' as evaluation city for this run (compatibility fallback).",
            selected_name,
        )
    elif preferred_cfg_path and not preferred_cfg_path.startswith(base_dir):
        logger.info(
            "Holdout city not found under base_dir='%s'; using '%s' for evaluation.",
            base_dir,
            os.path.dirname(os.path.dirname(preferred_cfg_path)),
        )

    def build_holdout_env():
        eval_cfg = dict(selected_cfg)
        eval_cfg["sumo_seed"] = int(eval_sumo_seed)
        env = build_federated_env(eval_cfg)
        env = ActionMaskPadder(env, action_dim)
        if dropout_cfg:
            env = CommDropoutWrapper(env, seed=int(eval_sumo_seed), **dropout_cfg)
        return env

    return HoldoutEvaluator(
        env_builder=build_holdout_env,
        episodes=episodes,
        eval_seed_base=int(eval_sumo_seed),
        deterministic_eval=True,
        rebuild_env_each_evaluate=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args):
    global run_dir

    set_seed(args.seed)

    if args.resume and not args.parallel:
        raise ValueError("--resume is only supported with --parallel (the sequential path "
                          "rebuilds clients/state each round and has no matching restore).")
    if args.resume and args.baseline_controller != "none":
        raise ValueError("--resume doesn't apply to --baseline_controller runs "
                          "(no training rounds/checkpoints to resume).")

    resume_ckpt_path = None
    resume_completed_round = 0
    if args.resume:
        run_dir, resume_ckpt_path, resume_completed_round = resolve_resume(args.resume)
        os.makedirs(run_dir, exist_ok=True)
    else:
        # PID suffix: without it, two processes launched within the same
        # wall-clock second (e.g. several concurrent runs kicked off by a
        # batch script) compute the identical run_dir string. `exist_ok=True`
        # below would then silently let the second process write into the
        # first's directory -- interleaved checkpoints/logs/history from two
        # different seeds in one folder, no error, no warning. The PID makes
        # collisions require literal PID reuse (not a real risk in practice).
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        run_dir = os.path.join("results", f"run_{timestamp}_{os.getpid()}")
        os.makedirs("results", exist_ok=True)
        os.makedirs(run_dir, exist_ok=False)

    # logging.FileHandler defaults to append mode, so re-pointing it at an
    # existing run_dir's log on --resume naturally continues the same file
    # rather than overwriting it.
    log_file = os.path.join(run_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("Arguments:\n%s", pprint.pformat(vars(args), sort_dicts=True))
    logger.info(
        "config: strategy=%s head_fix=%s neighbor_attention=%s eval_episodes=%d rounds=%d",
        args.aggregation_strategy,
        not args.disable_head_fix,
        not args.disable_neighbor_attention,
        args.eval_episodes,
        args.rounds,
    )
    if args.resume:
        logger.info(
            "Resuming from %s (completed round %d) -- continuing to round %d in %s",
            resume_ckpt_path, resume_completed_round, args.rounds, run_dir,
        )

    if args.no_federation and args.aggregation_strategy != "fedavg":
        logger.warning(
            "--no_federation ignores aggregation strategy '%s' because aggregation is skipped.",
            args.aggregation_strategy,
        )

    if args.base_dir is not None:
        base = args.base_dir
    else:
        base = "environments"

    # Resolved comm-dropout severity: each of the three probabilities falls
    # back independently to DEFAULT_COMM_DROPOUT's value when its CLI flag
    # is left unset (None), so e.g. only overriding p_isolate doesn't reset
    # p_link/p_hop_cutoff to 0. Applied identically to training and eval
    # envs unless a call site overrides it -- there was previously no CLI
    # surface for this at all, only the hardcoded DEFAULT_COMM_DROPOUT.
    comm_dropout_cfg = dict(DEFAULT_COMM_DROPOUT)
    if args.comm_dropout_p_link is not None:
        comm_dropout_cfg["p_link"] = args.comm_dropout_p_link
    if args.comm_dropout_p_isolate is not None:
        comm_dropout_cfg["p_isolate"] = args.comm_dropout_p_isolate
    if args.comm_dropout_p_hop_cutoff is not None:
        comm_dropout_cfg["p_hop_cutoff"] = args.comm_dropout_p_hop_cutoff
    if comm_dropout_cfg != DEFAULT_COMM_DROPOUT:
        logger.info("Comm-dropout override: %s (default was %s)",
                    comm_dropout_cfg, DEFAULT_COMM_DROPOUT)

    if args.baseline_controller != "none":
        city_configs, obs_dims, action_dim, _steps_per_ep = resolve_city_configs_and_dims(base)
        evaluator = make_holdout_evaluator(
            base,
            obs_dims,
            action_dim,
            episodes=args.eval_episodes,
            holdout_base_dir=args.eval_base_dir,
            eval_sumo_seed=args.eval_sumo_seed,
            eval_comm_dropout_cfg=comm_dropout_cfg,
        )
        if evaluator is None:
            raise RuntimeError("Could not construct holdout evaluator for baseline controller run.")

        metrics = evaluator.evaluate_controller(args.baseline_controller)
        evaluator.close()

        history = {
            "round": [0],
            "client_samples": [0],
            "round_eps_start": [{}],
            "round_eps_end": [{}],
            "eval_reward": [metrics.get("mean_reward")],
            "eval_reward_std": [metrics.get("std_reward")],
            "eval_reward_episodes": [metrics.get("per_episode_reward")],
            "eval_waiting_time": [metrics.get("mean_waiting_time")],
            "eval_waiting_time_std": [metrics.get("std_waiting_time")],
            "eval_waiting_time_episodes": [metrics.get("per_episode_waiting_time")],
            "eval_stopped": [metrics.get("mean_stopped")],
            "eval_stopped_std": [metrics.get("std_stopped")],
            "eval_stopped_episodes": [metrics.get("per_episode_stopped")],
            "eval_arrived": [metrics.get("mean_arrived")],
            "eval_action_counts": [metrics.get("action_counts")],
            "eval_q_gaps": [metrics.get("q_gaps")],
            "eval_mode": [f"baseline_{args.baseline_controller}"],
            "cluster_assignments": [None],
            "baseline_controller": args.baseline_controller,
            "baseline_only": True,
            "n_cities": len(city_configs),
        }
        history_path = os.path.join(run_dir, "federated_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(
            "Baseline controller '%s' holdout metrics | reward=%.4f waiting_time=%.2f stopped=%.2f arrived=%.2f",
            args.baseline_controller,
            float(metrics.get("mean_reward", 0.0)),
            float(metrics.get("mean_waiting_time", 0.0)),
            float(metrics.get("mean_stopped", 0.0)),
            float(metrics.get("mean_arrived", 0.0)),
        )
        logger.info("Baseline-only history saved to %s", history_path)
        return

    if args.parallel:
        city_configs, obs_dims, action_dim, steps_per_ep = resolve_city_configs_and_dims(base)
        own_dim, neighbor_dim, k_max = obs_dims

        eps_decay = compute_eps_decay(
            rounds=args.rounds,
            local_episodes=args.local_episodes,
            steps_per_episode=steps_per_ep,
            explore_fraction=args.explore_fraction,
        )
        logger.info(
            "[parallel] own_dim=%d neighbor_dim=%d k_max=%d action_dim=%d "
            "clients=%d eps_decay=%.1f",
            own_dim, neighbor_dim, k_max, action_dim, len(city_configs), eps_decay,
        )

        global_model = _make_agent(
            own_dim, neighbor_dim, k_max, action_dim, eps_decay,
            head_fix=not args.disable_neighbor_attention,
            tau=args.tau, target_update=args.target_update,
            mu=args.fedprox_mu,
            dueling=args.dueling,
            n_step=args.n_step,
        )

        start_round = 1
        init_steps_done = 0
        initial_history = None
        if args.resume:
            global_model.load(resume_ckpt_path)
            start_round = resume_completed_round + 1
            # Approximation, not an exact replay: assumes every completed
            # round ran the same step count (local_episodes * steps_per_ep),
            # which holds unless an episode terminated early. Close enough
            # to pick epsilon back up mid-decay instead of restarting
            # exploration at eps_start on a model that's already learned.
            init_steps_done = resume_completed_round * args.local_episodes * steps_per_ep
            history_path = os.path.join(run_dir, "federated_history.json")
            if os.path.exists(history_path):
                with open(history_path) as f:
                    initial_history = json.load(f)
            logger.info(
                "[parallel] Resume: start_round=%d init_steps_done=%d prior_history_rounds=%d",
                start_round, init_steps_done,
                len(initial_history["round"]) if initial_history else 0,
            )

        evaluator = make_holdout_evaluator(
            base,
            obs_dims,
            action_dim,
            episodes=args.eval_episodes,
            holdout_base_dir=args.eval_base_dir,
            eval_sumo_seed=args.eval_sumo_seed,
            eval_comm_dropout_cfg=comm_dropout_cfg,
        )

        aggregation_config = {
            "ema_beta": args.ema_beta,
            "survival_window": args.survival_window,
            "n_clusters": args.n_clusters,
        }
        # Per-city LR override: a city's own config.yaml may set `lr: ...`
        # to give it a different starting rate than the global default
        # (e.g. a lower rate for a small, noisy single-intersection city).
        per_city_lr = {
            name: float(cfg["lr"]) for name, cfg in city_configs if "lr" in cfg
        }
        if per_city_lr:
            logger.info("[parallel] Per-city LR overrides: %s", per_city_lr)

        server = ParallelFederatedServer(
            global_model=global_model,
            city_configs=city_configs,
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            k_max=k_max,
            action_dim=action_dim,
            eps_decay=eps_decay,                   # ← threaded through to workers
            comm_dropout_cfg=comm_dropout_cfg,
            local_episodes=args.local_episodes,
            log_loss_every_steps=args.log_loss_every_steps,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
            log_file=os.path.join(run_dir, "training.log"),
            aggregation_strategy=args.aggregation_strategy,
            aggregation_config=aggregation_config,
            default_lr=args.lr,
            lr_decay=args.lr_decay,
            min_lr=args.min_lr,
            per_city_lr=per_city_lr,
            head_fix=not args.disable_head_fix,
            neighbor_attention=not args.disable_neighbor_attention,
            no_federation=args.no_federation,
            fedavg_blend=args.fedavg_blend,
            tau=args.tau,
            target_update=args.target_update,
            seed=args.seed,
            mu=args.fedprox_mu,
            dueling=args.dueling,
            server_momentum=args.server_momentum,
            n_step=args.n_step,
            pseudo_grad_clip=args.pseudo_grad_clip,
            eval_ema_decay=args.eval_ema_decay,
            init_steps_done=init_steps_done,
        )
        history = server.run(
            rounds=args.rounds,
            eval_every=args.eval_every,
            start_round=start_round,
            initial_history=initial_history,
        )

        if evaluator:
            evaluator.close()

    else:
        clients, obs_dims, action_dim, eps_decay = load_clients(
            base_dir=base,
            rounds=args.rounds,
            local_episodes=args.local_episodes,
            explore_fraction=args.explore_fraction,
            log_loss_every_steps=args.log_loss_every_steps,
            comm_dropout_cfg=comm_dropout_cfg,
            head_fix=not args.disable_neighbor_attention,
            tau=args.tau,
            target_update=args.target_update,
            mu=args.fedprox_mu,
            dueling=args.dueling,
            n_step=args.n_step,
        )
        own_dim, neighbor_dim, k_max = obs_dims

        logger.info("Results directory : %s", run_dir)
        logger.info(
            "own_dim=%d neighbor_dim=%d k_max=%d action_dim=%d "
            "clients=%d eps_decay=%.1f",
            own_dim, neighbor_dim, k_max, action_dim, len(clients), eps_decay,
        )

        global_model = _make_agent(
            own_dim, neighbor_dim, k_max, action_dim, eps_decay,
            head_fix=not args.disable_neighbor_attention,
            tau=args.tau, target_update=args.target_update,
            mu=args.fedprox_mu,
            dueling=args.dueling,
            n_step=args.n_step,
        )
        evaluator = make_holdout_evaluator(
            base,
            obs_dims,
            action_dim,
            episodes=args.eval_episodes,
            holdout_base_dir=args.eval_base_dir,
            eval_sumo_seed=args.eval_sumo_seed,
            eval_comm_dropout_cfg=comm_dropout_cfg,
        )

        aggregation_config = {
            "ema_beta": args.ema_beta,
            "survival_window": args.survival_window,
            "n_clusters": args.n_clusters,
        }
        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
            aggregation_strategy=args.aggregation_strategy,
            aggregation_config=aggregation_config,
            use_masked_head=not args.disable_head_fix,
            no_federation=args.no_federation,
            fedavg_blend=args.fedavg_blend,
            dueling=args.dueling,
            server_momentum=args.server_momentum,
            pseudo_grad_clip=args.pseudo_grad_clip,
            eval_ema_decay=args.eval_ema_decay,
        )
        history = server.run(rounds=args.rounds, eval_every=args.eval_every)

        for c in clients:
            c.close()
        if evaluator:
            evaluator.close()

    global_model.save(os.path.join(run_dir, "global_fed.pth"))

    history_path = os.path.join(run_dir, "federated_history.json")

    logger.info("Federated training finished.")
    logger.info("History saved to %s", history_path)


if __name__ == "__main__":
    f = Figlet(font="slant", width=200)
    print("\n")
    print("#" * 100)
    print(f.renderText("FederatedTraining"))
    print("#" * 100)
    print("\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",               type=int,   default=10)
    parser.add_argument("--seed",                 type=int,   default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Continue an interrupted --parallel run instead of starting fresh. Pass either a "
             "run_dir (its latest global_round_*.pth is used) or a direct checkpoint path. "
             "Restores global model weights and the round counter, and approximates each "
             "worker's epsilon step count so exploration doesn't restart from eps_start -- but "
             "NOT each worker's replay buffer or optimizer momentum, which only ever lived in "
             "the killed process's memory. --rounds must be the original run's target total "
             "(e.g. the same --rounds 40), not a remaining-rounds count.",
    )
    parser.add_argument("--local_episodes",        type=int,   default=1)
    parser.add_argument("--eval_every",            type=int,   default=1)
    parser.add_argument("--eval_episodes",         type=int,   default=5)
    parser.add_argument("--log_loss_every_steps",  type=int,   default=50,
                        help="Print mid-episode loss every N steps (0 = end-of-episode only).")
    parser.add_argument("--explore_fraction",      type=float, default=0.5,
                        help="Fraction of total training steps to spend exploring. "
                             "eps_decay is computed automatically so epsilon reaches "
                             "its floor at this fraction of total steps. "
                             "Default 0.5 = explore first half, exploit second half.")
    parser.add_argument("--parallel", action="store_true",
                        help="Train all cities concurrently, one persistent OS process per "
                             "city. See federated/parallel_server.py.")
    parser.add_argument(
        "--disable_head_fix",
        action="store_true",
        help="Ablation flag: use naive uniform averaging on the head layer instead of "
            "masked_head_weighted_average. Used to reproduce the pre-fix behavior. "
            "Aggregation-time only -- independent of --disable_neighbor_attention (see "
            "below). Before 2026-08-15 these were accidentally the same underlying flag, "
            "confounding every masked-head ablation result with an attention-vs-pooling "
            "comparison in the Q-network itself; they're now decoupled.",
    )
    parser.add_argument(
        "--disable_neighbor_attention",
        action="store_true",
        help="Ablation flag: replace the Q-network's masked multi-head attention over "
            "neighbor_obs (NeighborAttentionQNetwork.forward, head_fix=True branch) with "
            "simple masked mean-pooling (the head_fix=False branch, originally built as "
            "part of --disable_head_fix before the two were split 2026-08-15). "
            "Network-forward-time only -- independent of --disable_head_fix (aggregation-"
            "time). Use this to directly test whether neighbor attention helps or hurts "
            "at a given roster size, without also changing how the aggregation step "
            "handles heterogeneous action-space widths.",
    )
    parser.add_argument("--aggregation_strategy", type=str, default="fedavg",
                        choices=["fedavg", "ema_loss", "ema_alignment",
                               "velocity_novelty", "gradient_survival", "clustered_fedavg"],
                        help="How to weight each client's update when aggregating. "
                             "'fedavg' = classic sample-weighted average (unchanged "
                             "default behavior). See federated/aggregation_strategies.py.")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of clusters for clustered_fedavg strategy.")
    parser.add_argument("--no_federation", action="store_true",
                        help="Train each city independently (no aggregation/broadcast across cities).")
    parser.add_argument("--baseline_controller", type=str, default="none",
                        choices=["none", "fixed_time", "max_pressure"],
                        help="Run holdout evaluation only with a rule-based controller and skip training.")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Polyak soft target-update coefficient. 0 = legacy hard copy every "
                             "--target_update steps; 0.005 = smooth update every step (default).")
    parser.add_argument("--target_update", type=int, default=200,
                        help="Hard target-network sync interval (steps). Only used when --tau 0.")
    parser.add_argument("--fedprox_mu", type=float, default=0.0,
                        help="FedProx proximal-term coefficient. Adds mu/2 * ||w - w_global||^2 "
                             "to each client's local training loss, penalizing drift from the "
                             "weights it started the round from. Swept 2026-08-06: no mu value "
                             "stabilized the seed-4 repro, mu=0.1 was measurably worse -- see "
                             "fidings/divergence_investigation.md sec 14. Not recommended; kept "
                             "for reference. 0 = disabled (default, exactly recovers plain local "
                             "training).")
    parser.add_argument("--dueling", action="store_true",
                        help="Dueling Q-head: split the final layer into V(s) (scalar, no "
                             "action_mask, aggregates cleanly across every city regardless of "
                             "action_dim) + A(s,a) (action-indexed, still masked-head-aggregated "
                             "same as the plain head), combined as Q = V + A - mean(A). Targets "
                             "the same action-indexed-head client-drift symptom as FedProx did, "
                             "structurally instead of via a loss penalty -- see "
                             "fidings/divergence_investigation.md sec 14. Default off (plain "
                             "single Linear head, unchanged behavior).")
    parser.add_argument("--server_momentum", type=float, default=0.0,
                        help="FedAvgM-style server-side momentum (0-1, typically ~0.9). Applies "
                             "this round's aggregated update through an exponentially-weighted "
                             "velocity buffer (velocity = beta*velocity_prev + (agg - global); "
                             "global += velocity) instead of jumping straight to the raw "
                             "aggregate. Targets the aggregated-model-level oscillation itself "
                             "(see fidings/divergence_investigation.md sec 9) rather than "
                             "anything client-side. 0 = disabled (default, exactly recovers "
                             "plain FedAvg).")
    parser.add_argument("--n_step", type=int, default=1,
                        help="n-step returns: accumulate n consecutive (clipped) rewards per "
                             "intersection before pushing a replay transition, bootstrapping "
                             "with gamma**n instead of gamma. 1 = disabled (default, exactly "
                             "recovers plain 1-step TD).")
    parser.add_argument("--pseudo_grad_clip", type=float, default=0.0,
                        help="Cap the total L2 norm of each round's aggregated update "
                             "(agg_state - global_state_before) at this value, rescaling "
                             "uniformly if over the cap. Cheap insurance against one bad round "
                             "moving the global model an outsized amount. Applied before "
                             "--server_momentum if both are set. 0 = disabled (default, exact "
                             "no-op).")
    parser.add_argument("--eval_ema_decay", type=float, default=0.0,
                        help="Evaluate (and report) a slowly-averaged EMA snapshot of the "
                             "global model each round instead of the raw just-aggregated "
                             "weights (eval_state = decay*eval_state + (1-decay)*global_state). "
                             "Purely a reporting-side smoothing -- never touches what's "
                             "broadcast to clients next round. 0 = disabled (default, exact "
                             "no-op, evaluates raw weights same as before).")
    parser.add_argument("--fedavg_blend", type=float, default=1.0,
                        help="FedAvg blending: 1.0 = fully replace global with aggregated (default). "
                             "0.7 = 70%% aggregated + 30%% previous global weights, preventing "
                             "single-round catastrophic forgetting.")
    parser.add_argument("--ema_beta", type=float, default=0.9,
                        help="Smoothing factor for EMA-based strategies (ema_loss, "
                             "ema_alignment, velocity_novelty).")
    parser.add_argument("--survival_window", type=int, default=3,
                        help="Rounds of gradient history kept for gradient_survival.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Default learning rate, used for any city without an explicit "
                             "'lr:' key in its own config.yaml.")
    parser.add_argument("--lr_decay", type=float, default=1.0,
                        help="Multiplicative LR decay applied once per federated round "
                             "(e.g. 0.97). 1.0 = no decay (default, unchanged behavior).")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Floor for lr_decay -- LR never drops below this.")
    parser.add_argument("--base_dir", default="environments",
                     help="Directory containing city_x subfolders with config.yaml")
    parser.add_argument(
        "--eval_base_dir",
        default=None,
        help="Optional directory to search for city_5_holdout when base_dir is a reduced subset.",
    )
    parser.add_argument(
        "--eval_sumo_seed",
        type=int,
        default=12345,
        help="Fixed SUMO seed for evaluation env so round-to-round comparisons use a deterministic scenario.",
    )
    parser.add_argument(
        "--comm_dropout_p_link", type=float, default=None,
        help="Override CommDropoutWrapper's per-neighbor-slot link-drop probability "
             "(applied every tick to both training and eval envs). Default (None) uses "
             "DEFAULT_COMM_DROPOUT's p_link=0.10 -- this has been the silent default on "
             "every run in this project's history; there was previously no CLI way to "
             "disable it. Pass 0 for clean (uncorrupted) neighbor communication.",
    )
    parser.add_argument(
        "--comm_dropout_p_isolate", type=float, default=None,
        help="Override CommDropoutWrapper's per-tick full-isolation probability (drops "
             "ALL neighbors for that intersection this tick). Default (None) uses "
             "DEFAULT_COMM_DROPOUT's p_isolate=0.05. Pass 1.0 to force every intersection "
             "isolated every tick -- an own-obs-only ablation (no neighbor info reaches "
             "the network at all) without touching agents/networks.py.",
    )
    parser.add_argument(
        "--comm_dropout_p_hop_cutoff", type=float, default=None,
        help="Override CommDropoutWrapper's random-hop-cutoff probability (drops every "
             "neighbor farther than a randomly chosen hop each tick). Default (None) uses "
             "DEFAULT_COMM_DROPOUT's p_hop_cutoff=0.10. Pass 0 to disable.",
    )
    args = parser.parse_args()
    if args.dueling and args.server_momentum > 0.0:
        # Measured net-negative interaction, not a theoretical concern: dueling
        # alone beat dueling+server_momentum on both mean AND best-round reward,
        # on both the 2-city and 3-city seed-4 repro rosters -- server-side
        # momentum damps exactly the fast, undamped advantage-head movement that
        # makes dueling effective on its own. See
        # fidings/divergence_investigation.md sec 18 before overriding this.
        parser.error(
            "--dueling and --server_momentum are both set, but this combination "
            "was tested (fidings/divergence_investigation.md sec 18) and is "
            "net-negative -- dueling alone beats dueling+momentum on mean and "
            "best-round reward on every roster tested. Use --dueling by itself, "
            "or --server_momentum by itself (weaker but non-negative). If you "
            "have new evidence this combination helps in some other setting, "
            "update sec 18 and remove/relax this check rather than silently "
            "bypassing it."
        )
    main(args)
