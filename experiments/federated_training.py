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
import logging
import os
import pprint
import sys
import yaml
import json

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
from federated.utils import compute_eps_decay

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


def _make_agent(own_dim, neighbor_dim, k_max, action_dim, eps_decay):
    """Single place that constructs a DQNAgent with the computed eps_decay."""
    return DQNAgent(
        own_dim=own_dim,
        neighbor_dim=neighbor_dim,
        k_max=k_max,
        action_dim=action_dim,
        eps_decay=eps_decay,
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
        ):
            return _make_agent(_own, _nbr, _k, _act, _eps)

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
    episodes: int = 1,
    eval_comm_dropout_cfg: dict | None = None,
) -> "HoldoutEvaluator | None":
    cfg_path = os.path.join(base_dir, "city_5_holdout", "config.yaml")
    if not os.path.exists(cfg_path):
        logger.warning("No holdout city found at %s, skipping evaluator.", cfg_path)
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    own_dim, neighbor_dim, k_max = obs_dims
    dropout_cfg = eval_comm_dropout_cfg if eval_comm_dropout_cfg is not None else DEFAULT_COMM_DROPOUT

    def build_holdout_env():
        env = build_federated_env(cfg)
        if env.own_dim != own_dim or env.neighbor_dim != neighbor_dim or env.k_max != k_max:
            raise RuntimeError(
                f"Holdout city obs shape ({env.own_dim},{env.neighbor_dim},{env.k_max}) "
                f"!= training obs shape ({own_dim},{neighbor_dim},{k_max})."
            )
        if env.max_action_dim > action_dim:
            raise RuntimeError(
                f"Holdout city max_action_dim={env.max_action_dim} exceeds "
                f"global action_dim={action_dim}."
            )
        env = ActionMaskPadder(env, action_dim)
        if dropout_cfg:
            env = CommDropoutWrapper(env, **dropout_cfg)
        return env

    return HoldoutEvaluator(env_builder=build_holdout_env, episodes=episodes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    global run_dir

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs("results", exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

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

    base = "environments"

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

        global_model = _make_agent(own_dim, neighbor_dim, k_max, action_dim, eps_decay)
        evaluator    = make_holdout_evaluator(base, obs_dims, action_dim, episodes=args.eval_episodes)

        aggregation_config = {
            "ema_beta": args.ema_beta,
            "survival_window": args.survival_window,
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
            comm_dropout_cfg=DEFAULT_COMM_DROPOUT,
            local_episodes=args.local_episodes,
            log_loss_every_steps=args.log_loss_every_steps,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
            aggregation_strategy=args.aggregation_strategy,
            aggregation_config=aggregation_config,
            default_lr=args.lr,
            lr_decay=args.lr_decay,
            min_lr=args.min_lr,
            per_city_lr=per_city_lr,
        )
        history = server.run(rounds=args.rounds, eval_every=args.eval_every)

        if evaluator:
            evaluator.close()

    else:
        clients, obs_dims, action_dim, eps_decay = load_clients(
            base_dir=base,
            rounds=args.rounds,
            local_episodes=args.local_episodes,
            explore_fraction=args.explore_fraction,
            log_loss_every_steps=args.log_loss_every_steps,
        )
        own_dim, neighbor_dim, k_max = obs_dims

        logger.info("Results directory : %s", run_dir)
        logger.info(
            "own_dim=%d neighbor_dim=%d k_max=%d action_dim=%d "
            "clients=%d eps_decay=%.1f",
            own_dim, neighbor_dim, k_max, action_dim, len(clients), eps_decay,
        )

        global_model = _make_agent(own_dim, neighbor_dim, k_max, action_dim, eps_decay)
        evaluator    = make_holdout_evaluator(base, obs_dims, action_dim, episodes=args.eval_episodes)

        aggregation_config = {
            "ema_beta": args.ema_beta,
            "survival_window": args.survival_window,
        }
        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
            aggregation_strategy=args.aggregation_strategy,
            aggregation_config=aggregation_config,
        )
        history = server.run(rounds=args.rounds, eval_every=args.eval_every)

        for c in clients:
            c.close()
        if evaluator:
            evaluator.close()

    global_model.save(os.path.join(run_dir, "global_fed.pth"))

    history_path = os.path.join(run_dir, "federated_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

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
    parser.add_argument("--rounds",               type=int,   default=5)
    parser.add_argument("--local_episodes",        type=int,   default=1)
    parser.add_argument("--eval_every",            type=int,   default=1)
    parser.add_argument("--eval_episodes",         type=int,   default=1)
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
    parser.add_argument("--aggregation_strategy", type=str, default="fedavg",
                        choices=["fedavg", "ema_loss", "ema_alignment",
                                 "velocity_novelty", "gradient_survival"],
                        help="How to weight each client's update when aggregating. "
                             "'fedavg' = classic sample-weighted average (unchanged "
                             "default behavior). See federated/aggregation_strategies.py.")
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
    args = parser.parse_args()
    main(args)
