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
import copy
import logging
import os
import random
import sys
import yaml
import json

import numpy as np
import torch
from pyfiglet import Figlet

from federated.server import FederatedServer
from federated.parallel_server import ParallelFederatedServer
from federated.client import CityClient
from federated.evaluator import HoldoutEvaluator
from federated.comm_dropout import CommDropoutWrapper
from environments.federated_env import build_federated_env, ActionMaskPadder
from agents.dqn import DQNAgent
from federated.utils import compute_eps_decay

timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
run_dir = os.path.join("results", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

logger = logging.getLogger(__name__)


def seed_everything(seed: int | None = None) -> None:
    """Make the training run deterministic when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# Default communication-unreliability profile applied during TRAINING.
# Tune per-experiment; set all to 0.0 to train under "perfect comms".
DEFAULT_COMM_DROPOUT = dict(
    p_link=0.10,        # per-neighbor-slot chance of dropping, every tick
    p_isolate=0.05,      # chance this intersection loses ALL neighbors this tick
    p_hop_cutoff=0.10,   # chance of a cascading "cut beyond hop h" failure
)


# ---------------------------------------------------------------------------
# Client loading
# ---------------------------------------------------------------------------

def load_city_configs(base_dir: str, seed: int | None = None) -> list:
    city_configs = []
    for name in sorted(os.listdir(base_dir)):
        if name == "city_5_holdout":
            continue

        cfg_path = os.path.join(base_dir, name, "config.yaml")
        if not os.path.exists(cfg_path):
            continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        cfg = copy.deepcopy(cfg)
        if seed is not None:
            cfg["seed"] = seed
            cfg["sumo_seed"] = seed
        city_configs.append((name, cfg))

    if not city_configs:
        raise RuntimeError(f"No city configs found under '{base_dir}'.")

    return city_configs


def infer_steps_per_episode(city_configs: list) -> int:
    steps_per_episode = []
    for _, cfg in city_configs:
        num_seconds = cfg.get("num_seconds")
        delta_time = cfg.get("delta_time")
        if num_seconds is None or delta_time in (None, 0):
            continue
        steps_per_episode.append(int(num_seconds / delta_time))

    if not steps_per_episode:
        raise RuntimeError("Could not infer steps_per_episode from city configs.")

    unique_steps = {s for s in steps_per_episode}
    if len(unique_steps) != 1:
        raise RuntimeError(f"Inconsistent steps_per_episode values: {sorted(unique_steps)}")

    return unique_steps.pop()


def load_clients(
    base_dir: str,
    local_episodes: int,
    log_loss_every_steps: int = 50,
    comm_dropout_cfg: dict | None = None,
    eps_decay: float = 20000.0,
    seed: int | None = None,
) -> tuple:
    """Build one CityClient per city directory.

    Every city needs only ``config.yaml`` + ``net.xml`` + ``routes.xml``.
    No ``phase_mapping``, no per-city adapter code: ``build_federated_env``
    is responsible for discovering intersections, building the neighbor
    graph, and exposing the fixed-size dict-observation contract described
    in ``agents/networks.py``. It must expose on the returned env:

        env.own_dim          int   -- dim of the per-intersection own obs
        env.neighbor_dim     int   -- dim of a single neighbor's feature vec
        env.k_max            int   -- max neighbor slots (any hop, 1..K)
        env.max_action_dim   int   -- max number of valid actions among
                                       this city's intersections
        env.reset()          -> Dict[ts_id, obs_dict]
        env.step(actions)    -> (Dict[ts_id, obs_dict],
                                  Dict[ts_id, float] rewards,
                                  Dict[ts_id, bool] dones (+ "__all__"),
                                  info dict)

    ``own_dim``, ``neighbor_dim`` and ``k_max`` must be identical across
    cities (they define the shared network's input shapes). Action
    counts do NOT need to match across intersections or cities -- the
    global model is sized to ``max(max_action_dim)`` and each
    intersection's ``action_mask`` hides the slots that don't apply to it.

    Returns:
        clients   list of CityClient
        obs_dims  (own_dim, neighbor_dim, k_max) shared across all cities
        action_dim  shared Q-output size (== max over all intersections)
    """
    comm_dropout_cfg = comm_dropout_cfg or DEFAULT_COMM_DROPOUT

    # Pass 1: build every city's raw federated env so we know each one's
    # dims (own_dim/neighbor_dim/k_max must match; max_action_dim doesn't
    # need to, since it gets padded to the global max below).
    city_envs = []
    for name, cfg in load_city_configs(base_dir, seed=seed):
        env = build_federated_env(cfg)
        city_envs.append((name, env))

    own_dims = {env.own_dim for _, env in city_envs}
    neighbor_dims = {env.neighbor_dim for _, env in city_envs}
    k_maxs = {env.k_max for _, env in city_envs}

    if len(own_dims) != 1:
        raise RuntimeError(f"Cities produce different own_dim values: {own_dims}.")
    if len(neighbor_dims) != 1:
        raise RuntimeError(f"Cities produce different neighbor_dim values: {neighbor_dims}.")
    if len(k_maxs) != 1:
        raise RuntimeError(f"Cities use different k_max (neighbor slot count) values: {k_maxs}.")

    own_dim, neighbor_dim, k_max = own_dims.pop(), neighbor_dims.pop(), k_maxs.pop()
    # Global action_dim = widest action space among ALL intersections in
    # ALL cities (e.g. a 5-way with protected lefts elsewhere). Cities
    # with narrower local action spaces get their action_mask padded with
    # zeros for the unused slots -- see ActionMaskPadder.
    action_dim = max(env.max_action_dim for _, env in city_envs)

    # Pass 2: pad each city up to the global action_dim, then layer on
    # communication-dropout simulation for training.
    city_envs = [
        (name, CommDropoutWrapper(ActionMaskPadder(env, action_dim), **comm_dropout_cfg))
        for name, env in city_envs
    ]

    clients = []
    for name, env in city_envs:
        # NOTE: default-arg capture (e=env) avoids the classic late-binding
        # closure bug -- each lambda is bound to ITS OWN env at definition
        # time, not whatever `env` happens to be when the loop finishes.
        def make_agent(e=env):
            return DQNAgent(
                own_dim=own_dim,
                neighbor_dim=neighbor_dim,
                k_max=k_max,
                action_dim=action_dim,
                eps_decay=eps_decay,
            )

        clients.append(
            CityClient(
                name=name,
                env_builder=lambda e=env: e,
                agent_builder=make_agent,
                local_episodes=local_episodes,
                log_loss_every_steps=log_loss_every_steps,
            )
        )

    return clients, (own_dim, neighbor_dim, k_max), action_dim


def resolve_city_configs_and_dims(base_dir: str, seed: int | None = None) -> tuple:
    """Used by the --parallel path: we need every city's raw config dict
    (to hand to worker processes, which build their OWN env instances --
    a live SUMO env generally can't cross a process boundary), plus the
    shared dims that size the network. Builds each env once just to read
    its dims/close it again; workers build fresh copies for real training.
    """
    city_configs = []
    dims = []
    for name, cfg in load_city_configs(base_dir, seed=seed):
        env = build_federated_env(cfg)
        dims.append((env.own_dim, env.neighbor_dim, env.k_max, env.max_action_dim))
        env.close()
        city_configs.append((name, cfg))

    own_dims = {d[0] for d in dims}
    neighbor_dims = {d[1] for d in dims}
    k_maxs = {d[2] for d in dims}
    if len(own_dims) != 1 or len(neighbor_dims) != 1 or len(k_maxs) != 1:
        raise RuntimeError(
            f"Mismatched dims across cities: own_dims={own_dims} "
            f"neighbor_dims={neighbor_dims} k_maxs={k_maxs}"
        )
    own_dim, neighbor_dim, k_max = own_dims.pop(), neighbor_dims.pop(), k_maxs.pop()
    action_dim = max(d[3] for d in dims)
    return city_configs, (own_dim, neighbor_dim, k_max), action_dim


# ---------------------------------------------------------------------------
# Holdout evaluator
# ---------------------------------------------------------------------------

def make_holdout_evaluator(
    base_dir: str,
    obs_dims: tuple,
    action_dim: int,
    episodes: int = 1,
    eval_comm_dropout_cfg: dict | None = None,
    output_dir: str | None = None,
    eval_seeds: int = 3,
    include_baselines: bool = False,
) -> "HoldoutEvaluator | None":
    cfg_path = os.path.join(base_dir, "city_5_holdout", "config.yaml")
    if not os.path.exists(cfg_path):
        logger.warning("No holdout city found at %s, skipping evaluator.", cfg_path)
        return None

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    own_dim, neighbor_dim, k_max = obs_dims
    # Default: evaluate under realistic (not zero) comm dropout too, since
    # that's the regime the model actually has to operate in. Pass an
    # empty dict to eval_comm_dropout_cfg for a "perfect comms" reading.
    dropout_cfg = eval_comm_dropout_cfg if eval_comm_dropout_cfg is not None else DEFAULT_COMM_DROPOUT

    def build_holdout_env():
        env = build_federated_env(cfg)
        if env.own_dim != own_dim or env.neighbor_dim != neighbor_dim or env.k_max != k_max:
            raise RuntimeError(
                f"Holdout city obs shape ({env.own_dim},{env.neighbor_dim},{env.k_max}) "
                f"!= training obs shape ({own_dim},{neighbor_dim},{k_max}). "
                "Check the universal lane encoder / neighbor graph config."
            )
        if env.max_action_dim > action_dim:
            raise RuntimeError(
                f"Holdout city max_action_dim={env.max_action_dim} exceeds the "
                f"global training action_dim={action_dim}; the global model "
                "can't represent that many actions. Retrain with a holdout-"
                "aware action_dim, or shrink the holdout intersection set."
            )
        env = ActionMaskPadder(env, action_dim)
        if dropout_cfg:
            env = CommDropoutWrapper(env, **dropout_cfg)
        return env

    return HoldoutEvaluator(
        env_builder=build_holdout_env,
        episodes=episodes,
        output_dir=output_dir,
        eval_seeds=eval_seeds,
        include_baselines=include_baselines,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    seed_everything(args.seed)

    base = "environments"
    city_configs_for_schedule = load_city_configs(base, seed=args.seed)
    steps_per_episode = infer_steps_per_episode(city_configs_for_schedule)
    eps_decay = compute_eps_decay(
        rounds=args.rounds,
        local_episodes=args.local_episodes,
        steps_per_episode=steps_per_episode,
        explore_fraction=args.explore_fraction,
    )

    if args.parallel:
        city_configs, obs_dims, action_dim = resolve_city_configs_and_dims(base, seed=args.seed)
        own_dim, neighbor_dim, k_max = obs_dims

        logger.info(
            "[parallel] own_dim=%d neighbor_dim=%d k_max=%d action_dim=%d clients=%d",
            own_dim, neighbor_dim, k_max, action_dim, len(city_configs),
        )

        global_model = DQNAgent(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            k_max=k_max,
            action_dim=action_dim,
            eps_decay=eps_decay,
        )
        evaluator = make_holdout_evaluator(
            base,
            obs_dims,
            action_dim,
            episodes=args.eval_episodes,
            output_dir=run_dir,
            eval_seeds=args.eval_seeds,
            include_baselines=args.include_baselines,
        )

        server = ParallelFederatedServer(
            global_model=global_model,
            city_configs=city_configs,
            own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
            comm_dropout_cfg=DEFAULT_COMM_DROPOUT,
            local_episodes=args.local_episodes,
            log_loss_every_steps=args.log_loss_every_steps,
            eps_decay=eps_decay,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
        )
        history = server.run(rounds=args.rounds, eval_every=args.eval_every)

        if evaluator:
            evaluator.close()

    else:
        clients, obs_dims, action_dim = load_clients(
            base,
            args.local_episodes,
            args.log_loss_every_steps,
            eps_decay=eps_decay,
            seed=args.seed,
        )
        own_dim, neighbor_dim, k_max = obs_dims

        logger.info("Results directory : %s", run_dir)
        logger.info(
            "own_dim=%d neighbor_dim=%d k_max=%d action_dim=%d clients=%d",
            own_dim, neighbor_dim, k_max, action_dim, len(clients),
        )

        global_model = DQNAgent(
            own_dim=own_dim,
            neighbor_dim=neighbor_dim,
            k_max=k_max,
            action_dim=action_dim,
            eps_decay=eps_decay,
        )

        evaluator = make_holdout_evaluator(
            base,
            obs_dims,
            action_dim,
            episodes=args.eval_episodes,
            output_dir=run_dir,
            eval_seeds=args.eval_seeds,
            include_baselines=args.include_baselines,
        )

        server = FederatedServer(
            global_model=global_model,
            clients=clients,
            evaluator=evaluator,
            checkpoint_dir=run_dir,
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
    parser.add_argument("--rounds",          type=int, default=5)
    parser.add_argument("--local_episodes",  type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used for reproducible training and evaluation.")
    parser.add_argument("--eval_every",      type=int, default=1)
    parser.add_argument("--eval_episodes",        type=int, default=1)
    parser.add_argument("--eval_seeds", type=int, default=3,
                        help="Number of distinct evaluation seeds to use for the baseline policies; the trained policy always uses the first seed.")
    parser.add_argument("--include_baselines", action="store_true",
                        help="Compare the trained policy against simple baselines during evaluation.")
    parser.add_argument("--log_loss_every_steps", type=int, default=50,
                        help="Print mid-episode loss every N steps (0 = end-of-episode only).")
    parser.add_argument("--explore_fraction", type=float, default=0.5,
                        help="Fraction of total training steps over which epsilon decays to its floor.")
    parser.add_argument("--parallel", action="store_true",
                        help="Train all cities concurrently, one persistent OS process per "
                             "city, instead of sequentially. Biggest win when you have >=2 "
                             "cities and spare CPU cores -- see federated/parallel_server.py.")
    args = parser.parse_args()

    log_file = os.path.join(run_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    main(args)
