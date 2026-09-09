"""Progressive Curriculum FedAvg (PCFT), a bespoke design per direct user
request -- not an existing named method, a synthesis of two things this
project has already confirmed:

  1. Sequential (non-federated) curriculum training substantially beats
     parallel FedAvg on cross-topology holdout generalization
     (fidings/divergence_investigation.md sec 85/86), at the cost of real,
     measured catastrophic forgetting on earlier-trained cities.
  2. Fine-tuning a shared checkpoint onto a NEW city works reliably
     (sec 66-70) -- the mechanism that makes sequential training's per-city
     jump-start so effective in the first place.

Idea: order training cities from SIMPLEST to most complex (by intersection
count, per direct user request -- the opposite of sec 85/86's incidental
city_1(16 intersections) -> city_4(3) -> city_6(7) ordering, which happened
to start with the MOST complex city). Warm up solo on the simplest city.
Each time a new (more complex) city is introduced, give it a short FOCUS
phase -- fine-tune the current shared weights on just that city alone,
exactly sec 66-70's proven mechanism -- before folding it into a genuine
multi-city FedAvg pool for several rounds. Already-active cities keep their
OWN persistent DQNAgent/replay buffer across FedAvg rounds (matching the
real federated pipeline's warm-start convention, agent.start_round() each
round) so they stay anchored against forgetting while the newly-focused
city's adaptation gets blended in via ordinary FedAvg.

Mechanically, in complexity order city[0..N-1]:
  Phase 0 (warm-up):  solo train on city[0] for --warmup_episodes.
  For each city[i], i=1..N-1:
    Phase i.a (focus): a FRESH DQNAgent for city[i], starting from the
                        current shared weights, fine-tuned solo on city[i]
                        alone for --focus_episodes.
    Phase i.b (fedavg): round-robin FedAvg across city[0..i] for
                        --fedavg_rounds rounds. Each active city's own
                        persistent agent is synced to the shared weights at
                        the start of every round (DQNAgent.start_round(),
                        matching the real pipeline) then trained
                        --local_episodes, and the resulting state_dicts are
                        combined via federated/aggregation.py::aggregate_round
                        (masked-head aggregation, the same primitive the
                        real --parallel pipeline uses).
Holdout eval after every phase; a final catastrophic-forgetting check
(each active city's own in-distribution performance) at the end, same
methodology as diagnostics/sequential_training.py.

Usage:
    python diagnostics/progressive_curriculum_fedavg.py \\
        --base_dir environments_c1_4_6 --pad_to_true_holdout \\
        --warmup_episodes 10 --focus_episodes 5 --fedavg_rounds 5 \\
        --local_episodes 2 --seed 3
"""
import argparse
import logging
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from agents.dqn import DQNAgent
from environments.federated_env import ActionMaskPadder, build_federated_env
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)
from federated.aggregation import aggregate_round, head_key_names
from federated.evaluator import HoldoutEvaluator
from federated.utils import set_seed

logger = logging.getLogger(__name__)


def _n_intersections(cfg, action_dim) -> int:
    env = build_federated_env(cfg)
    env = ActionMaskPadder(env, action_dim)
    env.reset()
    n = len(env.ts_ids)
    env.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--warmup_episodes", type=int, default=10,
                     help="Solo training episodes on the simplest city before any other city "
                          "is introduced.")
    ap.add_argument("--focus_episodes", type=int, default=5,
                     help="Solo fine-tune episodes for each new city, alone, starting from the "
                          "current shared weights, BEFORE it joins the FedAvg pool.")
    ap.add_argument("--fedavg_rounds", type=int, default=5,
                     help="Federated rounds run across the active city pool after each new "
                          "city's focus phase.")
    ap.add_argument("--local_episodes", type=int, default=2,
                     help="Per-city local episodes within each FedAvg round (matches the "
                          "standard --local_episodes convention).")
    ap.add_argument("--q_entropy_weight", type=float, default=0.05,
                     help="Matches the current standard baseline used throughout this project's "
                          "item-2X/sequential-training pilots.")
    ap.add_argument("--phase_relational", action="store_true",
                    help="Use the per-phase Q scorer instead of the action-indexed head "
                         "(fidings sec 96). PCFT changes the training CURRICULUM and this "
                         "changes the HEAD -- they are orthogonal and compose.")
    ap.add_argument("--city_order", default="complexity",
                    choices=["complexity", "reverse", "shuffled"],
                    help="Order cities are phased in. 'complexity' is PCFT as "
                         "proposed (simplest first). 'reverse'/'shuffled' keep "
                         "budget and focus phases identical and change only the "
                         "order -- the control that isolates the curriculum "
                         "from the fine-tuning (sec 87 confound, sec 101).")
    ap.add_argument("--run_dir", default=None,
                    help="Where to write pcft_state.pt (the resume checkpoint). "
                         "Defaults to results/pcft_runs/<order>_s<seed>_<head>.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from --run_dir's pcft_state.pt, skipping the "
                         "steps it records as done. Every step ends in a holdout "
                         "eval, so those are the resume boundaries. Per-city "
                         "replay buffers and optimizer state are NOT restored -- "
                         "same limitation as federated_training's --resume, so a "
                         "resumed run is not bit-identical to an uninterrupted one.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--log_loss_every_steps", type=int, default=50)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)

    city_configs, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    agent_kwargs = dict(
        own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max, action_dim=action_dim,
        q_entropy_weight=args.q_entropy_weight,
        phase_relational=args.phase_relational,
    )
    # The phase-relational head has NO per-action rows (one shared scorer over
    # [state, phase features]), so per-action row aggregation has nothing to
    # index -- pass keys that don't exist and masked-head aggregation silently
    # degrades to plain averaging, which is the sec 10/sec 24 bug class. Use
    # sentinel keys so aggregate_round takes the plain path deliberately.
    if args.phase_relational:
        head_weight_key = head_bias_key = "__no_indexed_head__"
    else:
        head_weight_key, head_bias_key = head_key_names(dueling=False)

    # --city_order is the control that separates PCFT's CURRICULUM from the
    # per-city focus fine-tuning embedded in every one of its steps. sec 87
    # confirmed PCFT at 6 seeds but flagged that confound as unresolved, and
    # sec 101 found the curriculum-over-clients idea is already published
    # (Vahidian et al., ICCV 2023), so ordering is now the whole claim. 'reverse'
    # and 'shuffled' keep the budget and the focus phases identical and change
    # ONLY the order, which is what makes the comparison interpretable.
    ranked = sorted(
        ((name, cfg, _n_intersections(cfg, action_dim)) for name, cfg in city_configs),
        key=lambda t: t[2],
    )
    if args.city_order == "complexity":
        logger.info("City order: by intersection count, simplest first (PCFT as proposed).")
    elif args.city_order == "reverse":
        ranked = list(reversed(ranked))
        logger.info("City order: REVERSED (most complex first) -- ordering control.")
    elif args.city_order == "shuffled":
        rng = random.Random(args.seed)
        rng.shuffle(ranked)
        logger.info("City order: SHUFFLED on seed %d -- ordering control.", args.seed)
    else:
        raise ValueError(f"unknown --city_order {args.city_order!r}")
    for name, _, n in ranked:
        logger.info("  %s: %d intersections", name, n)
    ordered_cities = [(name, cfg) for name, cfg, _ in ranked]

    evaluator = make_holdout_evaluator(
        args.base_dir, (own_dim, neighbor_dim, k_max), action_dim, episodes=args.eval_episodes,
    )
    if evaluator is None:
        raise RuntimeError("Could not construct holdout evaluator.")

    envs = {}

    def get_env(name, cfg):
        if name not in envs:
            e = build_federated_env(cfg)
            envs[name] = ActionMaskPadder(e, action_dim)
        return envs[name]

    def eval_holdout(label, shared_state):
        probe = DQNAgent(**agent_kwargs)
        probe.load_state_dict(shared_state)
        result = evaluator.evaluate(probe)
        logger.info(
            "%s -- HOLDOUT: mean_reward=%.2f std_reward=%.2f",
            label, result["mean_reward"], result["std_reward"],
        )
        return result

    def eval_in_distribution(name, cfg, shared_state, label):
        def env_builder(cfg=cfg):
            e = build_federated_env(cfg)
            return ActionMaskPadder(e, action_dim)
        city_eval = HoldoutEvaluator(env_builder=env_builder, episodes=args.eval_episodes)
        probe = DQNAgent(**agent_kwargs)
        probe.load_state_dict(shared_state)
        result = city_eval.evaluate(probe)
        city_eval.close()
        logger.info(
            "%s -- %s (in-distribution): mean_reward=%.2f std_reward=%.2f",
            label, name, result["mean_reward"], result["std_reward"],
        )
        return result

    # ---- Explicit step plan, so a run can be checkpointed and resumed ----
    # Every step below both advances `shared_state` and ends in a holdout eval,
    # which makes it a clean resume boundary. Steps are numbered from 1.
    steps = [("warmup", ordered_cities[0][0], 0)]
    for name, _ in ordered_cities[1:]:
        steps.append(("focus", name, 0))
        for r in range(1, args.fedavg_rounds + 1):
            steps.append(("fedavg", name, r))

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = os.path.join(
            "results", "pcft_runs",
            f"{args.city_order}_s{args.seed}"
            f"{'_phase' if args.phase_relational else '_indexed'}")
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "pcft_state.pt")
    logger.info("PCFT run dir: %s (%d steps)", run_dir, len(steps))

    done_steps, shared_state, history = 0, None, []
    if args.resume:
        if not os.path.exists(state_path):
            logger.warning("--resume: no %s, starting from scratch.", state_path)
        else:
            ck = torch.load(state_path, map_location="cpu", weights_only=False)
            done_steps = int(ck["done_steps"])
            shared_state = ck["shared_state"]
            history = ck.get("history", [])
            if ck.get("plan") != [list(s) for s in steps]:
                raise ValueError(
                    "--resume: checkpoint's step plan differs from this run's "
                    "(different --city_order / --fedavg_rounds / roster?). Refusing "
                    "to resume into a different experiment.")
            logger.info(
                "RESUMING from %s: %d/%d steps already done, continuing at step %d. "
                "NOTE: per-city replay buffers and optimizer state are NOT restored "
                "(same limitation as federated_training's --resume) -- a resumed run "
                "is not bit-identical to an uninterrupted one.",
                state_path, done_steps, len(steps), done_steps + 1)

    def save_state(done):
        torch.save({"done_steps": done, "shared_state": shared_state,
                    "history": history, "plan": [list(s) for s in steps],
                    "seed": args.seed, "city_order": args.city_order,
                    "phase_relational": args.phase_relational}, state_path)

    def record(label, result):
        history.append({"step": len(history) + 1, "label": label,
                        "mean_reward": result["mean_reward"],
                        "std_reward": result["std_reward"]})

    if done_steps == 0:
        eval_holdout("Stage 0 (random init, before any training)",
                     DQNAgent(**agent_kwargs).state_dict())

    # Rebuild the active-agent pool. On a fresh run this fills in as cities are
    # phased in; on a resume it is reconstructed from the step plan, with each
    # agent loaded from the shared weights (replay/optimizer state is gone --
    # see the resume note above).
    active_agents = {}

    def ensure_agent(city_name):
        if city_name not in active_agents:
            ag = DQNAgent(**agent_kwargs)
            if shared_state is not None:
                ag.load_state_dict(shared_state)
            active_agents[city_name] = ag
        return active_agents[city_name]

    if done_steps:
        for kind, name, _ in steps[:done_steps]:
            if kind in ("warmup", "focus"):
                ensure_agent(name)
        logger.info("Resumed active pool: %s", list(active_agents))

    cfg_of = {n: c for n, c in ordered_cities}

    for idx, (kind, name, round_num) in enumerate(steps, start=1):
        if idx <= done_steps:
            continue

        if kind == "warmup":
            logger.info("=== Warm-up: solo training on %s (simplest, %d episodes) ===",
                        name, args.warmup_episodes)
            agent = ensure_agent(name)
            agent.train(get_env(name, cfg_of[name]), episodes=args.warmup_episodes,
                        log_loss_every_steps=args.log_loss_every_steps)
            shared_state = agent.state_dict()
            res = eval_holdout(f"After warm-up on {name}", shared_state)

        elif kind == "focus":
            logger.info("=== Focus: fine-tuning shared weights on %s alone (%d episodes) ===",
                        name, args.focus_episodes)
            focus_agent = DQNAgent(**agent_kwargs)
            focus_agent.load_state_dict(shared_state)
            focus_agent.train(get_env(name, cfg_of[name]), episodes=args.focus_episodes,
                              log_loss_every_steps=args.log_loss_every_steps)
            shared_state = focus_agent.state_dict()
            res = eval_holdout(f"After focus on {name}", shared_state)
            active_agents[name] = focus_agent

        else:  # fedavg
            active_names = list(active_agents.keys())
            if round_num == 1:
                logger.info("=== FedAvg: %d rounds across %s ===",
                            args.fedavg_rounds, active_names)
            state_dicts, base_weights, action_counts = [], [], []
            for city_name, city_agent in active_agents.items():
                city_agent.start_round(shared_state)
                sd, n_samples, _, ac = city_agent.train(
                    get_env(city_name, cfg_of[city_name]), episodes=args.local_episodes,
                    log_loss_every_steps=args.log_loss_every_steps,
                )
                state_dicts.append(sd)
                base_weights.append(n_samples)
                action_counts.append(ac)
            shared_state = aggregate_round(
                state_dicts=state_dicts, base_weights=base_weights, action_counts=action_counts,
                use_masked_head=True, head_weight_key=head_weight_key, head_bias_key=head_bias_key,
                previous_global_state=shared_state,
            )
            res = eval_holdout(
                f"FedAvg round {round_num}/{args.fedavg_rounds} with {active_names}",
                shared_state)

        record(f"{kind}:{name}:{round_num}", res)
        save_state(idx)

    for env in envs.values():
        env.close()

    logger.info("=== Final forgetting check: re-evaluating every active city ===")
    for name, cfg in ordered_cities:
        eval_in_distribution(name, cfg, shared_state, "FINAL (progressive curriculum complete)")

    evaluator.close()


if __name__ == "__main__":
    main()
