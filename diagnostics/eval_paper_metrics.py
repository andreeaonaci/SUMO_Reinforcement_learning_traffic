"""Evaluate a checkpoint in the metrics the traffic-signal-control literature
reports, so results can sit in the same table as published numbers.

Why this is separate from federated/evaluator.py: that evaluator is used by
live training runs, and changing it mid-experiment is exactly the hazard sec 88
documented (an evaluator change leaking into training). This script only reads
checkpoints, so it can be run against finished experiments with zero risk to
anything in flight.

What this adds over the training-time evaluator:

  queue          The training evaluator asks SUMO's info dict for
                 "system_mean_queue_length", which that dict does NOT contain,
                 so it silently returned the 0.0 default in every run to date
                 (every eval line in the logs reads "queue=0.00"). Here it is
                 computed from halting vehicles per signalised intersection.
  trip time      Mean seconds from departure to arrival, tracked per vehicle
                 via traci's departed/arrived ID lists. Not available from the
                 info dict at all.
  delay          Mean per-vehicle timeLoss (SUMO's own definition: time lost
                 relative to travelling at the allowed speed). Polled while
                 each vehicle is still in the network, since a vehicle cannot
                 be queried after it arrives.

Reward is deliberately NOT reported: it is this project's internal
diff-waiting-time sum in units of 100 vehicle-seconds and is not comparable to
any published quantity.

Usage:
    python diagnostics/eval_paper_metrics.py CKPT [CKPT ...] \
        --base_dir environments_c1_4_6 --pad_to_true_holdout --episodes 5
"""
import argparse
import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.dqn import DQNAgent
from diagnostics.finetune_on_holdout import infer_arch_from_checkpoint
from experiments.federated_training import (
    resolve_city_configs_and_dims,
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
)


def detect_phase_relational(state: dict) -> bool:
    return any(k.startswith("phase_scorer.") for k in state)


def build_agent(state: dict, k_max: int, env_action_dim: int):
    """Construct the agent a checkpoint came from.

    The shared ``infer_arch_from_checkpoint`` reads action_dim off
    ``head.4.weight``, which a phase-relational checkpoint does not have -- its
    scorer emits ONE scalar per phase, so the head's shape carries no action
    count at all. That is the point of the architecture (parameters are
    independent of the action-space width), but it means action_dim has to come
    from the environment instead of the weights. phase_dim is still recoverable,
    from the scorer's input width minus d_model.
    """
    if detect_phase_relational(state):
        own_dim = state["own_encoder.0.weight"].shape[1]
        neighbor_dim = state["neighbor_encoder.0.weight"].shape[1]
        d_model = state["head.0.weight"].shape[0]
        phase_dim = state["phase_scorer.0.weight"].shape[1] - d_model
        agent = DQNAgent(
            own_dim=own_dim, neighbor_dim=neighbor_dim, k_max=k_max,
            action_dim=env_action_dim, d_model=d_model,
            phase_relational=True, phase_dim=phase_dim,
        )
        arch = {"own_dim": own_dim, "neighbor_dim": neighbor_dim,
                "action_dim": env_action_dim, "phase_dim": phase_dim}
    else:
        arch = infer_arch_from_checkpoint(state)
        agent = DQNAgent(
            own_dim=arch["own_dim"], neighbor_dim=arch["neighbor_dim"], k_max=k_max,
            action_dim=arch["action_dim"], dueling=arch["dueling"], head_fix=arch["head_fix"],
            encoder_depth=arch["encoder_depth"], n_attn_layers=arch["n_attn_layers"],
        )
    agent.load_state_dict(state)
    agent.q.eval()
    return agent, arch


def run_episode(env, agent, traci):
    """One greedy episode; returns literature-style metrics."""
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    n_signals = max(len(obs), 1)

    depart_t, timeloss, durations, delays = {}, {}, [], []
    seen = set()
    queue_samples, wait_samples = [], []
    t = 0.0
    done = False
    while not done:
        actions = agent.act_batch(obs, explore=False)
        obs, _, dones, info = env.step(actions)
        t = float(traci.simulation.getTime())

        # NOT getDepartedIDList()/getArrivedIDList(): those cover only the most
        # recent simulationStep, while one env.step() advances delta_time (5) of
        # them -- so ~80% of departures and arrivals are never seen, and the
        # trip count comes out ~20x too low. Set-differencing the live vehicle
        # list is granularity-independent; trip times are quantised to
        # delta_time seconds, which is immaterial against ~150s trips.
        present = set(traci.vehicle.getIDList())
        for vid in present - seen:
            depart_t[vid] = t
        for vid in present:
            try:
                timeloss[vid] = float(traci.vehicle.getTimeLoss(vid))
            except Exception:
                pass
        for vid in seen - present:
            if vid in depart_t:
                durations.append(t - depart_t.pop(vid))
            if vid in timeloss:
                delays.append(timeloss.pop(vid))
        seen = present

        last = info or {}
        stopped = last.get("system_total_stopped")
        if stopped is not None:
            queue_samples.append(float(stopped) / n_signals)
        w = last.get("system_mean_waiting_time")
        if w is not None:
            wait_samples.append(float(w))

        done = dones.get("__all__", all(dones.values()) if dones else True)

    return {
        "trip_time": statistics.fmean(durations) if durations else float("nan"),
        "delay": statistics.fmean(delays) if delays else float("nan"),
        "queue": statistics.fmean(queue_samples) if queue_samples else float("nan"),
        "wait": statistics.fmean(wait_samples) if wait_samples else float("nan"),
        "arrived": len(durations),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--base_dir", default="environments_c1_4_6")
    ap.add_argument("--pad_to_true_holdout", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--eval_sumo_seed", type=int, default=12345)
    args = ap.parse_args()

    import traci

    _, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    print(f"{'checkpoint':<44} {'head':<9} {'trip_s':>8} {'delay_s':>8} {'queue':>7} {'wait_s':>8} {'arrived':>8}")
    agg = {}
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
        agent, _ = build_agent(state, k_max, action_dim)
        head = "phase" if detect_phase_relational(state) else "indexed"

        evaluator = make_holdout_evaluator(
            args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
            episodes=1, eval_sumo_seed=args.eval_sumo_seed,
        )
        if evaluator is None:
            raise RuntimeError("Could not construct holdout evaluator.")
        eps = []
        for _ in range(args.episodes):
            eps.append(run_episode(evaluator._get_env(), agent, traci))
        evaluator.close()

        m = {k: statistics.fmean([e[k] for e in eps if not np.isnan(e[k])] or [float("nan")])
             for k in ("trip_time", "delay", "queue", "wait")}
        m["arrived"] = statistics.fmean([e["arrived"] for e in eps])
        agg.setdefault(head, []).append(m)
        print(f"{os.path.basename(os.path.dirname(path))[:43]:<44} {head:<9} "
              f"{m['trip_time']:>8.1f} {m['delay']:>8.1f} {m['queue']:>7.2f} "
              f"{m['wait']:>8.2f} {m['arrived']:>8.0f}")

    print()
    for head, rows in agg.items():
        n = len(rows)
        print(f"{head} head, mean over {n} checkpoint(s): "
              f"trip {statistics.fmean(r['trip_time'] for r in rows):.1f}s | "
              f"delay {statistics.fmean(r['delay'] for r in rows):.1f}s | "
              f"queue {statistics.fmean(r['queue'] for r in rows):.2f} | "
              f"wait {statistics.fmean(r['wait'] for r in rows):.2f}s | "
              f"arrived {statistics.fmean(r['arrived'] for r in rows):.0f}")


if __name__ == "__main__":
    main()
