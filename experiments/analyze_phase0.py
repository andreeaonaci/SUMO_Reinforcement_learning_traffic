#!/usr/bin/env python3
"""Summarize Phase-0 ablation runs (with_fix vs without_fix) across seeds.

Expected folder structure under results_root:
  <strategy>_<condition>_seed<seed>/federated_history.json

Example:
  results/phase0/fedavg_with_fix_seed1/federated_history.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


RUN_DIR_RE = re.compile(r"^(?P<strategy>[^_]+)_(?P<condition>with_fix|without_fix)_seed(?P<seed>\d+)$")


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _safe_mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _safe_ci95(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * _safe_std(values) / math.sqrt(len(values))


def _load_last_metric(history_path: str, key: str) -> float | None:
    with open(history_path) as f:
        hist = json.load(f)
    vals = hist.get(key, [])
    if not vals:
        return None
    last = vals[-1]
    if last is None:
        return None
    return float(last)


def collect_runs(results_root: str) -> Dict[Tuple[str, str], List[Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)

    for name in sorted(os.listdir(results_root)):
        run_path = os.path.join(results_root, name)
        if not os.path.isdir(run_path):
            continue

        m = RUN_DIR_RE.match(name)
        if not m:
            continue

        history_path = os.path.join(run_path, "federated_history.json")
        if not os.path.exists(history_path):
            continue

        reward = _load_last_metric(history_path, "eval_reward")
        wait = _load_last_metric(history_path, "eval_waiting_time")
        stopped = _load_last_metric(history_path, "eval_stopped")

        grouped[(m.group("strategy"), m.group("condition"))].append(
            {
                "seed": float(m.group("seed")),
                "reward": reward if reward is not None else float("nan"),
                "waiting": wait if wait is not None else float("nan"),
                "stopped": stopped if stopped is not None else float("nan"),
            }
        )

    return grouped


def print_summary(grouped: Dict[Tuple[str, str], List[Dict[str, float]]]) -> None:
    if not grouped:
        print("No matching runs found.")
        return

    print("strategy,condition,n,reward_mean,reward_std,reward_ci95,waiting_mean,waiting_std,waiting_ci95,stopped_mean,stopped_std,stopped_ci95")

    for (strategy, condition), rows in sorted(grouped.items()):
        rewards = [r["reward"] for r in rows if not math.isnan(r["reward"])]
        waits = [r["waiting"] for r in rows if not math.isnan(r["waiting"])]
        stopped = [r["stopped"] for r in rows if not math.isnan(r["stopped"])]

        print(
            f"{strategy},{condition},{len(rows)},"
            f"{_safe_mean(rewards):.6f},{_safe_std(rewards):.6f},{_safe_ci95(rewards):.6f},"
            f"{_safe_mean(waits):.6f},{_safe_std(waits):.6f},{_safe_ci95(waits):.6f},"
            f"{_safe_mean(stopped):.6f},{_safe_std(stopped):.6f},{_safe_ci95(stopped):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/phase0")
    args = parser.parse_args()

    grouped = collect_runs(args.results_root)
    print_summary(grouped)


if __name__ == "__main__":
    main()
