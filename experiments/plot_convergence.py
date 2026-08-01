#!/usr/bin/env python3
"""Plot convergence curves from one or more federated_history.json files."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt


def load_xy(history_path: str, metric: str):
    with open(history_path) as f:
        hist = json.load(f)

    rounds = hist.get("round", [])
    values = hist.get(metric, [])
    if not rounds or not values:
        return [], []

    y = [float(v) if v is not None else float("nan") for v in values]
    return rounds, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", nargs="+", required=True, help="Paths to federated_history.json files")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels matching --histories")
    parser.add_argument("--metric", default="eval_waiting_time", choices=["eval_reward", "eval_waiting_time", "eval_stopped"])
    parser.add_argument("--output", default=None, help="Output image path, e.g. results/phase0/convergence.png")
    args = parser.parse_args()

    labels: List[str]
    if args.labels is None:
        labels = [os.path.basename(os.path.dirname(p)) for p in args.histories]
    else:
        if len(args.labels) != len(args.histories):
            raise ValueError("--labels length must match --histories length")
        labels = args.labels

    plt.figure(figsize=(9, 5))
    for path, label in zip(args.histories, labels):
        x, y = load_xy(path, args.metric)
        if not x:
            continue
        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Federated Round")
    plt.ylabel(args.metric)
    plt.title(f"Convergence: {args.metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
