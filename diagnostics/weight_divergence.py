"""Per-round weight-divergence / gradient-conflict diagnostic for a federated run.

For each round r (2..N), compares the two (or more) cities' local weight
deltas from that round's pre-round global checkpoint:

    delta_city = client_round_r - global_round_(r-1)

and reports, per round:
  - ||delta_city|| for each city (how far each client moved)
  - cosine similarity between cities' deltas (gradient "conflict" --
    near 0 or negative means the clients are pulling the shared model in
    different/opposing directions that round)
  - ||agg_delta|| = global_round_r - global_round_(r-1) (how far the
    aggregated model actually moved)
  - the eval reward at r-1 and r (from federated_history.json) so
    divergence/conflict can be correlated against reward regressions

Usage:
    python diagnostics/weight_divergence.py results/run_.../ [--cities city_1 city_4]

Only reads existing checkpoints already on disk -- does not run any training.
"""
import argparse
import json
import os

import torch


def flatten(state_dict):
    return torch.cat([v.reshape(-1).float() for v in state_dict.values()])


def load_state(path):
    return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cities", nargs="+", default=None,
                     help="City names to compare (default: autodetect from clients/ dir)")
    args = ap.parse_args()

    clients_dir = os.path.join(args.run_dir, "clients")
    if args.cities is None:
        names = sorted({
            fn.rsplit("_round_", 1)[0]
            for fn in os.listdir(clients_dir)
            if fn.endswith(".pth")
        })
    else:
        names = args.cities

    hist_path = os.path.join(args.run_dir, "federated_history.json")
    with open(hist_path) as f:
        hist = json.load(f)
    reward_by_round = dict(zip(hist["round"], hist["eval_reward"]))

    rounds = sorted({
        int(fn.rsplit("_round_", 1)[1].split(".")[0])
        for fn in os.listdir(clients_dir)
        if fn.endswith(".pth")
    })

    print(f"run_dir={args.run_dir}  cities={names}  rounds={rounds[0]}..{rounds[-1]}")
    print(f"{'round':>5} {'||d_'+names[0]+'||':>12} {'||d_'+names[1]+'||':>12} "
          f"{'cos_sim':>8} {'||agg_d||':>10} {'reward[r-1]':>12} {'reward[r]':>10} {'d_reward':>10}")

    for r in rounds:
        prev_global_path = os.path.join(args.run_dir, f"global_round_{r-1:03d}.pth")
        cur_global_path = os.path.join(args.run_dir, f"global_round_{r:03d}.pth")
        if not os.path.exists(prev_global_path) or not os.path.exists(cur_global_path):
            continue

        prev_global = flatten(load_state(prev_global_path))
        cur_global = flatten(load_state(cur_global_path))
        agg_delta = cur_global - prev_global
        agg_norm = agg_delta.norm().item()

        deltas = {}
        for name in names:
            ckpt = os.path.join(clients_dir, f"{name}_round_{r:03d}.pth")
            if not os.path.exists(ckpt):
                deltas[name] = None
                continue
            client_state = flatten(load_state(ckpt))
            deltas[name] = client_state - prev_global

        if any(d is None for d in deltas.values()) or len(names) != 2:
            cos_sim = float("nan")
            n0 = deltas[names[0]].norm().item() if deltas.get(names[0]) is not None else float("nan")
            n1 = deltas[names[1]].norm().item() if deltas.get(names[1]) is not None else float("nan")
        else:
            d0, d1 = deltas[names[0]], deltas[names[1]]
            n0, n1 = d0.norm().item(), d1.norm().item()
            cos_sim = torch.dot(d0, d1).item() / (n0 * n1 + 1e-12)

        rew_prev = reward_by_round.get(r - 1, float("nan"))
        rew_cur = reward_by_round.get(r, float("nan"))
        d_reward = rew_cur - rew_prev if rew_prev == rew_prev and rew_cur == rew_cur else float("nan")

        print(f"{r:5d} {n0:12.4f} {n1:12.4f} {cos_sim:8.4f} {agg_norm:10.4f} "
              f"{rew_prev:12.2f} {rew_cur:10.2f} {d_reward:10.2f}")


if __name__ == "__main__":
    main()
