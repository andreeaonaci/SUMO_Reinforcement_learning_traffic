"""Per-round cross-city weight-divergence measurement.

Directly measures the "client drift in weight space" mechanism the project
has repeatedly inferred but never measured directly (flagged, not yet done,
in fidings/divergence_investigation.md sec 28's "reallocates priority" note).
Sections 9/11 measured Q-head row-delta MAGNITUDE (how fast masked-head
aggregation lets specialized rows move); this script measures whether
different cities' updates each round point in the SAME direction (healthy --
just noisy/high-variance) or CONFLICTING directions (client drift proper --
the mechanism FedProx was supposed to fix but didn't, sec 14).

For each round r >= 2, uses artifacts every real training run already saves,
no new training needed:

    global_round_{r-1}.pth        -- the global model each city trained FROM
    clients/city_X_round_r.pth    -- each city's LOCAL model after training

delta_X_r = client_X_round_r - global_round_{r-1}  (this city's own
"pseudo-gradient" that round, same framing as
federated/aggregation.py::_state_delta).

Reports, per round:
  - per-city delta L2 norm, split into "trunk" (own_encoder/neighbor_encoder/
    attn/value_head -- shared cleanly across every city regardless of
    action_dim) vs "head" (advantage_head or head.4 -- action-indexed, only
    moves on rows a city actually touched)
  - mean pairwise cosine similarity of TRUNK deltas across every city pair
    that round -- the key number. Close to +1: cities agree on direction,
    aggregation should be stable. Close to 0: uncorrelated/noisy but not
    fighting. Negative: cities are actively pulling the shared trunk in
    opposite directions -- direct evidence of the client-drift-conflict
    hypothesis, not just high variance.

Head-row cosine similarity is NOT computed the same way: different cities
touch different, mostly-disjoint action rows (that's the whole reason
masked-head aggregation exists), so a naive full-head cosine similarity
would mostly measure "these two cities didn't touch the same rows" rather
than genuine conflict. Head divergence is reported as per-row magnitude only,
consistent with how sec 11 already analyzed it.

Usage:
    python analyse/weight_divergence.py --run_dir results/run_2026_08_15-...
    python analyse/weight_divergence.py --run_dir results/run_... --dueling --csv out.csv
"""
import argparse
import csv
import glob
import itertools
import os
import re
from typing import Dict, List, Tuple

import torch


def head_key_names(dueling: bool) -> Tuple[str, str]:
    """Mirrors federated/aggregation.py::head_key_names (duplicated, not
    imported, so this script has no repo-internal import dependency and can
    run as a plain script from any cwd -- keep in sync if the real network's
    head naming ever changes)."""
    if dueling:
        return "advantage_head.weight", "advantage_head.bias"
    return "head.4.weight", "head.4.bias"


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def _discover_rounds(run_dir: str) -> List[int]:
    global_ckpts = glob.glob(os.path.join(run_dir, "global_round_*.pth"))
    rounds = []
    for p in global_ckpts:
        m = re.search(r"global_round_(\d+)\.pth$", os.path.basename(p))
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)


def _discover_client_states(run_dir: str, round_num: int) -> Dict[str, str]:
    """{city_name: path} for every clients/<city>_round_{round_num:03d}.pth."""
    pattern = os.path.join(run_dir, "clients", f"*_round_{round_num:03d}.pth")
    out = {}
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        m = re.match(r"(.+)_round_\d+\.pth$", base)
        if m:
            out[m.group(1)] = p
    return out


def _split_trunk_and_head_keys(
    state_keys, head_weight_key: str, head_bias_key: str
) -> Tuple[List[str], List[str]]:
    head_keys = {head_weight_key, head_bias_key}
    trunk_keys = [k for k in state_keys if k not in head_keys]
    return trunk_keys, [k for k in state_keys if k in head_keys]


def _delta(client_state, global_state, keys) -> Dict[str, torch.Tensor]:
    return {k: (client_state[k].float() - global_state[k].float()) for k in keys}


def _flatten(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in delta.values()])


def _l2_norm(delta: Dict[str, torch.Tensor]) -> float:
    return float(_flatten(delta).norm().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom.item() <= 1e-12:
        return float("nan")
    return float((a @ b / denom).item())


def analyze_run(run_dir: str, dueling: bool) -> List[dict]:
    head_weight_key, head_bias_key = head_key_names(dueling)
    rounds = _discover_rounds(run_dir)
    if len(rounds) < 1:
        raise ValueError(f"No global_round_*.pth checkpoints found under {run_dir}")

    rows = []
    for r in rounds:
        if r < 2:
            continue  # no global_round_{r-1} reference for round 1
        global_before_path = os.path.join(run_dir, f"global_round_{r - 1:03d}.pth")
        if not os.path.exists(global_before_path):
            continue
        client_paths = _discover_client_states(run_dir, r)
        if len(client_paths) < 1:
            continue

        global_before = _load_state(global_before_path)
        client_states = {name: _load_state(p) for name, p in client_paths.items()}

        sample_keys = next(iter(client_states.values())).keys()
        trunk_keys, head_keys_present = _split_trunk_and_head_keys(
            sample_keys, head_weight_key, head_bias_key
        )

        trunk_deltas = {}
        head_norms = {}
        for name, state in client_states.items():
            trunk_delta = _delta(state, global_before, trunk_keys)
            trunk_deltas[name] = _flatten(trunk_delta)
            if head_keys_present:
                head_delta = _delta(state, global_before, head_keys_present)
                head_norms[name] = _l2_norm(head_delta)

        pair_cosines = []
        for a, b in itertools.combinations(trunk_deltas.keys(), 2):
            c = _cosine(trunk_deltas[a], trunk_deltas[b])
            if c == c:  # not NaN
                pair_cosines.append(c)

        mean_cosine = sum(pair_cosines) / len(pair_cosines) if pair_cosines else float("nan")
        min_cosine = min(pair_cosines) if pair_cosines else float("nan")

        rows.append({
            "round": r,
            "n_cities": len(client_states),
            "trunk_delta_norm_mean": sum(float(v.norm().item()) for v in trunk_deltas.values()) / len(trunk_deltas),
            "trunk_delta_norm_per_city": {k: float(v.norm().item()) for k, v in trunk_deltas.items()},
            "trunk_cosine_mean": mean_cosine,
            "trunk_cosine_min": min_cosine,
            "trunk_cosine_n_pairs": len(pair_cosines),
            "head_delta_norm_per_city": head_norms,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="results/run_<timestamp>[_<pid>] directory to analyze.")
    parser.add_argument("--dueling", action="store_true", help="Must match the --dueling flag the run was trained with.")
    parser.add_argument("--csv", default=None, help="Optional path to write a flat per-round CSV summary.")
    args = parser.parse_args()

    rows = analyze_run(args.run_dir, args.dueling)

    if not rows:
        print(f"No analyzable rounds found under {args.run_dir} (need >=2 global checkpoints "
              f"and matching clients/*_round_NNN.pth files).")
        return

    print(f"\n{'round':>5} {'n_cities':>8} {'trunk|delta| mean':>18} "
          f"{'trunk cos mean':>15} {'trunk cos min':>14}")
    for row in rows:
        print(f"{row['round']:>5} {row['n_cities']:>8} "
              f"{row['trunk_delta_norm_mean']:>18.4f} "
              f"{row['trunk_cosine_mean']:>15.4f} "
              f"{row['trunk_cosine_min']:>14.4f}")

    overall_mean_cosine = sum(
        r["trunk_cosine_mean"] for r in rows if r["trunk_cosine_mean"] == r["trunk_cosine_mean"]
    ) / max(1, sum(1 for r in rows if r["trunk_cosine_mean"] == r["trunk_cosine_mean"]))
    print(f"\nOverall mean trunk-delta cosine similarity across all rounds/pairs: {overall_mean_cosine:.4f}")
    print("Reading guide: near +1 = cities agree on update direction (aggregation should be "
          "stable); near 0 = noisy/uncorrelated; negative = cities are pulling the shared "
          "trunk in conflicting directions (direct evidence for client drift, not just variance).")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "n_cities", "trunk_delta_norm_mean", "trunk_cosine_mean", "trunk_cosine_min", "trunk_cosine_n_pairs"])
            for row in rows:
                writer.writerow([
                    row["round"], row["n_cities"], row["trunk_delta_norm_mean"],
                    row["trunk_cosine_mean"], row["trunk_cosine_min"], row["trunk_cosine_n_pairs"],
                ])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
