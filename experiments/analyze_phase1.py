"""
Phase 1 (and Phase 3) results analysis.

Reads every federated_history.json under results_root and produces:
  1. A summary table: strategy | metric mean±std across seeds | n_seeds
  2. A per-round CSV with reward/waiting_time/arrived/stopped learning curves
  3. Baseline rows (fixed_time, max_pressure) shown as single reference values
  4. Cluster-aware rows for clustered_fedavg (per-cluster metrics)

Run-type auto-detection via the `eval_mode` field in history:
  "federated"          → standard FedAvg / EMA / gradient-survival run
  "no_federation"      → independent per-city DQN (--no_federation)
  "clustered_fedavg"   → clustered FedAvg (--aggregation_strategy clustered_fedavg)
  "baseline_*"         → rule-based controller (--baseline_controller fixed_time/max_pressure)

Folder naming convention (produced by run_phase1_ablation.sh):
  results/phase1/<strategy>_<condition>_seed<n>/federated_history.json

Where <condition> is one of: with_fix, without_fix, no_federation,
clustered_fedavg_k<N>, fixed_time, max_pressure.

Usage:
    python experiments/analyze_phase1.py --results_root results/phase1
    python experiments/analyze_phase1.py --results_root results/phase0
    python experiments/analyze_phase1.py --results_root results/phase1 --csv curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import glob
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ── Metrics emitted by HoldoutEvaluator ───────────────────────────────────
METRICS = ["mean_reward", "mean_waiting_time", "mean_stopped", "mean_arrived"]
METRIC_LABELS = {
    "mean_reward":       "reward",
    "mean_waiting_time": "waiting_time (s)",
    "mean_stopped":      "stopped",
    "mean_arrived":      "arrived",
}
# Keys in federated_history.json for each metric
HISTORY_KEYS: Dict[str, str] = {
    "mean_reward":       "eval_reward",
    "mean_waiting_time": "eval_waiting_time",
    "mean_stopped":      "eval_stopped",
    "mean_arrived":      "eval_arrived",
}


# ---------------------------------------------------------------------------
# History loading + extraction
# ---------------------------------------------------------------------------

def detect_run_type(history: dict) -> str:
    """Return a canonical run-type string from a loaded history dict."""
    # Explicit marker from baseline-only runs
    if history.get("baseline_only"):
        ctrl = history.get("baseline_controller", "unknown")
        return f"baseline_{ctrl}"

    # Explicit mode list (training runs)
    modes = history.get("eval_mode", [])
    if modes:
        return modes[-1]  # last round's mode is authoritative

    return "federated"


def extract_last_metrics(history: dict) -> Dict[str, Optional[float]]:
    """Pull last-round holdout metrics from our flat-dict-of-lists schema."""
    out: Dict[str, Optional[float]] = {}
    for metric, key in HISTORY_KEYS.items():
        vals = history.get(key, [])
        # filter None (rounds where evaluator was absent)
        valid = [v for v in vals if v is not None]
        out[metric] = float(valid[-1]) if valid else None
    return out


def extract_all_rounds(history: dict) -> List[Dict[str, Any]]:
    """Return a list of per-round dicts for learning-curve CSVs."""
    rounds = history.get("round", [])
    rows = []
    for i, r in enumerate(rounds):
        row: Dict[str, Any] = {"round": r}
        for metric, key in HISTORY_KEYS.items():
            vals = history.get(key, [])
            row[metric] = vals[i] if i < len(vals) else None
        row["eval_mode"] = (history.get("eval_mode") or [None])[i] if i < len(history.get("eval_mode", [])) else None
        rows.append(row)
    return rows


def load_run(run_dir: str) -> Optional[Dict]:
    path = os.path.join(run_dir, "federated_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> str:
    if not values:
        return "(missing)"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.pstdev(values):.1f}"


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", default="results/phase1",
                    help="Root directory containing <run_name>/federated_history.json files.")
    ap.add_argument("--csv", default=None,
                    help="If set, also write per-round learning curves to this CSV file.")
    ap.add_argument("--last_n_rounds", type=int, default=1,
                    help="Average over the last N rounds instead of only the final round (default 1).")
    args = ap.parse_args()

    # ── Collect all runs ────────────────────────────────────────────────────
    # Group by (run_type, condition, seed) where condition = anything after
    # the first underscore (e.g. "with_fix", "clustered_fedavg_k2", …).
    # Strategy key = everything except "_seed<N>" suffix.
    by_strategy: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_run_rows: List[Dict[str, Any]] = []  # for CSV

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.results_root, "*"))
        if os.path.isdir(d)
    )
    if not run_dirs:
        print(f"No subdirectories found under '{args.results_root}'.")
        return

    seen = 0
    skipped = 0
    for run_dir in run_dirs:
        history = load_run(run_dir)
        if history is None:
            skipped += 1
            continue
        seen += 1

        run_type = detect_run_type(history)
        run_name = os.path.basename(run_dir)

        # Build a display key: strip trailing "_seed<N>"
        import re
        strategy_key = re.sub(r"_seed\d+$", "", run_name)

        # Extract headline metrics (average of last_n_rounds)
        metrics_by_round = extract_all_rounds(history)
        if not metrics_by_round:
            skipped += 1
            continue

        tail = metrics_by_round[-args.last_n_rounds:]
        for metric in METRICS:
            tail_vals = [r[metric] for r in tail if r[metric] is not None]
            if tail_vals:
                by_strategy[strategy_key][metric].append(statistics.mean(tail_vals))

        # Learning-curve rows for CSV
        for row in metrics_by_round:
            all_run_rows.append({
                "run_name": run_name,
                "strategy_key": strategy_key,
                "run_type": run_type,
                **row,
            })

    if skipped:
        print(f"[warn] {skipped} director(y/ies) skipped (no federated_history.json).")
    print(f"Loaded {seen} run(s) from '{args.results_root}'.\n")

    # ── Print summary table ─────────────────────────────────────────────────
    # Sort: baseline rows last, rest alphabetically
    def sort_key(s: str) -> tuple:
        is_baseline = 1 if s.startswith(("baseline_", "fixed_time", "max_pressure")) else 0
        return (is_baseline, s)

    strategies = sorted(by_strategy.keys(), key=sort_key)

    col_widths = [32] + [22] * len(METRICS) + [8]
    header_parts = ["strategy"] + [METRIC_LABELS[m] for m in METRICS] + ["n_seeds"]
    header = "".join(f"{h:<{w}}" for h, w in zip(header_parts, col_widths))
    print(f"=== Phase 1 results: last {args.last_n_rounds} round(s), all run types ===\n")
    print(header)
    print("-" * len(header))

    for strat in strategies:
        vals_by_metric = by_strategy[strat]
        n_seeds = max((len(v) for v in vals_by_metric.values()), default=0)
        row_parts = [strat] + [_mean_std(vals_by_metric.get(m, [])) for m in METRICS] + [str(n_seeds)]
        print("".join(f"{p:<{w}}" for p, w in zip(row_parts, col_widths)))

    # ── Significance tests ──────────────────────────────────────────────────
    if HAVE_SCIPY:
        ref_key = next((k for k in strategies if "fedavg_with_fix" in k), None)
        if ref_key is None:
            ref_key = next((k for k in strategies if "fedavg" in k and "without" not in k
                            and not k.startswith("baseline")), None)
        if ref_key:
            print(f"\n--- Wilcoxon signed-rank vs reference '{ref_key}' ---")
            ref_vals = by_strategy[ref_key]
            for strat in strategies:
                if strat == ref_key:
                    continue
                for metric in ["mean_reward", "mean_waiting_time"]:
                    a = ref_vals.get(metric, [])
                    b = by_strategy[strat].get(metric, [])
                    n = min(len(a), len(b))
                    if n < 3:
                        continue
                    try:
                        stat, p = scipy_stats.wilcoxon(a[:n], b[:n])
                        label = METRIC_LABELS[metric]
                        print(f"  {strat:<30} {label:<20} p={p:.4f}  (n={n})")
                    except ValueError:
                        pass
    else:
        print("\n(pip install scipy to get paired significance tests)")

    # ── Per-round cluster breakdown ─────────────────────────────────────────
    clustered_dirs = [d for d in run_dirs if "clustered" in os.path.basename(d).lower()]
    if clustered_dirs:
        print("\n--- Clustered FedAvg: per-cluster last-round metrics ---")
        for run_dir in clustered_dirs[:3]:  # cap at 3 runs for brevity
            history = load_run(run_dir)
            if not history:
                continue
            per_model_rounds = history.get("eval_per_model", [])
            cluster_assignments = history.get("cluster_assignments", [])
            if per_model_rounds:
                last_pm = per_model_rounds[-1] or {}
                last_ca = cluster_assignments[-1] if cluster_assignments else {}
                print(f"  {os.path.basename(run_dir)}")
                print(f"    cluster assignments: {last_ca}")
                for cluster_id, metrics in last_pm.items():
                    if metrics:
                        print(
                            f"    {cluster_id}: "
                            f"reward={_fmt(metrics.get('mean_reward'))}  "
                            f"waiting={_fmt(metrics.get('mean_waiting_time'))}s  "
                            f"arrived={_fmt(metrics.get('mean_arrived'))}"
                        )

    # ── No-federation per-city breakdown ───────────────────────────────────
    nofed_dirs = [d for d in run_dirs if "no_federation" in os.path.basename(d).lower()]
    if nofed_dirs:
        print("\n--- No-federation: per-city last-round metrics ---")
        for run_dir in nofed_dirs[:3]:
            history = load_run(run_dir)
            if not history:
                continue
            per_model_rounds = history.get("eval_per_model", [])
            if per_model_rounds:
                last_pm = per_model_rounds[-1] or {}
                print(f"  {os.path.basename(run_dir)}")
                for city, metrics in last_pm.items():
                    if metrics:
                        print(
                            f"    {city}: "
                            f"reward={_fmt(metrics.get('mean_reward'))}  "
                            f"waiting={_fmt(metrics.get('mean_waiting_time'))}s  "
                            f"arrived={_fmt(metrics.get('mean_arrived'))}"
                        )

    # ── Write CSV if requested ──────────────────────────────────────────────
    if args.csv and all_run_rows:
        fieldnames = ["run_name", "strategy_key", "run_type", "round"] + METRICS + ["eval_mode"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_run_rows)
        print(f"\nLearning curves written to '{args.csv}'.")

    # ── Decision gate ────────────────────────────────────────────────────────
    print(
        "\nDecision gate for Phase 1 → Phase 3 scale-up:\n"
        "  - fedavg_with_fix should beat fedavg_without_fix on arrived/waiting_time "
        "across all 5 seeds with non-overlapping ranges.\n"
        "  - At least one adaptive strategy (ema_alignment, clustered_fedavg) should "
        "beat fedavg_with_fix on ≥2 metrics before scaling to Phase 3.\n"
        "  - Trained policies should beat max_pressure on reward; "
        "max_pressure is a strong oracle for waiting_time only."
    )


if __name__ == "__main__":
    main()
