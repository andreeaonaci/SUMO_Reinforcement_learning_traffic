"""
Aggregate Phase 0 ablation runs (with_fix vs without_fix, 3 seeds) and report
mean +/- std per metric, plus a paired significance test.

ASSUMPTIONS (adjust to match your real federated_history.json / evaluator
output schema):
- Each run dir (results/phase0/<run_name>/) contains federated_history.json
  which is a list of per-round dicts, OR a dict with a "rounds" key holding
  that list.
- Each per-round dict has holdout eval metrics under keys like
  "arrived", "waiting_time", "stopped", "reward" (top-level or nested under
  "eval"). Edit `extract_metrics()` below if your schema nests these
  differently -- this is the one function most likely to need a tweak.
- We report metrics from the LAST round (round 10) as the headline number,
  matching the table already in your project summary.

Usage:
    python analyze_phase0.py --results_root results/phase0
"""

import argparse
import json
import glob
import os
import statistics
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

METRICS = ["arrived", "waiting_time", "stopped", "reward"]


def extract_metrics(history):
    """Pull the last round's holdout metrics out of a loaded history object.
    EDIT THIS if your federated_history.json nests things differently."""
    rounds = history["rounds"] if isinstance(history, dict) and "rounds" in history else history
    last = rounds[-1]

    # try a few plausible nesting patterns
    candidates = [last, last.get("eval", {}), last.get("holdout_eval", {})]
    metrics = {}
    for m in METRICS:
        for c in candidates:
            if isinstance(c, dict) and m in c:
                metrics[m] = c[m]
                break
        if m not in metrics:
            metrics[m] = None
    return metrics


def load_run(run_dir):
    path = os.path.join(run_dir, "federated_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        history = json.load(f)
    return extract_metrics(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/phase0")
    ap.add_argument("--strategy", default="fedavg")
    args = ap.parse_args()

    by_condition = defaultdict(lambda: defaultdict(list))  # condition -> metric -> [values across seeds]

    run_dirs = sorted(glob.glob(os.path.join(args.results_root, f"{args.strategy}_*_seed*")))
    if not run_dirs:
        print(f"No runs found under {args.results_root} matching {args.strategy}_*_seed*")
        return

    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        cond = "with_fix" if "with_fix" in name else "without_fix"
        metrics = load_run(run_dir)
        if metrics is None:
            print(f"[warn] no federated_history.json in {run_dir}, skipping")
            continue
        for m, v in metrics.items():
            if v is not None:
                by_condition[cond][m].append(v)

    print(f"\n=== Phase 0 ablation: {args.strategy}, head-fix on vs off ===\n")
    header = f"{'metric':<15}{'with_fix (mean±std)':<28}{'without_fix (mean±std)':<28}{'n_seeds':<10}"
    print(header)
    print("-" * len(header))

    for m in METRICS:
        wf = by_condition["with_fix"].get(m, [])
        wof = by_condition["without_fix"].get(m, [])
        if not wf or not wof:
            print(f"{m:<15}{'(missing)':<28}{'(missing)':<28}")
            continue
        wf_str = f"{statistics.mean(wf):.1f} ± {statistics.pstdev(wf):.1f}"
        wof_str = f"{statistics.mean(wof):.1f} ± {statistics.pstdev(wof):.1f}"
        n = min(len(wf), len(wof))
        print(f"{m:<15}{wf_str:<28}{wof_str:<28}{n:<10}")

    if HAVE_SCIPY:
        print("\n--- Paired significance (Wilcoxon signed-rank, n=seeds) ---")
        print("Note: n=3 is thin for a real test; treat this as a directional")
        print("signal for Phase 0, not a claim for the paper. Re-run with 5")
        print("seeds on the full roster (Phase 1) before reporting a p-value.\n")
        for m in METRICS:
            wf = by_condition["with_fix"].get(m, [])
            wof = by_condition["without_fix"].get(m, [])
            if len(wf) == len(wof) and len(wf) >= 3:
                try:
                    stat, p = scipy_stats.wilcoxon(wf, wof)
                    print(f"{m:<15} statistic={stat:.3f}  p={p:.4f}")
                except ValueError as e:
                    print(f"{m:<15} (test not applicable: {e})")
    else:
        print("\n(scipy not installed -- pip install scipy --break-system-packages "
              "to get the paired significance test)")

    print("\nDecision gate: if with_fix clearly beats without_fix on arrived/"
          "waiting_time/reward across all 3 seeds with non-overlapping "
          "ranges, proceed to Phase 1 (full roster, more seeds). If it's "
          "ambiguous, that's the critical finding to know now -- don't scale "
          "up yet.")


if __name__ == "__main__":
    main()