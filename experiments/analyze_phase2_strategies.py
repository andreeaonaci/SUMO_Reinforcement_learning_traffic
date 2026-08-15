"""Phase 2 aggregation-strategy comparison: 7-city, --dueling --n_step 3.

Discovers every completed run from the ad-hoc `run_concurrent_batch.sh`
strategy-comparison batch (2026-08-13/15) by parsing the batch log files for
`finished <tag> ... exit=0 run_dir=<path>` lines -- these runs were never
given the `results/phase1/<strategy>_<condition>_seed<n>/` folder-naming
convention `analyze_phase1.py` expects, so they need their own discovery
logic rather than a glob.

For each strategy, computes (mean of per-seed mean eval_reward, std across
seeds) and (mean of per-seed best-round eval_reward), then compares against
the known 7-city `fedavg` reference from
`fidings/divergence_investigation.md` §23 (5 seeds: mean -6918.4 std 889.0,
best-round mean -2182.0) and the rule-based baselines from §24
(fixed_time -2.73, max_pressure -0.34 -- single-episode, not multi-seed).

Writes a dated, numbered section to fidings/divergence_investigation.md
(matching this repo's existing convention) and prints a short human summary.

Usage:
    python experiments/analyze_phase2_strategies.py
"""
import glob
import json
import math
import os
import re
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_LOGS = [
    "results/phase2_7city_strategy_comparison.log",
    "results/phase2_7city_strategy_comparison_resume.log",
    "results/phase2_7city_strategy_comparison_resume2.log",
    "results/phase2_7city_strategy_comparison_resume3.log",
    "results/phase2_7city_strategy_comparison_resume4.log",
]
FINDINGS_PATH = os.path.join(REPO_ROOT, "fidings", "divergence_investigation.md")
CLAUDE_MD_PATH = os.path.join(REPO_ROOT, "CLAUDE.md")

# Known reference points, cited from the existing writeup so this script's
# output stays traceable back to where they came from.
FEDAVG_REF = {"mean": -6918.4, "std": 889.0, "n": 5, "best_round_mean": -2182.0}
BASELINES = {"fixed_time": -2.73, "max_pressure": -0.34}

FINISHED_RE = re.compile(
    r"finished (?P<tag>\S+) base_dir=\S+ exit=(?P<exit>\d+) run_dir=(?P<run_dir>\S+)"
)
SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


def discover_completed_runs() -> Dict[str, str]:
    """tag -> run_dir, only for exit=0 jobs, across every batch log found."""
    tag_to_dir: Dict[str, str] = {}
    for rel_path in BATCH_LOGS:
        path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            for line in f:
                m = FINISHED_RE.search(line)
                if m and m.group("exit") == "0":
                    tag_to_dir[m.group("tag")] = m.group("run_dir")
    return tag_to_dir


def load_rewards(run_dir: str) -> Optional[List[float]]:
    path = os.path.join(REPO_ROOT, run_dir, "federated_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        history = json.load(f)
    rewards = [r for r in history.get("eval_reward", []) if r is not None]
    return rewards or None


def se_diff(std_a: float, n_a: int, std_b: float, n_b: int) -> float:
    var = (std_a ** 2) / max(n_a, 1) + (std_b ** 2) / max(n_b, 1)
    return math.sqrt(var) if var > 0 else 0.0


def main() -> None:
    tag_to_dir = discover_completed_runs()

    by_strategy: Dict[str, Dict[int, str]] = {}
    for tag, run_dir in tag_to_dir.items():
        m = re.search(r"^(.*)_seed(\d+)$", tag)
        if not m:
            continue
        strategy, seed = m.group(1), int(m.group(2))
        by_strategy.setdefault(strategy, {})[seed] = run_dir

    rows = []
    for strategy in sorted(by_strategy):
        seeds = by_strategy[strategy]
        means, bests = [], []
        for seed in sorted(seeds):
            rewards = load_rewards(seeds[seed])
            if not rewards:
                continue
            means.append(sum(rewards) / len(rewards))
            bests.append(max(rewards))
        if not means:
            continue
        mean_of_means = sum(means) / len(means)
        std_across_seeds = statistics.pstdev(means) if len(means) > 1 else 0.0
        best_round_mean = sum(bests) / len(bests)
        diff = mean_of_means - FEDAVG_REF["mean"]
        se = se_diff(std_across_seeds, len(means), FEDAVG_REF["std"], FEDAVG_REF["n"])
        signal = abs(diff) / se if se > 0 else float("inf")
        rows.append({
            "strategy": strategy,
            "n_seeds": len(means),
            "mean": mean_of_means,
            "std": std_across_seeds,
            "best_round_mean": best_round_mean,
            "diff_vs_fedavg": diff,
            "abs_diff_over_se": signal,
        })

    rows.sort(key=lambda r: -r["mean"])  # best mean reward first (least negative)

    # ---- console summary -------------------------------------------------
    print(f"Reference (fedavg, n={FEDAVG_REF['n']}): mean={FEDAVG_REF['mean']:.1f} "
          f"std={FEDAVG_REF['std']:.1f}  best_round_mean={FEDAVG_REF['best_round_mean']:.1f}")
    print(f"{'strategy':20s} {'n':>2s} {'mean':>10s} {'std':>8s} {'best_rnd':>10s} "
          f"{'diff_v_fedavg':>14s} {'|diff|/SE':>10s}")
    for r in rows:
        print(f"{r['strategy']:20s} {r['n_seeds']:2d} {r['mean']:10.1f} {r['std']:8.1f} "
              f"{r['best_round_mean']:10.1f} {r['diff_vs_fedavg']:+14.1f} {r['abs_diff_over_se']:10.2f}")
    for name, val in BASELINES.items():
        print(f"  (reference) {name}: {val}")

    # ---- markdown for fidings/divergence_investigation.md ---------------
    today = datetime.now().strftime("%Y-%m-%d")
    winner = rows[0] if rows else None
    beats_fedavg = [r for r in rows if r["abs_diff_over_se"] >= 2.0 and r["diff_vs_fedavg"] > 0]
    best_still_loses_to_baselines = (
        winner is not None and winner["mean"] < BASELINES["fixed_time"]
        and winner["mean"] < BASELINES["max_pressure"]
    )

    lines = []
    lines.append(f"## 27. Phase 2 aggregation-strategy comparison, 7-city, auto-generated {today}\n")
    lines.append(
        "**Auto-generated by `experiments/analyze_phase2_strategies.py`** once the overnight "
        "strategy-comparison batch (`ema_loss`, `ema_alignment`, `velocity_novelty`, "
        "`gradient_survival`, `clustered_fedavg`, all `--dueling --n_step 3`, 7-city roster, "
        "masked-head fix on) finished. Compares each strategy's per-seed mean/best-round eval "
        "reward (loaded straight from each run's `federated_history.json`) against the known "
        "`fedavg` reference (§23: 5 seeds, mean -6918.4 std 889.0, best-round mean -2182.0).\n"
    )
    lines.append("| strategy | seeds | mean reward | std | best-round mean | vs fedavg | \\|diff\\|/SE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        tag = " *(reference)*" if r["strategy"] == "fedavg" else ""
        lines.append(
            f"| `{r['strategy']}`{tag} | {r['n_seeds']} | {r['mean']:.1f} | {r['std']:.1f} | "
            f"{r['best_round_mean']:.1f} | {r['diff_vs_fedavg']:+.1f} | {r['abs_diff_over_se']:.2f} |"
        )
    lines.append(
        f"| `fedavg` *(known, §23)* | {FEDAVG_REF['n']} | {FEDAVG_REF['mean']:.1f} | "
        f"{FEDAVG_REF['std']:.1f} | {FEDAVG_REF['best_round_mean']:.1f} | — | — |"
    )
    lines.append("")

    if winner:
        lines.append(
            f"**Best mean reward: `{winner['strategy']}`** ({winner['mean']:.1f}, "
            f"{winner['n_seeds']} seeds, |diff|/SE={winner['abs_diff_over_se']:.2f} vs fedavg). "
            + ("This clears the |diff|/SE >= 2 bar this project has used elsewhere as a real "
               "(not noise-level) signal." if winner["abs_diff_over_se"] >= 2.0 else
               "This does **not** clear the |diff|/SE >= 2 bar this project has used elsewhere "
               "for a real (not noise-level) signal -- treat as a lead, not a settled result.")
        )
    if beats_fedavg:
        names = ", ".join(f"`{r['strategy']}`" for r in beats_fedavg)
        lines.append(f"\n**Strategies clearing that bar over fedavg: {names}.**")
    else:
        lines.append(
            "\n**No strategy clears |diff|/SE >= 2 over plain `fedavg` on mean reward.** "
            "Phase 2's core question (\"does any smarter aggregation strategy beat plain FedAvg\") "
            "reads as a negative/null result so far, on the seed counts gathered."
        )

    lines.append(
        "\n**The bigger unresolved issue is unchanged by this batch.** " +
        (f"Even the best strategy here (`{winner['strategy']}`, mean {winner['mean']:.1f}) "
         if winner else "Every strategy tested ") +
        f"is still far below both rule-based baselines (`fixed_time` {BASELINES['fixed_time']}, "
        f"`max_pressure` {BASELINES['max_pressure']}, §24 -- single-episode, not yet multi-seed). "
        "Comparing aggregation strategies against each other doesn't touch this gap; all of them "
        "lose to trivial heuristics by 3-4 orders of magnitude. Open item 7 in this file's "
        "\"Open questions / next steps\" list is still the higher-priority open question."
    )
    lines.append(
        "\n**Recommended next step (not auto-executed -- needs a decision, same as item 7 "
        "already flagged):** before spending more compute on further strategy seeds or Phase 2 "
        "scale-up, investigate *why* every trained-DQN configuration loses to `fixed_time`/"
        "`max_pressure` on the 7-city holdout. §26's mechanism dig (residual end-of-episode "
        "congestion, not policy collapse) narrows this but doesn't resolve it -- the standing "
        "fork from §26 (aggregation dilution vs. genuine undertraining) is still open. The "
        "2026-08-14 40-round `fedavg` mechanism-test run (`results/run_2026_08_14-00_59_53_6995`) "
        "exists as data toward that fork and hasn't been read yet as of this writeup."
    )
    lines.append("")

    new_section = "\n".join(lines)

    with open(FINDINGS_PATH) as f:
        content = f.read()
    marker = "## Open questions / next steps"
    idx = content.index(marker)
    updated = content[:idx] + new_section + "\n" + content[idx:]
    with open(FINDINGS_PATH, "w") as f:
        f.write(updated)
    print(f"\nAppended §27 to {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
