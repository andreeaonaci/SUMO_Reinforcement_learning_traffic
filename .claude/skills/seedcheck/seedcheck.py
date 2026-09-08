#!/usr/bin/env python3
"""Multi-seed significance analysis for federated_training runs.

Computes this project's standing |diff|/SE statistic between two (or more)
experimental arms, on the three measures the fidings log uses throughout:
best-ever round, final round, and mean over a post-warmup window.

Why this exists: this computation is the most-repeated analysis step in the
project and had been done ad hoc every time. Doing it ad hoc is also how
three separate "clean 3-seed wins" (CQL, TC-FedAvg, n_attn_layers=2) got
reported before the checks that would have flagged them as one-outlier-seed
artifacts. Those checks are built in here and are not optional.

Run discovery
-------------
Each arm is a comma-separated list of run-dir globs, or a batch-log tag:

    --arm "baseline=results/run_2026_09_0*_1?????"
    --arm "mylever=results/mylever_seed*"
    --batch-log results/foo.log --arm "baseline=tag:base" --arm "mylever=tag:lever"

Seeds and full flag sets are read from each run's own ``training.log`` header
(the pprint'd argparse dump), not from directory names -- so arms are matched
on what actually ran, and a config difference beyond the flag under test gets
reported rather than silently confounding the comparison.

Usage
-----
    python .claude/skills/seedcheck/seedcheck.py \
        --arm "baseline=results/runA*" --arm "treatment=results/runB*" \
        --baseline baseline [--window auto|all|21:|1:20] [--metric reward|waiting]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_lib"))
import runs as runlib  # noqa: E402

# This project's bar for "a real, non-noise signal" (used since fidings sec 20).
SIGNAL_BAR = 2.0
# Below this many seeds, nothing is confirmable at this roster/budget -- three
# separate leads died going from 3 seeds to 6 (fidings sec 73-77, sec 91).
CONFIRM_MIN_SEEDS = 5


@dataclass
class Run:
    run_dir: str
    args: Dict[str, object]
    rewards: List[float]
    waiting: List[float]
    is_true_holdout: Optional[bool]
    eval_city: Optional[str]

    @property
    def seed(self) -> Optional[int]:
        s = self.args.get("seed")
        return s if isinstance(s, int) else None


def load_run(run_dir: str) -> Optional[Run]:
    hist = runlib.load_history(run_dir)
    if hist is None:
        return None
    rewards = runlib.floats(hist, "eval_reward")
    if not rewards:
        return None
    return Run(
        run_dir=run_dir,
        args=runlib.parse_training_log_args(run_dir),
        rewards=rewards,
        waiting=runlib.floats(hist, "eval_waiting_time"),
        is_true_holdout=runlib.scalar(hist, "is_true_holdout"),
        eval_city=runlib.scalar(hist, "eval_city_name"),
    )


def resolve_arm(spec: str, batch_tags: Dict[str, str]) -> Tuple[List[str], List[str]]:
    dirs: List[str] = []
    missing: List[str] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.startswith("tag:"):
            tag = piece[4:]
            if tag in batch_tags and batch_tags[tag]:
                dirs.append(batch_tags[tag])
            else:
                missing.append(tag)
        else:
            dirs.extend(sorted(glob.glob(piece)))
    seen, uniq = set(), []
    for d in dirs:
        d = d.rstrip("/")
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            uniq.append(d)
    return uniq, missing


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------
def resolve_window(spec: str, n_rounds: int) -> Tuple[int, int]:
    """Return an inclusive 1-indexed (start, end) round window.

    'auto' reproduces the convention in the fidings log: skip the first 20
    rounds as warm-up on long runs (the "mean(21-63)" numbers), use every
    round on short ones.
    """
    if spec == "all":
        return 1, n_rounds
    if spec == "auto":
        return (21, n_rounds) if n_rounds >= 30 else (1, n_rounds)
    m = re.fullmatch(r"(\d+)?:(\d+)?", spec)
    if not m:
        raise SystemExit(f"bad --window {spec!r}; use auto|all|START:END|START:|:END")
    start = int(m.group(1)) if m.group(1) else 1
    end = int(m.group(2)) if m.group(2) else n_rounds
    return max(1, start), min(end, n_rounds)


def measures(values: Sequence[float], window: Tuple[int, int], higher_is_better: bool) -> Dict[str, float]:
    """best/final are over ALL rounds; only 'mean' respects the window.

    That matches the fidings convention -- "best-ever round" there means
    best of the whole run, and 'mean(21-63)' is the windowed one.
    """
    start, end = window
    win = list(values[start - 1:end]) or list(values)
    return {
        "best": max(values) if higher_is_better else min(values),
        "final": values[-1],
        "mean": statistics.fmean(win),
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def se_diff(std_a: float, n_a: int, std_b: float, n_b: int) -> float:
    """Unpaired two-sample SE -- the formula from
    experiments/analyze_phase2_strategies.py::se_diff."""
    var = (std_a ** 2) / max(n_a, 1) + (std_b ** 2) / max(n_b, 1)
    return math.sqrt(var) if var > 0 else 0.0


def unpaired_stat(a: Sequence[float], b: Sequence[float], sample_std: bool = False) -> Optional[float]:
    """|diff|/SE across seeds.

    ``sample_std=False`` uses population std (statistics.pstdev), matching
    experiments/analyze_phase2_strategies.py and analyze_phase1.py -- i.e.
    every historical number in fidings/divergence_investigation.md.
    ``sample_std=True`` uses the sample (n-1) std, which is the more
    defensible estimator for inference across seeds and always yields a
    SMALLER statistic (by sqrt((n-1)/n): ~18% lower at n=3, ~11% at n=5).
    Both are reported so a borderline result cannot hide behind the choice.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    f = statistics.stdev if sample_std else statistics.pstdev
    se = se_diff(f(a), len(a), f(b), len(b))
    return abs(statistics.fmean(a) - statistics.fmean(b)) / se if se > 0 else None


def paired_stat(diffs: Sequence[float]) -> Optional[float]:
    if len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    return abs(statistics.fmean(diffs)) / (sd / math.sqrt(len(diffs))) if sd > 0 else None


def compare(measure: str,
            treat: List[Tuple[Optional[int], float]],
            base: List[Tuple[Optional[int], float]],
            higher_is_better: bool) -> Dict[str, object]:
    tv = [v for _, v in treat]
    bv = [v for _, v in base]
    mean_t, mean_b = statistics.fmean(tv), statistics.fmean(bv)
    diff = mean_t - mean_b
    out: Dict[str, object] = {
        "measure": measure,
        "mean_treat": mean_t,
        "mean_base": mean_b,
        "diff": diff,
        "improved": diff > 0 if higher_is_better else diff < 0,
        "unpaired": unpaired_stat(tv, bv),
        "unpaired_sample": unpaired_stat(tv, bv, sample_std=True),
        "n_treat": len(tv),
        "n_base": len(bv),
    }
    # Paired analysis + per-seed direction, only where seeds actually match.
    tmap = {s: v for s, v in treat if s is not None}
    bmap = {s: v for s, v in base if s is not None}
    shared = sorted(set(tmap) & set(bmap))
    if len(shared) >= 2:
        diffs = [tmap[s] - bmap[s] for s in shared]
        out["paired"] = paired_stat(diffs)
        out["wins"] = sum(1 for d in diffs if (d > 0) == higher_is_better and d != 0)
        out["n_pairs"] = len(diffs)
        # Drop-1 influence: does the headline statistic survive removing any
        # single seed? This is the check that would have caught the 3-seed
        # leads that later evaporated.
        drop = [s for s in (unpaired_stat([tmap[k] for k in shared if k != x],
                                          [bmap[k] for k in shared if k != x])
                            for x in shared) if s is not None]
        if drop:
            out["drop1_min"] = min(drop)
            out["drop1_max"] = max(drop)
    return out


def cleared(rep: Dict[str, object]) -> bool:
    return (rep["unpaired"] or 0) >= SIGNAL_BAR


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt(x: Optional[float], nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def config_diff(arms: Dict[str, List[Run]]) -> Dict[str, Dict[str, set]]:
    """Which argparse flags differ between arms (ignoring per-run bookkeeping)."""
    ignore = {"seed", "resume", "resume_from"}
    keys: set = set()
    for arm_runs in arms.values():
        for r in arm_runs:
            keys |= set(r.args)
    keys -= ignore
    per_arm: Dict[str, Dict[str, set]] = {}
    for name, arm_runs in arms.items():
        vals: Dict[str, set] = {}
        for r in arm_runs:
            for k in keys & set(r.args):
                vals.setdefault(k, set()).add(repr(r.args[k]))
        per_arm[name] = vals
    return per_arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SPEC",
                    help="Arm definition; SPEC is comma-separated run-dir globs or tag:NAME")
    ap.add_argument("--baseline", help="Name of the arm to compare others against (default: first --arm)")
    ap.add_argument("--batch-log", action="append", default=[],
                    help="run_concurrent_batch.sh log to resolve tag: specs from")
    ap.add_argument("--window", default="auto", help="Round window for the mean measure (default auto)")
    ap.add_argument("--metric", default="reward", choices=["reward", "waiting"],
                    help="reward (higher better, default) or waiting (lower better)")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON as well")
    args = ap.parse_args()

    problems: List[str] = []
    batch_tags: Dict[str, str] = {}
    for path in args.batch_log:
        jobs, warns = runlib.parse_batch_log(path)
        problems.extend(warns)
        for job in jobs:
            if job.exit_code is None:
                problems.append(f"{job.tag}: started but never finished -- arm is incomplete")
            elif job.exit_code != "0":
                problems.append(f"{job.tag}: exit={job.exit_code} -- arm is incomplete")
            if job.run_dir:
                batch_tags[job.tag] = job.run_dir

    arms: Dict[str, List[Run]] = {}
    order: List[str] = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm needs NAME=SPEC, got {spec!r}")
        name, rest = spec.split("=", 1)
        name = name.strip()
        dirs, missing = resolve_arm(rest, batch_tags)
        for tag in missing:
            problems.append(f"arm {name!r}: tag {tag!r} has no usable run_dir in the batch log "
                            "-- that seed is MISSING from this arm, not merely unfinished")
        arm_runs = []
        for d in dirs:
            run = load_run(d)
            if run is None:
                problems.append(f"skipped {d}: no usable federated_history.json")
            else:
                arm_runs.append(run)
        if not arm_runs:
            raise SystemExit(f"arm {name!r} matched no usable runs (spec: {rest})")
        arms[name] = arm_runs
        order.append(name)

    baseline = args.baseline or order[0]
    if baseline not in arms:
        raise SystemExit(f"--baseline {baseline!r} is not one of {order}")

    higher_is_better = args.metric == "reward"
    metric_label = "eval_reward" if higher_is_better else "eval_waiting_time"

    print("=" * 78)
    print(f"seedcheck -- metric={metric_label} ({'higher' if higher_is_better else 'lower'} is better)")
    print("=" * 78)

    # ---- per-run detail -------------------------------------------------
    per_arm: Dict[str, Dict[str, List[Tuple[Optional[int], float]]]] = {}
    round_counts: Dict[str, set] = {}
    for name in order:
        print(f"\n[{name}]  {len(arms[name])} run(s)")
        print(f"  {'seed':>5}  {'rounds':>6}  {'best':>12}  {'final':>12}  {'mean(win)':>12}  window  holdout  dir")
        collected: Dict[str, List[Tuple[Optional[int], float]]] = {"best": [], "final": [], "mean": []}
        for r in sorted(arms[name], key=lambda r: (r.seed is None, r.seed or 0, r.run_dir)):
            if higher_is_better:
                series = r.rewards
            else:
                series = r.waiting or r.rewards
                if not r.waiting:
                    problems.append(f"{r.run_dir}: no eval_waiting_time recorded, fell back to eval_reward")
            win = resolve_window(args.window, len(series))
            m = measures(series, win, higher_is_better)
            for k in collected:
                collected[k].append((r.seed, m[k]))
            round_counts.setdefault(name, set()).add(len(series))
            ho = "TRUE" if r.is_true_holdout else ("FALLBACK" if r.is_true_holdout is False else "?")
            print(f"  {str(r.seed):>5}  {len(series):>6}  {m['best']:>12.2f}  {m['final']:>12.2f}  "
                  f"{m['mean']:>12.2f}  {win[0]}-{win[1]}  {ho:>8}  {os.path.basename(r.run_dir)}")
            if r.is_true_holdout is False:
                problems.append(
                    f"{r.run_dir}: is_true_holdout=False (eval city {r.eval_city!r}) -- "
                    "this is the sec 25 silent-fallback trap, NOT a cross-topology result")
            if not r.args:
                problems.append(f"{r.run_dir}: could not parse training.log args (seed/config unverified)")
        per_arm[name] = collected

    # ---- config sanity --------------------------------------------------
    cfgs = config_diff(arms)
    base_cfg = cfgs[baseline]
    differing: Dict[str, List[str]] = {}
    predates: Dict[str, List[str]] = {}
    for name in order:
        if name == baseline:
            continue
        for k, vals in cfgs[name].items():
            bvals = base_cfg.get(k, set())
            if vals == bvals:
                continue
            if not bvals:
                # The flag did not exist when the baseline ran: not a value
                # confound, but the flag under test shows up here too.
                predates.setdefault(name, []).append(f"{k} -> {sorted(vals)}")
            else:
                differing.setdefault(name, []).append(f"{k}: {sorted(bvals)} -> {sorted(vals)}")
    if differing or predates:
        print("\nConfig differences vs. baseline (the flag under test should be the ONLY entry):")
        for name in order:
            if name == baseline or not (differing.get(name) or predates.get(name)):
                continue
            print(f"  [{name}]")
            for it in sorted(differing.get(name, [])):
                print(f"    {it}")
            for it in sorted(predates.get(name, [])):
                print(f"    (baseline predates this flag) {it}")
        for name, items in differing.items():
            if len(items) > 1:
                problems.append(
                    f"arm {name!r} differs from baseline in {len(items)} flags, not 1 -- confounded "
                    "unless every extra difference is intended: " + "; ".join(sorted(items)))
            if any(it.startswith("rounds:") for it in items):
                problems.append(
                    f"arm {name!r} has a different --rounds than baseline: compute_eps_decay sizes the "
                    "exploration schedule from --rounds, so early rounds are NOT comparable "
                    "(the sec 69 dose-response confound)")
        if predates:
            print("  (a 'predates' entry means the baseline run was made before that flag existed,")
            print("   so it cannot be a value confound -- but the flag under test appears here too,")
            print("   and a baseline from a different code revision is its own risk.)")

    for name, counts in round_counts.items():
        if len(counts) > 1:
            problems.append(f"arm {name!r} has runs of differing length {sorted(counts)} -- "
                            "an unfinished run drags 'final' and shortens 'mean'")

    # ---- comparisons ----------------------------------------------------
    results: Dict[str, List[Dict[str, object]]] = {}
    for name in order:
        if name == baseline:
            continue
        print("\n" + "-" * 78)
        print(f"{name}  vs  {baseline}")
        print("-" * 78)
        reports = [compare(m, per_arm[name][m], per_arm[baseline][m], higher_is_better)
                   for m in ("best", "final", "mean")]
        results[name] = reports
        for rep in reports:
            arrow = "better" if rep["improved"] else "WORSE"
            print(f"\n  {rep['measure']:>5}: {rep['mean_treat']:.2f} vs {rep['mean_base']:.2f} "
                  f"(diff {rep['diff']:+.2f}, {arrow})")
            print(f"         |diff|/SE unpaired = {fmt(rep['unpaired'])}   "
                  f"[project convention (pstdev), n={rep['n_treat']}v{rep['n_base']}]")
            print(f"         same, sample std   = {fmt(rep['unpaired_sample'])}   "
                  f"[more conservative; always lower]")
            if "paired" in rep:
                print(f"         |diff|/SE paired   = {fmt(rep['paired'])}   "
                      f"({rep['wins']}/{rep['n_pairs']} seeds favor {name})")
            if "drop1_min" in rep:
                print(f"         drop-1 range       = {fmt(rep['drop1_min'])} .. {fmt(rep['drop1_max'])}")

        n_seeds = min(len(per_arm[name]["best"]), len(per_arm[baseline]["best"]))
        clears = [r for r in reports if cleared(r) and r["improved"]]
        regress = [r for r in reports if cleared(r) and not r["improved"]]
        print("\n  VERDICT")
        if regress:
            print(f"    NEGATIVE on {', '.join(r['measure'] for r in regress)} "
                  f"(clears the >={SIGNAL_BAR} bar in the WRONG direction)")
        if clears:
            print(f"    Clears the >={SIGNAL_BAR} bar on: {', '.join(r['measure'] for r in clears)}")
        elif not regress:
            print(f"    NULL -- no measure clears |diff|/SE >= {SIGNAL_BAR}")
        if n_seeds < CONFIRM_MIN_SEEDS:
            print(f"    SCREEN ONLY -- {n_seeds} seeds. Nothing under {CONFIRM_MIN_SEEDS}-6 seeds is")
            print("    confirmable at this roster/budget (sec 73-77, sec 91: three separate clean")
            print("    3-seed leads each collapsed to null at 6). Escalate before writing this up")
            print("    as a result, and do NOT report it as confirmed.")
        for r in reports:
            if "drop1_min" in r and cleared(r) and r["drop1_min"] < SIGNAL_BAR:
                print(f"    OUTLIER RISK on '{r['measure']}': dropping one seed takes the statistic to "
                      f"{r['drop1_min']:.2f}, below the bar -- one seed is carrying this result.")
            if cleared(r) and (r["unpaired_sample"] or 0) < SIGNAL_BAR:
                print(f"    CONVENTION-SENSITIVE on '{r['measure']}': clears the bar on the project's "
                      f"pstdev convention ({fmt(r['unpaired'])}) but not on sample std "
                      f"({fmt(r['unpaired_sample'])}).")
            if "wins" in r and r["improved"] and r["wins"] <= r["n_pairs"] / 2:
                print(f"    DIRECTION SPLIT on '{r['measure']}': only {r['wins']}/{r['n_pairs']} seeds "
                      "individually favor the treatment despite a positive mean.")

    if problems:
        print("\n" + "!" * 78)
        print("WARNINGS")
        print("!" * 78)
        for p in dict.fromkeys(problems):
            print(f"  - {p}")

    if args.json:
        print("\nJSON")
        print(json.dumps({"baseline": baseline, "comparisons": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
