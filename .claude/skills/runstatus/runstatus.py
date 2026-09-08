#!/usr/bin/env python3
"""Live status of a training batch: progress, pace, stalls, lock-in.

Reads each run's own federated_history.json (rewritten every round, so it is
current even mid-run) rather than tailing logs. Distinguishes a genuinely
stalled job from an idle-looking one -- host sleep freezes a run without
killing it and it resumes cleanly on wake (sec 30, sec 42), so a long gap is
not by itself a reason to restart anything.

Usage:
    python .claude/skills/runstatus/runstatus.py --batch-log results/foo.log
    python .claude/skills/runstatus/runstatus.py --dirs "results/run_2026_09_0*"
    python .claude/skills/runstatus/runstatus.py            # recently-touched runs
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_lib"))
import runs as runlib  # noqa: E402

# Round-to-round eval std below this is the cheap confident-lock-in screen
# (sec 33 -- known to have false negatives, so it is a hint, not a verdict).
LOCKIN_STD = 50.0
STALL_MINUTES = 45
# With no --dirs/--batch-log, only look at runs touched recently. Globbing all
# of results/ parses ~450MB of history for a few hundred numbers.
DEFAULT_RECENT_DAYS = 2

COLW = (5, 11, 4, 11, 7)


def live_pids() -> Set[int]:
    try:
        out = subprocess.run(["ps", "-eo", "pid,args", "--no-headers"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids = set()
    for line in out.splitlines():
        line = line.strip()
        if "experiments.federated_training" not in line:
            continue
        head = line.split(None, 1)[0]
        if head.isdigit():
            pids.add(int(head))
    return pids


def run_pid(run_dir: str) -> Optional[int]:
    """run_2026_09_07-19_30_37_985023 -> 985023 (the PID suffix, sec 22 fix)."""
    m = re.search(r"_(\d+)$", os.path.basename(run_dir.rstrip("/")))
    return int(m.group(1)) if m else None


def dirs_by_pid(results_root: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    try:
        entries = os.listdir(results_root)
    except OSError:
        return out
    for name in entries:
        if not name.startswith("run_"):
            continue
        pid = run_pid(name)
        if pid is not None:
            out[pid] = os.path.join(results_root, name)
    return out


def summarize(run_dir: str) -> Optional[Dict[str, object]]:
    hist = runlib.load_history(run_dir)
    if hist is None:
        return None
    rewards = runlib.floats(hist, "eval_reward")
    if not rewards:
        return None
    stds = runlib.floats(hist, "eval_reward_std")
    log_path = os.path.join(run_dir, "training.log")
    hist_path = os.path.join(run_dir, "federated_history.json")
    mtime = os.path.getmtime(log_path if os.path.exists(log_path) else hist_path)
    best = max(rewards)
    return {
        "rounds": len(rewards),
        "best": best,
        "best_round": rewards.index(best) + 1,
        "last": rewards[-1],
        "idle_min": (time.time() - mtime) / 60.0,
        "lockin_rounds": [i + 1 for i, s in enumerate(stds) if s < LOCKIN_STD],
        "is_true_holdout": runlib.scalar(hist, "is_true_holdout"),
    }


def short(tag: str, width: int = 16) -> str:
    return tag if len(tag) <= width else tag[:width - 1] + "\u2026"


def blank_row(tag: str, note: str) -> None:
    cells = " ".join(f"{'--':>{w}}" for w in COLW)
    print(f"{short(tag):<16} {cells}  {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-log", help="run_concurrent_batch.sh log to read tags from")
    ap.add_argument("--dirs", help="comma-separated run-dir globs (alternative to --batch-log)")
    ap.add_argument("--total-rounds", type=int, help="expected --rounds, for ETA")
    ap.add_argument("--days", type=float, default=DEFAULT_RECENT_DAYS,
                    help=f"with no --dirs/--batch-log, only runs touched in the last N days "
                         f"(default {DEFAULT_RECENT_DAYS})")
    args = ap.parse_args()

    results_root = os.path.join(runlib.REPO, "results")
    pids = live_pids()
    jobs: List[Tuple[str, str, str]] = []  # (tag, run_dir, note)
    done: Dict[str, str] = {}

    if args.batch_log:
        batch_jobs, warns = runlib.parse_batch_log(args.batch_log)
        if not batch_jobs and warns:
            raise SystemExit(warns[0])
        pid_dirs = dirs_by_pid(results_root)
        # Live dirs not already claimed by a finished job -- used to place
        # started-but-unfinished tags. Only unambiguous when exactly one is free.
        claimed = {j.run_dir for j in batch_jobs if j.run_dir}
        free_live = [pid_dirs[p] for p in sorted(pids) if p in pid_dirs and pid_dirs[p] not in claimed]
        for job in batch_jobs:
            if job.exit_code is not None:
                done[job.tag] = job.exit_code
                jobs.append((job.tag, job.run_dir, job.note))
            elif len(free_live) == 1:
                jobs.append((job.tag, free_live[0], "matched to the only live run"))
            else:
                # Guessing here would assign the same dir to every unfinished
                # tag, which is worse than saying nothing.
                jobs.append((job.tag, "", f"running; {len(free_live)} live runs, mapping ambiguous"))
        n_started = len(batch_jobs)
        print(f"batch: {n_started} started, {len(done)} finished "
              f"({sum(1 for v in done.values() if v == '0')} exit=0)")
        bad = {t: v for t, v in done.items() if v != "0"}
        if bad:
            print(f"  NONZERO EXITS: {bad}")
        for w in warns:
            print(f"  WARNING: {w}")
    else:
        if args.dirs:
            candidates = [d for spec in args.dirs.split(",") for d in sorted(glob.glob(spec.strip()))]
        else:
            cutoff = time.time() - args.days * 86400
            candidates = sorted(
                d for d in glob.glob(os.path.join(results_root, "run_*"))
                if os.path.isdir(d) and os.path.getmtime(d) >= cutoff
            )
            print(f"(no --dirs/--batch-log: showing runs touched in the last {args.days:g} day(s); "
                  f"{len(candidates)} of {len(glob.glob(os.path.join(results_root, 'run_*')))})")
        jobs = [(os.path.basename(d), d, "") for d in candidates if os.path.isdir(d)]

    print(f"\n{len(pids)} live federated_training process(es)\n")
    header = f"{'tag':<16} {'rnds':>5} {'best':>11} {'@rd':>4} {'last':>11} {'idle':>7}  state"
    print(header)
    print("-" * len(header))

    for tag, run_dir, note in jobs:
        if not run_dir:
            blank_row(tag, note or "run_dir unknown")
            continue
        s = summarize(run_dir)
        if s is None:
            blank_row(tag, f"no history yet ({os.path.basename(run_dir)})")
            continue
        pid = run_pid(run_dir)
        alive = pid is not None and pid in pids
        if tag in done:
            state = "done exit=" + done[tag]
        elif not alive and s["idle_min"] > STALL_MINUTES:
            # Long idle and no process: an ordinary finished run, not news.
            state = "not running"
        elif not alive:
            state = "PROCESS GONE (exited in the last hour without a finish marker)"
        elif s["idle_min"] > STALL_MINUTES:
            state = f"no log write for {s['idle_min']:.0f}min -- stalled, or host slept (resumes on wake)"
        else:
            state = "running"
        if args.total_rounds and alive:
            state += f"  [{s['rounds']}/{args.total_rounds}, {s['rounds'] / args.total_rounds:.0%}]"
        if note:
            state += f"  ({note})"
        print(f"{short(tag):<16} {s['rounds']:>5} {s['best']:>11.1f} {s['best_round']:>4} "
              f"{s['last']:>11.1f} {s['idle_min']:>6.0f}m  {state}")
        if s["is_true_holdout"] is False:
            print(f"{'':<16} WARNING: is_true_holdout=False -- evaluating on a TRAINING city (sec 25)")
        if s["lockin_rounds"]:
            rl = s["lockin_rounds"]
            shown = ", ".join(str(r) for r in rl[:8]) + (" ..." if len(rl) > 8 else "")
            print(f"{'':<16} lock-in screen fired on {len(rl)} round(s): {shown}")

    print("\nNotes: the lock-in screen (eval std < 50) has known false negatives (sec 33/49) --")
    print("confirm with diagnostics/reeval_checkpoint.py --pad_to_true_holdout at 30 episodes.")
    print("Host sleep freezes a run without killing it; it resumes on wake. Don't restart on")
    print("a long idle gap alone -- check whether the process is still alive first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
