#!/usr/bin/env python3
"""Shared run/batch-log parsing for the analysis skills.

Exists because seedcheck and runstatus each re-derived this and each lost a
fix that `experiments/analyze_phase2_strategies.py` already had: a job's
`finished` line carries an EMPTY run_dir when the job was `--resume`d (the
batch script finds the dir by `ls results/run_*_$PID`, and a resumed job
reuses an existing dir instead of creating one) or when it was a rule-based
baseline controller that never writes a run dir at all. 12 such lines exist
in this repo's logs. Silently skipping them drops seeds from an arm without
warning -- exactly the failure the skills exist to prevent.

Stdlib only, no torch/SUMO imports, so it runs anywhere the repo is checked
out.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Repo root: .claude/skills/_lib/runs.py -> up four levels.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FINISH_RE = re.compile(
    r"finished\s+(?P<tag>\S+)\s+base_dir=(?P<base_dir>\S+)\s+exit=(?P<exit>\d+)\s+run_dir=(?P<run_dir>\S*)"
)
# A resumed job names the dir it is resuming INTO in its own starting line --
# the fallback when the finish line's run_dir is empty.
STARTING_RESUME_RE = re.compile(
    # The flags are single-quoted, so exclude the closing quote from the path --
    # it is captured otherwise whenever --resume is the last flag in the list.
    r"starting\s+(?P<tag>\S+)\s+base_dir=\S+\s+flags='[^']*--resume\s+(?P<run_dir>[^\s']+)"
)
STARTING_RE = re.compile(r"starting\s+(?P<tag>\S+)\s+base_dir=(?P<base_dir>\S+)")

# Keys the skills actually read. A federated_history.json can reach 52MB,
# almost entirely `eval_per_model`, so slice these out of the raw text rather
# than parsing the whole document for a few hundred floats.
_WANTED = ("eval_reward", "eval_reward_std", "eval_waiting_time", "is_true_holdout", "eval_city_name")
_SLICE_RES = {k: re.compile(r'"%s"\s*:\s*(\[[^]\[]*\]|[^,}\n]+)' % k) for k in _WANTED}


def load_history(run_dir: str) -> Optional[Dict[str, object]]:
    """Read only the keys the skills need, falling back to a full parse."""
    path = os.path.join(run_dir, "federated_history.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    out: Dict[str, object] = {}
    for key, rx in _SLICE_RES.items():
        m = rx.search(raw)
        if not m:
            continue
        try:
            out[key] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    if "eval_reward" not in out:
        # Slicing missed (unusual formatting) -- pay for the full parse.
        try:
            full = json.loads(raw)
        except json.JSONDecodeError:
            return None
        out = {k: full[k] for k in _WANTED if k in full}
    return out or None


def floats(hist: Dict[str, object], key: str) -> List[float]:
    vals = hist.get(key) or []
    if not isinstance(vals, list):
        return []
    return [float(x) for x in vals if isinstance(x, (int, float))]


def scalar(hist: Dict[str, object], key: str):
    """Some keys are recorded per-round as a list, some as a bare value."""
    v = hist.get(key)
    if isinstance(v, list):
        return v[-1] if v else None
    return v


def parse_training_log_args(run_dir: str) -> Dict[str, object]:
    """Pull the argparse dump pprint'd at the top of a run's training.log.

    Takes the LAST such block, not the first: a `--resume`d run's log contains
    the original run's Arguments block followed by the resuming run's, and the
    resuming one is what actually produced the later rounds. Seven 63-round
    extended-budget runs in this repo are affected -- reading the first block
    reports rounds=20 for a run that actually did 63.
    """
    path = os.path.join(run_dir, "training.log")
    if not os.path.exists(path):
        return {}
    blocks: List[str] = []
    buf: List[str] = []
    depth = 0
    collecting = False
    armed = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if not collecting:
                if line.rstrip().endswith("Arguments:"):
                    armed = True
                    continue
                if armed:
                    if line.lstrip().startswith("{"):
                        collecting = True
                    else:
                        armed = False
                        continue
                else:
                    continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                blocks.append("".join(buf))
                buf, depth, collecting, armed = [], 0, False, False
    for block in reversed(blocks):
        try:
            parsed = ast.literal_eval(block)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


@dataclass
class BatchJob:
    tag: str
    run_dir: str
    exit_code: Optional[str]  # None = started but not finished
    note: str = ""


def parse_batch_log(path: str) -> Tuple[List[BatchJob], List[str]]:
    """Return (jobs, warnings) for a run_concurrent_batch.sh log."""
    warnings: List[str] = []
    if not os.path.exists(path):
        return [], [f"no such batch log: {path}"]
    resume_dirs: Dict[str, str] = {}
    started: List[str] = []
    finished: Dict[str, Tuple[str, str]] = {}  # tag -> (exit, run_dir)
    with open(path, errors="replace") as fh:
        for line in fh:
            mr = STARTING_RESUME_RE.search(line)
            if mr:
                resume_dirs[mr.group("tag")] = mr.group("run_dir")
            ms = STARTING_RE.search(line)
            if ms and ms.group("tag") not in started:
                started.append(ms.group("tag"))
            mf = FINISH_RE.search(line)
            if mf:
                finished[mf.group("tag")] = (mf.group("exit"), mf.group("run_dir"))

    jobs: List[BatchJob] = []
    for tag in started:
        if tag not in finished:
            jobs.append(BatchJob(tag=tag, run_dir="", exit_code=None))
            continue
        code, run_dir = finished[tag]
        note = ""
        if not run_dir:
            # The batch script's `ls results/run_*_$PID` found nothing.
            run_dir = resume_dirs.get(tag, "")
            if run_dir:
                note = "run_dir recovered from the --resume flag in its starting line"
            else:
                note = ("finish line has an empty run_dir and no --resume fallback -- "
                        "typically a --baseline_controller job, which writes no run dir")
                warnings.append(f"{tag}: {note}")
        jobs.append(BatchJob(tag=tag, run_dir=run_dir, exit_code=code, note=note))
    return jobs, warnings
