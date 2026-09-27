"""Controller-agnostic trip metrics shared by EVERY row of the Braun comparison.

Stdlib only, so it imports identically in this project's miniconda env (our
checkpoints, our rule-based controllers) and in Braun's isolated uv venv (his
learned policy and his baselines).

Two independent measurements are produced for every episode:

1. ``run_episode`` -- the EXACT function body of
   ``diagnostics/eval_paper_metrics.py::run_episode`` (the pipeline behind every
   delay / trip-time number in fidings sec 103b and paper Table VI).  It is
   extracted from that file's source with ``ast`` at runtime and exec'd, rather
   than imported, because importing that module pulls in torch/agents/
   experiments, which Braun's venv does not have.  Nothing is retyped, so the
   two cannot drift.  It polls the simulator every 5 simulated seconds
   (delta_time), tracks per-vehicle trip time and last-polled timeLoss, and
   counts ``arrived`` = vehicles that left the network during the episode.

2. SUMO's own ``--tripinfo-output`` for the same episode, parsed here: per
   arrived vehicle duration, timeLoss and waitingTime (total time at
   speed < 0.1 m/s).  This is the definition RESCO's published numbers use, it
   is exact (no 5 s quantisation), and it is the only source of a per-vehicle
   WAITING TIME that means the same thing for every controller.

Completion is arrived / (vehicles in the route file departing inside the
evaluation window), i.e. the "Dep." column of paper Table I.
"""

from __future__ import annotations

import ast
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_PAPER_METRICS = REPO / 'diagnostics' / 'eval_paper_metrics.py'


def load_run_episode(timeloss_constant):
    """Return eval_paper_metrics.run_episode, exec'd from its own source.

    ``timeloss_constant``: traci.constants.VAR_TIMELOSS to use the function's
    subscription branch (our env), or None to use its per-vehicle
    ``getTimeLoss`` fallback branch.  Braun's runtime MUST use None: his
    VehicleSnapshotCollector owns per-vehicle subscriptions, and a second
    subscribe() on the same vehicle replaces the first one's variable list, so
    the two would silently corrupt each other.  Both branches read the same
    SUMO quantity (timeLoss at the poll instant).
    """
    source = EVAL_PAPER_METRICS.read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_episode')
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {'statistics': statistics, '_TIMELOSS': timeloss_constant}
    exec(compile(module, str(EVAL_PAPER_METRICS), 'exec'), namespace)
    return namespace['run_episode']


def route_departures(route_file: str | Path, begin: float, end: float) -> int:
    """Vehicles/trips in a route file departing in [begin, end)."""
    n = 0
    for _, elem in ET.iterparse(str(route_file), events=('end',)):
        if elem.tag in ('vehicle', 'trip'):
            dep = elem.get('depart')
            try:
                d = float(dep)
            except (TypeError, ValueError):
                d = None
            if d is not None and begin <= d < end:
                n += 1
            elem.clear()
    return n


def parse_tripinfo(path: str | Path, begin: float, end: float) -> dict:
    """Mean duration / timeLoss / waitingTime over vehicles that departed in the
    window and arrived by its end (tripinfo only lists arrived vehicles)."""
    durations, losses, waits = [], [], []
    for _, elem in ET.iterparse(str(path), events=('end',)):
        if elem.tag == 'tripinfo':
            dep = float(elem.get('depart'))
            arr = float(elem.get('arrival'))
            if begin <= dep < end and arr <= end:
                durations.append(float(elem.get('duration')))
                losses.append(float(elem.get('timeLoss')))
                waits.append(float(elem.get('waitingTime')))
            elem.clear()

    def mean(xs):
        return statistics.fmean(xs) if xs else float('nan')

    return {
        'ti_arrived': len(durations),
        'ti_trip_time': mean(durations),
        'ti_delay': mean(losses),
        'ti_wait': mean(waits),
    }
