"""Summarise the unseen-network test (fidings sec 112).

Reads results/unseen/<network>/ours_*.json (per-episode trip metrics from
eval_ours.py). Learned readouts: mean over episodes per checkpoint, then mean and
SE over the six seeds. Rule-based controllers: mean over episodes.
    python analyse/unseen_table.py
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "unseen")
NETS = ("cologne1", "cologne8", "ingolstadt21", "grid_3x3_drop20", "grid_4x4_drop30",
        "grid_5x5_drop20", "grid_6x6_drop20")
KEYS = ("delay", "completion", "ti_wait", "trip_time")


def load(path):
    eps = json.load(open(path))["episodes"]
    return {k: st.fmean(e[k] for e in eps) for k in KEYS}, len(eps)


def group(net, pat):
    files = sorted(glob.glob(os.path.join(ROOT, net, pat)))
    rows = [load(f)[0] for f in files]
    if not rows:
        return None
    out = {"n": len(rows)}
    for k in KEYS:
        v = [r[k] for r in rows]
        out[k] = st.fmean(v)
        out[k + "_se"] = st.pstdev(v) / len(v) ** .5 if len(v) > 1 else 0.0
        out[k + "_all"] = v
    return out


def z(a, b, k):
    se = (a[k + "_se"] ** 2 + b[k + "_se"] ** 2) ** .5
    return abs(a[k] - b[k]) / se if se else float("inf")


summary = {}
print("%-17s %-15s %4s %9s %8s %9s" % ("network", "controller", "n", "delay s", "compl %", "wait s"))
for net in NETS:
    g = {"phase-relational": group(net, "ours_*_phase_*.json"),
         "indexed": group(net, "ours_*_indexed_*.json"),
         "max pressure": group(net, "ours_*_max_pressure_*.json"),
         "fixed time": group(net, "ours_*_fixed_time_*.json")}
    summary[net] = g
    for name, r in g.items():
        if r is None:
            print("%-17s %-15s  (not yet)" % (net, name))
            continue
        print("%-17s %-15s %4d %9.1f %8.1f %9.1f" % (net, name, r["n"], r["delay"], 100 * r["completion"], r["ti_wait"]))
    p, i, m = g["phase-relational"], g["indexed"], g["max pressure"]
    if p and i:
        print("%-17s   phase vs indexed: delay |d|/SE %.1f, completion %.1f" % ("", z(p, i, "delay"), z(p, i, "completion")))
    if p and m and p["n"] > 1:
        better = sum(1 for d in p["delay_all"] if d < m["delay"])
        wbetter = sum(1 for w in p["ti_wait_all"] if w < m["ti_wait"])
        print("%-17s   phase vs max pressure: delay lower on %d/%d seeds, wait lower on %d/%d"
              % ("", better, p["n"], wbetter, p["n"]))
    print()
json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=1)
