"""Delay / completion table for the Braun comparison at iteration 85, all seeds found.

Reads results/braun/eval/*.json directly (per-episode trip metrics, one pipeline for
every row). Per learned run: mean over its episodes; per arm: mean and SE over seeds.
    python3 baselines/braun/table_it85.py
"""
import glob
import json
import os
import statistics as st

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(W, "results", "braun", "eval")
SCEN = ("grid4x4", "cologne3", "ingolstadt7")


def run_mean(path):
    eps = json.load(open(path))["episodes"]
    return st.fmean(e["delay"] for e in eps), 100 * st.fmean(e["completion"] for e in eps)


def arm(pattern):
    files = sorted(glob.glob(os.path.join(E, pattern)))
    rows = [run_mean(f) for f in files]
    d = [r[0] for r in rows]
    c = [r[1] for r in rows]
    se = st.pstdev(d) / len(d) ** .5 if len(d) > 1 else 0.0
    return dict(n=len(rows), delay=st.fmean(d), comp=st.fmean(c), se=se, per_seed=d)


ROWS = [
    ("Braun, own phases (sampled)", "braun_{s}_synthfb_s*_it0085_sample.json"),
    ("Braun, own phases (greedy)", "braun_{s}_synthfb_s*_it0085_greedy.json"),
    ("Braun, RESCO phases (sampled)", "braun_{s}_native_s*_it0085_sample.json"),
    ("Braun, RESCO phases (greedy)", "braun_{s}_native_s*_it0085_greedy.json"),
    ("Braun max pressure, own phases", "braun_{s}_synthfb_max-pressure.json"),
    ("Braun max pressure, RESCO phases", "braun_{s}_native_max-pressure.json"),
    ("Ours, phase-relational", "ours_{s}_phase_*.json"),
    ("Our max pressure", "ours_{s}_max_pressure_*.json"),
]

out = {}
for label, pat in ROWS:
    out[label] = {s: arm(pat.format(s=s)) for s in SCEN}

print("%-34s" % "delay s / completion % (n)" + "".join("%26s" % s for s in SCEN))
for label, _ in ROWS:
    print("%-34s" % label + "".join(
        "%14.1f / %5.1f%% (%d)" % (out[label][s]["delay"], out[label][s]["comp"], out[label][s]["n"])
        for s in SCEN))

print("\n|diff|/SE on delay vs ours (unpaired, population std):")
ours = out["Ours, phase-relational"]
for label in ("Braun, own phases (sampled)", "Braun, RESCO phases (sampled)"):
    zs = []
    for s in SCEN:
        a, b = out[label][s], ours[s]
        se = (a["se"] ** 2 + b["se"] ** 2) ** .5
        zs.append(abs(a["delay"] - b["delay"]) / se if se else float("inf"))
    print("  %-32s" % label + "".join("%10.2f" % z for z in zs))
json.dump(out, open(os.path.join(W, "results", "braun", "table_it85.json"), "w"), indent=1)
