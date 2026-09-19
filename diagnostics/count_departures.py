"""Count vehicles departing inside each scenario's evaluation window.

`arrived` in the results tables is meaningless without this denominator: a
reader cannot tell whether 2615 is 93% of demand or 60% of it.
"""
import os, sys, xml.etree.ElementTree as ET

W = "/mnt/c/users/Deea/SUMO_2/SUMO_Reinforcement_learning_traffic/SUMO_Reinforcement_learning_traffic/.claude/worktrees/rescofull-writeup"
os.chdir(W)

SCEN = {
    "cologne3":    ("sumo_rl/nets/RESCO/cologne3/cologne3.rou.xml",       25200, 28800),
    "ingolstadt7": ("sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.rou.xml", 57600, 61200),
    "grid4x4":     ("sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml",           0,  3600),
    "arterial4x4": ("sumo_rl/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml",   0,  3600),
}

for name, (path, t0, t1) in SCEN.items():
    if not os.path.exists(path):
        print(f"{name:14s} MISSING {path}")
        continue
    root = ET.parse(path).getroot()
    inwin = total = 0
    flows = 0
    for v in root.iter():
        if v.tag == "vehicle" or v.tag == "trip":
            d = v.get("depart")
            if d is None:
                continue
            try:
                d = float(d)
            except ValueError:
                continue
            total += 1
            if t0 <= d < t1:
                inwin += 1
        elif v.tag == "flow":
            flows += 1
    extra = f"  (+{flows} <flow> elements, not counted)" if flows else ""
    print(f"{name:14s} window {t0}-{t1}:  departing in window = {inwin:5d}"
          f"   (file total {total}){extra}")
