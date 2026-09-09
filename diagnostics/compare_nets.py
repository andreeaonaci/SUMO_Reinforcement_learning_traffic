"""Compare this project's vendored RESCO .net.xml files against RESCO's own.

Why (fidings sec 99, sec 101): sec 99 found three scenario-configuration
mismatches (route file, evaluation window, yellow_time) and fixed them. It never
checked the NET FILES themselves. While validating RESCO's hand-authored FRAP
signal config against our nets, four of RESCO's authored signal IDs for
ingolstadt7 turned out not to exist in our copy of that net -- which cannot
happen if the two files describe the same intersections.

md5 alone proves nothing here (netconvert re-encodes on version change), so this
compares the things that actually affect an experiment: signalised junction IDs,
their green-phase counts, and lane/edge counts.

Usage:
    python diagnostics/compare_nets.py --resco_src /path/to/RESCO
"""
import argparse
import os
import xml.etree.ElementTree as ET

SCENARIOS = ["arterial4x4", "cologne3", "grid4x4", "ingolstadt7"]


def summarize(net_file):
    root = ET.parse(net_file).getroot()
    tls = {}
    for tl in root.iter("tlLogic"):
        greens = 0
        for ph in tl.iter("phase"):
            state = ph.get("state") or ""
            if ("G" in state or "g" in state) and "y" not in state:
                greens += 1
        tls[tl.get("id")] = greens
    lanes = sum(1 for _ in root.iter("lane"))
    edges = sum(1 for _ in root.iter("edge"))
    version = root.get("version")
    return {"tls": tls, "lanes": lanes, "edges": edges, "version": version}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resco_src", required=True)
    ap.add_argument("--ours_root", default="sumo_rl/nets/RESCO")
    args = ap.parse_args()

    resco_root = os.path.join(args.resco_src, "resco_benchmark", "environments")

    any_semantic = False
    for scen in SCENARIOS:
        rf = os.path.join(resco_root, scen, f"{scen}.net.xml")
        of = os.path.join(args.ours_root, scen, f"{scen}.net.xml")
        if not (os.path.exists(rf) and os.path.exists(of)):
            print(f"{scen:14s} SKIP (missing file)")
            continue
        r, o = summarize(rf), summarize(of)

        same_ids = set(r["tls"]) == set(o["tls"])
        same_phases = r["tls"] == o["tls"]
        same_counts = (r["lanes"], r["edges"]) == (o["lanes"], o["edges"])
        semantic = not (same_ids and same_phases and same_counts)
        any_semantic = any_semantic or semantic

        print(f"=== {scen} === {'SEMANTIC DIFFERENCE' if semantic else 'equivalent'}")
        print(f"    netconvert version   RESCO={r['version']}  ours={o['version']}")
        print(f"    signalised junctions RESCO={len(r['tls'])}  ours={len(o['tls'])}"
              f"   ids match: {same_ids}")
        print(f"    lanes/edges          RESCO={r['lanes']}/{r['edges']}  "
              f"ours={o['lanes']}/{o['edges']}")
        if not same_ids:
            only_r = sorted(set(r["tls"]) - set(o["tls"]))
            only_o = sorted(set(o["tls"]) - set(r["tls"]))
            print(f"    only in RESCO ({len(only_r)}): {[i[:44] for i in only_r[:4]]}")
            print(f"    only in ours  ({len(only_o)}): {[i[:44] for i in only_o[:4]]}")
        elif not same_phases:
            diff = {k: (r["tls"][k], o["tls"][k])
                    for k in r["tls"] if r["tls"][k] != o["tls"][k]}
            print(f"    green-phase count differs for {len(diff)} signal(s) "
                  f"(RESCO, ours): {list(diff.items())[:4]}")
        print()

    print("SEMANTIC DIFFERENCES FOUND" if any_semantic
          else "all nets semantically equivalent (byte differences are re-encoding only)")


if __name__ == "__main__":
    main()
