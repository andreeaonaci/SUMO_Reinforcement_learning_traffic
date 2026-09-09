"""Validate the extracted RESCO FRAP config against this project's own net files.

The config in configs/resco_frap/phase_pairs.json is RESCO's hand-authored
signal metadata (sec 101). It is only usable here if RESCO's signal IDs and lane
IDs are the same ones our SUMO nets expose -- they should be, since both read
the same .net.xml, but "should be" is exactly the assumption that produced sec
25, sec 95b and sec 99. Check it before spending any compute.

Checks, per city:
  1. every signal ID in the config exists as a <tlLogic> in our net file
  2. every signal in our net file is covered by the config
  3. every lane ID referenced by lane_sets exists as a <lane> in our net file
  4. each signal's mapped action count vs. its real green-phase count

Usage:
    python diagnostics/check_frap_config.py --base_dir environments_rescofull
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def net_signals_and_lanes(net_file):
    """-> ({tl_id: n_green_phases}, {lane_id})."""
    root = ET.parse(net_file).getroot()
    lanes = {ln.get("id") for ln in root.iter("lane") if ln.get("id")}
    signals = {}
    for tl in root.iter("tlLogic"):
        greens = 0
        for ph in tl.iter("phase"):
            state = ph.get("state") or ""
            # sumo_rl's convention: a phase with any G/g and no yellow is a
            # controllable green phase.
            if ("G" in state or "g" in state) and "y" not in state:
                greens += 1
        signals[tl.get("id")] = greens
    return signals, lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="environments_rescofull")
    ap.add_argument("--config", default="configs/resco_frap/phase_pairs.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    print(f"union phase-pair table: {len(cfg['union_pairs'])} pairs\n")

    ok = True
    for city, blk in cfg["scenarios"].items():
        city_yaml = os.path.join(args.base_dir, city, "config.yaml")
        if not os.path.exists(city_yaml):
            print(f"{city:16s} SKIP (not in {args.base_dir})")
            continue
        ycfg = yaml.safe_load(open(city_yaml))
        net_file = ycfg["net_file"]
        net_sigs, net_lanes = net_signals_and_lanes(net_file)

        cfg_sigs = set(blk["act_to_union"])
        missing = sorted(cfg_sigs - set(net_sigs))
        extra = sorted(set(net_sigs) - cfg_sigs)

        bad_lanes = []
        for sid, lane_sets in blk["lane_sets"].items():
            for move, lids in lane_sets.items():
                for lid in lids:
                    if lid not in net_lanes:
                        bad_lanes.append((sid, move, lid))

        act_mismatch = []
        for sid, mapping in blk["act_to_union"].items():
            if sid in net_sigs and len(mapping) != net_sigs[sid]:
                act_mismatch.append((sid, len(mapping), net_sigs[sid]))

        status = "OK" if not (missing or extra or bad_lanes or act_mismatch) else "PROBLEM"
        if status != "OK":
            ok = False
        print(f"{city:16s} ({blk['resco_name']:12s}) net={os.path.basename(net_file):28s} {status}")
        print(f"    signals: config={len(cfg_sigs)} net={len(net_sigs)}")
        if missing:
            print(f"    !! {len(missing)} config signal(s) NOT in net: {missing[:4]}")
        if extra:
            print(f"    !! {len(extra)} net signal(s) NOT in config: {extra[:4]}")
        if bad_lanes:
            print(f"    !! {len(bad_lanes)} lane id(s) in lane_sets NOT in net: {bad_lanes[:3]}")
        if act_mismatch:
            print(f"    !! mapped-action count != net green-phase count "
                  f"for {len(act_mismatch)} signal(s): {act_mismatch[:4]}")
        print()

    print("ALL CHECKS PASSED" if ok else "FAILURES ABOVE -- do not run FRAP on this config yet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
