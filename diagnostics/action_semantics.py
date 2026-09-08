"""Does action index k MEAN the same thing across intersections and cities?

The hypothesis this tests (fidings sec 95): the observation contract carries no
action semantics at all. ``TopKEncoder`` builds own_obs from congestion-sorted
lanes plus 5 global scalars, and ``action_mask`` marks which indices are VALID
without ever saying what any of them DOES. So the network must learn the
mapping "index k -> which movements get green" implicitly, from each city's
arbitrary phase ordering -- and that mapping cannot transfer to an unseen
intersection whose ordering is unrelated.

If true, it explains the project's central negative results at once: why more
training cities didn't help (sec 71 -- more conflicting index assignments, not
more transferable structure), why every capacity/architecture change was null
(no capacity recovers information absent from the input), why in-distribution
training works fine (sec 59 -- consistent semantics within a city), and most
sharply why a RANDOM init beat a federated-pretrained one after identical
fine-tuning (sec 70 -- pretrained weights carry confidently WRONG index
associations that must first be unlearned, which is worse than starting blank).

Method: parse each net.xml directly (no SUMO process needed). Rebuild the green
phase list exactly the way sumo_rl/environment/traffic_signal.py::_build_phases
does -- phases with no 'y' that aren't all-red -- so index k here is the same k
the agent acts on. Then characterise phase k by which MOVEMENT DIRECTIONS it
turns green, using each connection's linkIndex and dir attribute.

Usage:
    python diagnostics/action_semantics.py \
        sumo_rl/nets/RESCO/arterial4x4/arterial4x4.net.xml \
        sumo_rl/nets/RESCO/cologne3/cologne3.net.xml ...
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

DIR_NAMES = {"s": "straight", "l": "left", "r": "right", "t": "turn",
             "L": "partial-left", "R": "partial-right", "i": "internal"}


def green_phases_for(tl_elem):
    """sumo_rl's green-phase filter: no yellow, and not entirely red/stop."""
    out = []
    for ph in tl_elem.findall("phase"):
        state = ph.get("state", "")
        if "y" in state:
            continue
        if state.count("r") + state.count("s") == len(state):
            continue
        out.append(state)
    return out


def parse_net(path):
    """-> {tls_id: (green_phase_states, {linkIndex: dir})}"""
    root = ET.parse(path).getroot()
    link_dirs = defaultdict(dict)
    for conn in root.findall("connection"):
        tl = conn.get("tl")
        if tl is None:
            continue
        idx = conn.get("linkIndex")
        if idx is None:
            continue
        link_dirs[tl][int(idx)] = conn.get("dir", "?")
    out = {}
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        phases = green_phases_for(tl)
        if phases:
            out[tid] = (phases, link_dirs.get(tid, {}))
    return out


def phase_signature(state, dirs):
    """Which movement directions phase `state` turns green, as a Counter."""
    c = Counter()
    for i, ch in enumerate(state):
        if ch in "Gg":
            c[dirs.get(i, "?")] += 1
    return c


def dominant(c):
    if not c:
        return "none"
    total = sum(c.values())
    d, n = c.most_common(1)[0]
    return f"{DIR_NAMES.get(d, d)}({n}/{total})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--max_index", type=int, default=8)
    args = ap.parse_args()

    # index -> list of (city, tls, dominant-direction-label, raw counter)
    by_index = defaultdict(list)
    per_city_counts = {}

    for path in args.nets:
        city = os.path.basename(path).replace(".net.xml", "")
        try:
            tls = parse_net(path)
        except (ET.ParseError, OSError) as e:
            print(f"{city}: could not parse ({e})")
            continue
        n_phases = Counter()
        for tid, (phases, dirs) in sorted(tls.items()):
            n_phases[len(phases)] += 1
            for k, state in enumerate(phases):
                sig = phase_signature(state, dirs)
                by_index[k].append((city, tid, dominant(sig), sig))
        per_city_counts[city] = (len(tls), dict(sorted(n_phases.items())))

    print("=" * 78)
    print("PHASE-COUNT STRUCTURE (how wide each city's action space is)")
    print("=" * 78)
    for city, (n_tls, counts) in per_city_counts.items():
        print(f"  {city:14s} {n_tls:3d} signals   green-phase counts {counts}")

    print("\n" + "=" * 78)
    print("WHAT DOES ACTION INDEX k MEAN? (dominant movement direction greened)")
    print("=" * 78)
    inconsistent = 0
    for k in range(args.max_index):
        entries = by_index.get(k, [])
        if not entries:
            continue
        labels = Counter(lbl.split("(")[0] for _, _, lbl, _ in entries)
        # Cross-CITY view: what does index k dominantly mean in each city?
        per_city = defaultdict(Counter)
        for city, _, lbl, _ in entries:
            per_city[city][lbl.split("(")[0]] += 1
        print(f"\n  index {k}:  {len(entries)} intersections across "
              f"{len(per_city)} cities")
        print(f"    overall dominant-direction mix: {dict(labels)}")
        for city, c in sorted(per_city.items()):
            print(f"      {city:14s} {dict(c)}")
        if len(labels) > 1:
            inconsistent += 1

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    total = sum(1 for k in range(args.max_index) if by_index.get(k))
    print(f"  {inconsistent}/{total} action indices have MORE THAN ONE dominant meaning")
    print("  across the intersections that define them.")
    print()
    print("  A high number is the transfer barrier, quantified: the Q-head's row k")
    print("  is trained on conflicting targets across cities, and at an unseen")
    print("  intersection row k has no reliable meaning at all. A low number would")
    print("  falsify the hypothesis -- indices would already be de-facto aligned")
    print("  and something else explains the cross-topology gap.")


if __name__ == "__main__":
    sys.exit(main())
