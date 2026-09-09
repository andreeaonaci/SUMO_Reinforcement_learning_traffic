"""Extract RESCO's hand-authored MPLight/FRAP signal configuration into a form
this project's pipeline can consume.

Why this exists (fidings sec 101): MPLight/FRAP cannot be applied to an
intersection at all until a human authors that intersection's movement
configuration. RESCO's own docs say so:

    phase_pairs      "MPLight requires a defined set of traffic movements ...
                      consistent across traffic signals."
    pair_to_act_map  "actions which two signals have in common must be remapped
                      to correspond to the same action in each agent."

Those are written by hand, per scenario AND per signal -- cologne3's
pair_to_act_map is three hand-written dicts, one per intersection. This script
does NOT re-derive them (that would defeat the purpose of the comparison); it
copies RESCO's own authored values verbatim, so the FRAP arm gets the ORACLE
configuration it needs while this project's phase-relational arm gets none.

One adaptation is required and is stated plainly in the write-up: RESCO trains
MPLight one scenario at a time, and its phase_pairs list differs per scenario
(arterial4x4=5, grid4x4=8, cologne3=9, ingolstadt7=11), so a single shared FRAP
cannot span them as RESCO ships them. We build the UNION of the three TRAINING
cities' pairs and map every signal's local action index into that union. The
holdout is mapped into the same union without contributing to it -- sec 101
measured that all 8 of grid4x4's pairs are already present in the training
union, so this costs the FRAP arm nothing.

Usage:
    python diagnostics/build_frap_config.py --resco_src /path/to/RESCO \
        --out configs/resco_frap/phase_pairs.json
"""
import argparse
import json
import os

import yaml

# RESCO's canonical traffic-movement index table, from their
# docs/Environment Configuration.md. Read "S-W" as southbound incoming traffic
# turning onto the westbound outbound lanes.
DIRECTIONS = [
    "S-W", "S-S", "S-E",
    "W-N", "W-W", "W-S",
    "N-E", "N-N", "N-W",
    "E-S", "E-E", "E-N",
]

# RESCO scenario name -> this project's city directory name.
SCENARIO_TO_CITY = {
    "arterial4x4": "city_1",
    "cologne3": "city_4",
    "ingolstadt7": "city_6",
    "grid4x4": "city_5_holdout",
}
# Only the training cities define the shared union; the holdout is mapped into
# it but never extends it (otherwise the head would carry holdout-only rows).
TRAINING_SCENARIOS = ["arterial4x4", "cologne3", "ingolstadt7"]


def _signals(scenario_cfg):
    """The real signals in a RESCO scenario block (entries carrying lane_sets)."""
    return {
        k: v for k, v in scenario_cfg.items()
        if isinstance(v, dict) and "lane_sets" in v
    }


def _net_signal_lanes(net_file):
    """-> {tl_id: frozenset(incoming lane ids it controls)} from a .net.xml.

    Read off <connection ... tl="X" fromLane=...> entries, which is how the net
    records which lanes a traffic light governs.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(net_file).getroot()
    controlled = {}
    for conn in root.iter("connection"):
        tl = conn.get("tl")
        if not tl:
            continue
        frm, lane = conn.get("from"), conn.get("fromLane")
        if frm is None or lane is None:
            continue
        controlled.setdefault(tl, set()).add(f"{frm}_{lane}")
    return {k: frozenset(v) for k, v in controlled.items()}


def _green_phase_states(net_file):
    """-> {tl_id: [state strings of its green phases, in programme order]}."""
    import xml.etree.ElementTree as ET
    root = ET.parse(net_file).getroot()
    out = {}
    for tl in root.iter("tlLogic"):
        greens = []
        for ph in tl.iter("phase"):
            state = ph.get("state") or ""
            if ("G" in state or "g" in state) and "y" not in state:
                greens.append(state)
        out[tl.get("id")] = greens
    return out


def align_actions_by_phase_state(act_map, resco_greens, our_greens):
    """Re-index a signal's {local_act: union_pair} from RESCO's net to ours.

    sec 101: our vendored ingolstadt7 is missing a green phase that RESCO's net
    has at one intersection (RESCO 4 green phases, ours 3 -- the 25s
    'rrrrrrGGGGrr' is absent), so RESCO's action indices past that point are
    shifted relative to ours. Truncating by count would silently mis-map every
    later action, so phases are matched by their STATE STRING, which is exact.

    Returns (new_act_map, dropped) where dropped lists RESCO action indices
    whose phase does not exist in our net at all.
    """
    ours_by_state = {}
    for i, state in enumerate(our_greens):
        ours_by_state.setdefault(state, i)

    new_map, dropped = {}, []
    for resco_act, union_pair in sorted(act_map.items()):
        if resco_act >= len(resco_greens):
            dropped.append(resco_act)
            continue
        state = resco_greens[resco_act]
        if state in ours_by_state:
            new_map[ours_by_state[state]] = union_pair
        else:
            dropped.append(resco_act)
    return new_map, dropped


def remap_signal_ids(lane_sets, net_file):
    """RESCO signal id -> this project's signal id, matched by controlled lanes.

    sec 101: our vendored cologne3 and ingolstadt7 nets are topologically
    identical to RESCO's (same lane and edge counts, same signal counts) but
    five junctions were renamed at some point by a netedit re-save -- a 'GS_'
    prefix on cologne3, 'gneJ*' on ingolstadt7. Lane IDs are untouched, so the
    signals can be re-identified by the set of lanes each one controls rather
    than by name. Exact-name matches are kept as-is; only genuinely missing
    names are resolved by lane overlap, and only when the best match is
    unambiguous.
    """
    net_lanes = _net_signal_lanes(net_file)
    mapping, unresolved = {}, []
    used = set()
    for sid in lane_sets:
        if sid in net_lanes:
            mapping[sid] = sid
            used.add(sid)
    for sid, moves in lane_sets.items():
        if sid in mapping:
            continue
        want = {lid for lids in moves.values() for lid in lids}
        scored = []
        for cand, lanes in net_lanes.items():
            if cand in used:
                continue
            overlap = len(want & lanes)
            if overlap:
                scored.append((overlap, cand))
        scored.sort(reverse=True)
        # Unambiguous means: a best match exists and it strictly beats the next.
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            mapping[sid] = scored[0][1]
            used.add(scored[0][1])
        else:
            unresolved.append(sid)
    return mapping, unresolved


def build(resco_src, base_dir=None):
    signal_yaml = os.path.join(
        resco_src, "resco_benchmark", "config", "signal.yaml")
    cfg = yaml.safe_load(open(signal_yaml))

    # Unordered movement pairs, so ['W-W','E-E'] and ['E-E','W-W'] are one row.
    union = []
    for scen in TRAINING_SCENARIOS:
        for pair in cfg[scen]["phase_pairs"]:
            key = tuple(sorted(pair))
            if key not in union:
                union.append(key)
    union_index = {key: i for i, key in enumerate(union)}

    out = {
        "directions": DIRECTIONS,
        "union_pairs": [list(k) for k in union],
        "training_scenarios": TRAINING_SCENARIOS,
        "scenarios": {},
    }

    for scen, city in SCENARIO_TO_CITY.items():
        block = cfg[scen]
        sigs = _signals(block)
        pairs = [tuple(sorted(p)) for p in block["phase_pairs"]]
        pair_to_act = block.get("pair_to_act_map")

        act_to_union, unmapped = {}, []
        for sid in sigs:
            mapping = {}
            if pair_to_act and sid in pair_to_act:
                # RESCO authored this signal explicitly: {pair_index: local_act}
                for pair_i, local_act in pair_to_act[sid].items():
                    key = pairs[int(pair_i)]
                    if key in union_index:
                        mapping[int(local_act)] = union_index[key]
                    else:
                        unmapped.append((sid, list(key)))
            else:
                # pair_to_act_map null => homogeneous signals, local action i is
                # phase_pairs[i] (RESCO indexes actions by <tlLogic> order).
                for i, key in enumerate(pairs):
                    if key in union_index:
                        mapping[i] = union_index[key]
                    else:
                        unmapped.append((sid, list(key)))
            act_to_union[sid] = mapping

        lane_sets = {sid: sigs[sid]["lane_sets"] for sid in sigs}

        # Re-identify RESCO's signal names against our own net file where they
        # were renamed by a netedit re-save (sec 101).
        renamed, unresolved, phase_drops = {}, [], {}
        if base_dir:
            city_yaml = os.path.join(base_dir, city, "config.yaml")
            if os.path.exists(city_yaml):
                net_file = yaml.safe_load(open(city_yaml))["net_file"]
                id_map, unresolved = remap_signal_ids(lane_sets, net_file)
                renamed = {k: v for k, v in id_map.items() if k != v}
                act_to_union = {id_map.get(k, k): v
                                for k, v in act_to_union.items()}
                lane_sets = {id_map.get(k, k): v for k, v in lane_sets.items()}

                # Re-index actions by phase state string, since our vendored
                # nets can differ from RESCO's in which green phases exist.
                resco_net = os.path.join(
                    resco_src, "resco_benchmark", "environments", scen,
                    f"{scen}.net.xml")
                if os.path.exists(resco_net):
                    resco_greens = _green_phase_states(resco_net)
                    our_greens = _green_phase_states(net_file)
                    rev = {v: k for k, v in id_map.items()}
                    for sid in list(act_to_union):
                        rid = rev.get(sid, sid)
                        new_map, dropped = align_actions_by_phase_state(
                            act_to_union[sid],
                            resco_greens.get(rid, []),
                            our_greens.get(sid, []))
                        act_to_union[sid] = new_map
                        if dropped:
                            phase_drops[sid] = dropped

        out["scenarios"][city] = {
            "resco_name": scen,
            "n_signals": len(sigs),
            "act_to_union": act_to_union,
            "lane_sets": lane_sets,
            "pairs_outside_union": sorted({tuple(p) for _, p in unmapped}),
            "renamed_signals": renamed,
            "unresolved_signals": unresolved,
            "actions_dropped_vs_resco": phase_drops,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resco_src", required=True,
                    help="Path to a checkout of github.com/Pi-Star-Lab/RESCO")
    ap.add_argument("--out", default="configs/resco_frap/phase_pairs.json")
    ap.add_argument("--base_dir", default="environments_rescofull",
                    help="Roster whose net files RESCO's signal ids are "
                         "re-identified against (see remap_signal_ids).")
    args = ap.parse_args()

    out = build(args.resco_src, args.base_dir)

    print(f"UNION phase-pair table: {len(out['union_pairs'])} pairs "
          f"(from {', '.join(out['training_scenarios'])})")
    for i, pair in enumerate(out["union_pairs"]):
        print(f"  {i:2d}  {pair}")
    print()
    for city, blk in out["scenarios"].items():
        sizes = [len(m) for m in blk["act_to_union"].values()]
        note = ""
        if blk["pairs_outside_union"]:
            note = f"  !! {len(blk['pairs_outside_union'])} pair(s) outside union"
        if blk.get("unresolved_signals"):
            note += f"  !! {len(blk['unresolved_signals'])} signal(s) unresolved"
        if blk.get("actions_dropped_vs_resco"):
            note += (f"  !! {len(blk['actions_dropped_vs_resco'])} signal(s) "
                     "missing a green phase RESCO has")
        print(f"{city:16s} ({blk['resco_name']:12s}) signals={blk['n_signals']:3d}  "
              f"actions/signal min={min(sizes)} max={max(sizes)}{note}")
        for old, new in blk.get("renamed_signals", {}).items():
            print(f"      re-identified by controlled lanes: {old[:46]} -> {new}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
