"""Track, per intersection, whether its Q-gap and action distribution are
still evolving or have plateaued -- the actual test for "stuck" vs.
"confidently correct" from a single snapshot.

Usage:
    python diagnostics/q_gap_trend.py results/run_2026_07_05-16_56_38/federated_history.json

Only needs numpy from stdlib+numpy; no project imports, so it also works
on a history file copied off the training machine.
"""
import argparse
import json
import sys


def fmt(x, width=8):
    return f"{x:.4f}".rjust(width) if isinstance(x, (int, float)) else str(x).rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_path")
    parser.add_argument("--ts", nargs="*", default=None,
                         help="Only show these ts_ids (default: all seen)")
    args = parser.parse_args()

    with open(args.history_path) as f:
        history = json.load(f)

    rounds = history.get("round", [])
    q_gaps_per_round = history.get("eval_q_gaps", [])
    action_counts_per_round = history.get("eval_action_counts", [])

    if not rounds:
        print("No rounds recorded in this history file.")
        sys.exit(1)

    # eval_q_gaps / eval_action_counts are lists of per-episode dicts (one
    # list entry per eval call); flatten to per-round dict of ts_id -> value
    # by averaging over episodes when there's more than one.
    def flatten(per_round_entry, key_is_gap: bool):
        if not per_round_entry:
            return {}
        if isinstance(per_round_entry, list) and per_round_entry and isinstance(per_round_entry[0], dict):
            # list of per-episode {ts_id: value} dicts -> average across episodes
            merged = {}
            for ep_dict in per_round_entry:
                for ts_id, v in ep_dict.items():
                    merged.setdefault(ts_id, []).append(v)
            return {ts_id: sum(vs) / len(vs) for ts_id, vs in merged.items()} if key_is_gap else merged
        return {}

    all_ts_ids = set()
    flat_gaps = []
    flat_counts = []
    for r, gaps, counts in zip(rounds, q_gaps_per_round, action_counts_per_round):
        fg = flatten(gaps, key_is_gap=True)
        fc_raw = flatten(counts, key_is_gap=False)
        # fc_raw: ts_id -> list of {action: count} dicts (one per episode) -> merge
        fc = {}
        for ts_id, dicts in fc_raw.items():
            merged = {}
            for d in dicts:
                for a, c in d.items():
                    merged[a] = merged.get(a, 0) + c
            fc[ts_id] = merged
        flat_gaps.append(fg)
        flat_counts.append(fc)
        all_ts_ids.update(fg.keys())
        all_ts_ids.update(fc.keys())

    ts_ids = sorted(args.ts) if args.ts else sorted(all_ts_ids, key=lambda x: (len(x), x))

    print(f"Rounds evaluated: {rounds}\n")
    for ts_id in ts_ids:
        gaps = [fg.get(ts_id) for fg in flat_gaps]
        counts = [fc.get(ts_id) for fc in flat_counts]

        print(f"ts_id={ts_id}")
        print(f"  Q-gap by round:    " + "  ".join(fmt(g) if g is not None else "   n/a" for g in gaps))

        # Fraction of ticks that picked the non-majority action, by round
        # -- a cleaner single number than the full {action: count} dict.
        fracs = []
        for c in counts:
            if not c:
                fracs.append(None)
                continue
            total = sum(c.values())
            majority = max(c.values())
            fracs.append((total - majority) / total if total else 0.0)
        print(f"  minority-action %: " + "  ".join(
            (f"{f*100:6.1f}%" if f is not None else "   n/a") for f in fracs
        ))

        # Simple trend read: is the Q-gap growing (more confident) or
        # shrinking (still moving) over the recorded rounds?
        valid_gaps = [g for g in gaps if g is not None]
        if len(valid_gaps) >= 2:
            delta = valid_gaps[-1] - valid_gaps[0]
            trend = "growing (more confident)" if delta > 0.01 else (
                "shrinking (still changing)" if delta < -0.01 else "flat"
            )
            print(f"  trend: {trend} (Δ={delta:+.4f} over {len(valid_gaps)} evals)")
        print()


if __name__ == "__main__":
    main()
