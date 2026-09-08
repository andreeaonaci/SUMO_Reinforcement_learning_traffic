"""Scale a SUMO route file's demand up or down, preserving its trip pattern.

Why this exists: comparing controllers across scenarios is only meaningful if
demand INTENSITY is controlled. Measured per-signal demand over a 3600s episode
in this project's own scenarios:

    grid4x4_1            1473 veh / 16 signals =  92 veh/signal   (moderate)
    grid4x4_dense        1200 veh / 16 signals =  75 veh/signal   (LIGHTER --
                         despite the name; it is a randomTrips file with 60
                         flows at period=60 over begin=0,end=1200)
    3x3grid routes14000  7000 veh /  9 signals = 778 veh/signal   (saturated;
                         max_pressure itself scores -11273 there)

Picking a holdout without checking this produces either a floor effect (every
controller gridlocks) or a ceiling effect (every controller is free-flow), and
in both cases the comparison measures nothing.

  --factor <1  subsample vehicles uniformly, keeping the departure profile
  --factor >1  duplicate vehicles, offsetting departures within the headway so
               duplicates do not stack on one instant

Usage:
    python diagnostics/scale_demand.py IN.rou.xml OUT.rou.xml --factor 0.12
"""
import argparse
import copy
import xml.etree.ElementTree as ET


def scale(in_path: str, out_path: str, factor: float, seed: int = 0) -> tuple:
    tree = ET.parse(in_path)
    root = tree.getroot()
    vehicles = [c for c in root if c.tag == "vehicle"]
    if not vehicles:
        raise SystemExit(
            f"{in_path} has no <vehicle> elements (flow-based files are not supported "
            "-- scale their period/vehsPerHour directly instead)."
        )

    def depart(v):
        try:
            return float(v.get("depart", "0"))
        except ValueError:
            return 0.0

    vehicles.sort(key=depart)
    for v in vehicles:
        root.remove(v)

    kept = []
    if factor <= 1.0:
        # Uniform stride keeps the temporal profile rather than truncating it.
        step = 1.0 / max(factor, 1e-9)
        i = 0.0
        while int(i) < len(vehicles):
            kept.append(copy.deepcopy(vehicles[int(i)]))
            i += step
    else:
        reps = int(factor)
        frac = factor - reps
        for idx, v in enumerate(vehicles):
            n = reps + (1 if (idx * frac) % 1.0 < frac else 0)
            # Spread duplicates across the gap to the next departure so they do
            # not all appear at the same simulation second.
            nxt = depart(vehicles[idx + 1]) if idx + 1 < len(vehicles) else depart(v) + 1.0
            gap = max(nxt - depart(v), 0.1)
            for k in range(n):
                c = copy.deepcopy(v)
                c.set("id", f"{v.get('id')}_{k}" if k else v.get("id"))
                c.set("depart", f"{depart(v) + gap * k / max(n, 1):.2f}")
                kept.append(c)

    kept.sort(key=depart)
    for v in kept:
        root.append(v)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return len(vehicles), len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--factor", type=float, required=True)
    args = ap.parse_args()
    before, after = scale(args.input, args.output, args.factor)
    print(f"{args.input}: {before} vehicles -> {args.output}: {after} "
          f"(factor {args.factor}, actual {after / max(before,1):.3f})")


if __name__ == "__main__":
    main()
