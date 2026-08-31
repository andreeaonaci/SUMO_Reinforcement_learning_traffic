"""Generate N independent randomized-traffic route files for a SUMO net,
via SUMO's own randomTrips.py tool (the same tool that generated this
project's own grid4x4_dense.rou.xml, per its header comment -- not a new
traffic-generation mechanism).

Built for the holdout fine-tune experiment (diagnostics/finetune_on_holdout.py,
fidings/divergence_investigation.md sec 65's "prio next step #2"): fine-tuning
on the true holdout city's topology must never touch the REAL evaluation
route file (grid4x4_1.rou.xml, what every eval_reward number in this project
is measured against) -- these generated files are a synthetic substitute
traffic pattern on the same net, used only for the fine-tuning phase.

Each variant gets its own --seed (seed_base + i) so the N route files are
reproducible but genuinely different traffic instances of the same topology
-- fine-tuning federates across all of them so the model sees varied random
traffic, not one fixed synthetic pattern.

Usage:
    python diagnostics/generate_random_routes.py \\
        --net sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml \\
        --out_dir sumo_rl/nets/RESCO/grid4x4/generated_random \\
        --n_variants 5 --duration 3600 --insertion_rate 1470
"""
import argparse
import os
import subprocess
import sys


def generate_variants(
    net_file: str,
    out_dir: str,
    n_variants: int = 5,
    duration: int = 3600,
    insertion_rate: float = 1470.0,
    seed_base: int = 900000,
    force: bool = False,
) -> list:
    """Returns the list of generated (or already-existing) route file paths."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set -- see CLAUDE.md's Setup section.")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(random_trips):
        raise RuntimeError(f"randomTrips.py not found at {random_trips}")

    os.makedirs(out_dir, exist_ok=True)
    net_basename = os.path.basename(net_file)
    for suffix in (".net.xml", ".xml"):
        if net_basename.endswith(suffix):
            net_basename = net_basename[: -len(suffix)]
            break

    route_paths = []
    for i in range(n_variants):
        route_path = os.path.join(out_dir, f"{net_basename}_random_{i}.rou.xml")
        trip_path = os.path.join(out_dir, f"{net_basename}_random_{i}.trips.xml")
        route_paths.append(route_path)

        if os.path.exists(route_path) and not force:
            print(f"[skip] {route_path} already exists (use --force to regenerate)")
            continue

        seed = seed_base + i
        cmd = [
            sys.executable, random_trips,
            "-n", net_file,
            "-o", trip_path,
            "-r", route_path,
            "-b", "0",
            "-e", str(duration),
            "--insertion-rate", str(insertion_rate),
            "--seed", str(seed),
            "--validate",
            "--random",
            "--fringe-factor", "5",
        ]
        print(f"[generate] variant {i} (seed={seed}) -> {route_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"randomTrips.py failed for variant {i} (seed={seed})")

    return route_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml")
    ap.add_argument("--out_dir", default="sumo_rl/nets/RESCO/grid4x4/generated_random")
    ap.add_argument("--n_variants", type=int, default=5)
    ap.add_argument("--duration", type=int, default=3600,
                     help="Simulation seconds to spread insertions over -- match the "
                          "target city's num_seconds (city_5_holdout uses 3600).")
    ap.add_argument("--insertion_rate", type=float, default=1470.0,
                     help="Vehicles/hour. Default matches grid4x4_1.rou.xml's own "
                          "density (1473 vehicles over 3600s, measured directly).")
    ap.add_argument("--seed_base", type=int, default=900000,
                     help="Variant i uses seed_base+i. Kept well clear of this "
                          "project's other seed ranges (training seeds 3-5, "
                          "eval_seed_base 12345) to avoid any accidental overlap.")
    ap.add_argument("--force", action="store_true", help="Regenerate even if a variant already exists.")
    args = ap.parse_args()

    paths = generate_variants(
        net_file=args.net, out_dir=args.out_dir, n_variants=args.n_variants,
        duration=args.duration, insertion_rate=args.insertion_rate,
        seed_base=args.seed_base, force=args.force,
    )
    print(f"\n{len(paths)} route file(s) ready under {args.out_dir}:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
