"""Generate synthetic SUMO grid cities with IRREGULAR topology, to widen this
project's training roster beyond the 7 vendored RESCO nets.

Motivation (2026-09-01 session, after sec 70): every roster this project has
trained on is small and topologically narrow, and sec 70's random-init control
suggests the federated pre-training isn't currently transferring anything
useful. One plausible reason is that the model never sees enough *variety* of
intersection shapes during training to learn anything topology-general -- it
only meets a genuinely different topology at holdout-eval time. This script
produces many cities cheaply, with a controllable spread of intersection
difficulty, so that hypothesis can actually be tested.

How the irregularity works: `netgenerate --grid` produces a perfect NxM lattice
where every interior intersection is an identical 4-way. That is exactly the
monoculture we do NOT want. So after generating the plain-XML form, this script
randomly deletes a fraction (--drop_fraction) of the interior traffic-light
nodes and every edge touching them, then rebuilds with netconvert. The
survivors around each hole become 3-way T-junctions and dead-ends, so a single
generated city contains a genuine mix of 2-, 3-, and 4-way intersections --
i.e. a mix of "simple" and "tough" ones, with different phase counts and hence
different action_dim, which is precisely the axis `ActionMaskPadder` and the
masked-head aggregation exist to handle.

netconvert is run with `--keep-edges.components 1` so that a deletion which
splits the lattice yields the largest connected component rather than a broken
net that route generation would fail on.

Connections and traffic-light programs are deliberately NOT carried over from
netgenerate's .con.xml/.tll.xml -- those reference deleted nodes. netconvert
re-guesses both from the surviving geometry (`--tls.guess`), which is also what
gives the reduced-degree junctions sensible phase counts.

Outputs, per generated city:
  <out_root>/<city_name>/<city_name>.net.xml       the network
  <out_root>/<city_name>/<city_name>.rou.xml       randomTrips.py traffic
  <roster_dir>/<city_name>/config.yaml             config in this project's format

The roster_dir is a NEW directory (default `environments_grid/`), never
`environments/` -- appending cities to `environments/` would silently change the
default 7-city roster every existing experiment uses.

Usage:
    # 6 cities: 3x3/4x4/5x5, each perfect and 20%-dropped
    python diagnostics/generate_grid_cities.py

    # custom sweep
    python diagnostics/generate_grid_cities.py \\
        --specs 4x4:0.0,4x4:0.2,5x5:0.2,6x6:0.3 --lanes 2 --seed_base 700000

Validate what was produced (this project's standard check):
    python experiments/validate_sumo_cities.py --base_dir environments_grid
"""
import argparse
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

DEFAULT_SPECS = "3x3:0.0,3x3:0.2,4x4:0.0,4x4:0.2,5x5:0.2,5x5:0.3"


def _run(cmd, what):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{what} failed: {' '.join(cmd[:3])}...")
    return result


def generate_net(city_dir, city_name, nx, ny, drop_fraction, lanes, length, seed,
                 tls_layout="incoming"):
    """netgenerate a perfect grid, delete a random subset of its interior
    traffic-light nodes (plus every edge touching them), rebuild with
    netconvert. Returns (net_path, n_dropped, n_tls_remaining)."""
    os.makedirs(city_dir, exist_ok=True)
    prefix = os.path.join(city_dir, "_plain")
    net_path = os.path.join(city_dir, f"{city_name}.net.xml")

    _run([
        "netgenerate", "--grid",
        "--grid.x-number", str(nx), "--grid.y-number", str(ny),
        "--grid.length", str(length), "--grid.attach-length", str(length / 2),
        "--default.lanenumber", str(lanes),
        "--tls.guess", "true", "--tls.layout", tls_layout,
        "--plain-output-prefix", prefix,
        "--output-file", os.path.join(city_dir, "_full.net.xml"),
    ], "netgenerate")

    nod_path, edg_path = prefix + ".nod.xml", prefix + ".edg.xml"
    nod_tree = ET.parse(nod_path)
    edg_tree = ET.parse(edg_path)

    # Only interior (traffic-light) nodes are candidates -- dropping the
    # fringe attach-nodes would just shrink the map, not add shape variety.
    tls_nodes = [n.get("id") for n in nod_tree.getroot().findall("node")
                 if n.get("type") == "traffic_light"]
    rng = random.Random(seed)
    n_drop = int(round(len(tls_nodes) * drop_fraction))
    dropped = set(rng.sample(sorted(tls_nodes), n_drop)) if n_drop else set()

    if dropped:
        nod_root = nod_tree.getroot()
        for node in list(nod_root.findall("node")):
            if node.get("id") in dropped:
                nod_root.remove(node)
        edg_root = edg_tree.getroot()
        for edge in list(edg_root.findall("edge")):
            if edge.get("from") in dropped or edge.get("to") in dropped:
                edg_root.remove(edge)
        nod_tree.write(nod_path)
        edg_tree.write(edg_path)

    _run([
        "netconvert",
        "--node-files", nod_path, "--edge-files", edg_path,
        "--output-file", net_path,
        "--tls.guess", "true", "--tls.layout", tls_layout,
        # A deletion can split the lattice; keep the largest component so the
        # result is always a single routable network.
        "--keep-edges.components", "1",
        "--remove-edges.isolated", "true",
        "--no-turnarounds", "true",
    ], "netconvert")

    net_root = ET.parse(net_path).getroot()
    n_tls = len([j for j in net_root.findall("junction")
                 if j.get("type") == "traffic_light"])

    for scratch in (prefix + ".nod.xml", prefix + ".edg.xml", prefix + ".con.xml",
                    prefix + ".tll.xml", prefix + ".typ.xml",
                    os.path.join(city_dir, "_full.net.xml")):
        if os.path.exists(scratch):
            os.remove(scratch)

    return net_path, len(dropped), n_tls


def generate_routes(city_dir, city_name, net_path, duration, insertion_rate, seed):
    route_path = os.path.join(city_dir, f"{city_name}.rou.xml")
    trip_path = os.path.join(city_dir, f"{city_name}.trips.xml")
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set -- see CLAUDE.md's Setup section.")
    _run([
        sys.executable, os.path.join(sumo_home, "tools", "randomTrips.py"),
        "-n", net_path, "-o", trip_path, "-r", route_path,
        "-b", "0", "-e", str(duration),
        "--insertion-rate", str(insertion_rate),
        "--seed", str(seed), "--validate", "--fringe-factor", "5",
    ], "randomTrips.py")
    if os.path.exists(trip_path):
        os.remove(trip_path)
    return route_path


def write_config(roster_dir, city_name, net_path, route_path, duration):
    """Config in this project's format. max_lanes/k_max/max_hops deliberately
    match environments/city_*/config.yaml exactly -- resolve_city_configs_and_dims
    requires every city in a roster to agree on own_dim/neighbor_dim/k_max, and
    those are what determine them."""
    cfg_dir = os.path.join(roster_dir, city_name)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(
            f"name: {city_name}\n"
            f"net_file: {net_path}\n"
            f"route_file: {route_path}\n"
            f"\n"
            f"delta_time: 5\n"
            f"num_seconds: {duration}\n"
            f"sumo_seed: 42\n"
            f"\n"
            f"max_lanes: 16\n"
            f"max_queue: 50.0\n"
            f"max_wait: 300.0\n"
            f"max_speed: 50.0\n"
            f"\n"
            f"k_max: 8\n"
            f"max_hops: 3\n"
            f"\n"
            f"use_libsumo: true\n"
            f"begin_time: 0.0\n"
        )
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=DEFAULT_SPECS,
                     help="Comma-separated NxM:drop_fraction, e.g. '4x4:0.0,5x5:0.2'.")
    ap.add_argument("--out_root", default=os.path.join("sumo_rl", "nets", "generated"))
    ap.add_argument("--roster_dir", default="environments_grid",
                     help="NEW roster dir for the configs. Never point this at "
                          "'environments/' -- that would silently change the default "
                          "7-city roster every existing experiment uses.")
    ap.add_argument("--lanes", type=int, default=2,
                     help="Lanes per direction. Must be >= 2: with 1 lane netconvert "
                          "declines to signalize the junctions at all (measured -- zero "
                          "tlLogic elements in the output).")
    ap.add_argument("--tls_layout", default="incoming", choices=["incoming", "opposites"],
                     help="'incoming' gives each approach its own green phase, so phase "
                          "count (and hence action_dim) FOLLOWS junction degree -- a 3-way "
                          "gets 3 actions, a 4-way gets 4. That topology-driven action_dim "
                          "spread is the whole point of this script; 'opposites' (SUMO's "
                          "default) collapses every junction to 2 actions regardless of "
                          "shape, which is the monoculture we're trying to get away from.")
    ap.add_argument("--length", type=float, default=200.0, help="Metres between intersections.")
    ap.add_argument("--duration", type=int, default=3600)
    ap.add_argument("--insertion_rate", type=float, default=1200.0, help="Vehicles/hour.")
    ap.add_argument("--seed_base", type=int, default=700000,
                     help="City i uses seed_base+i for both node-dropping and traffic. "
                          "Kept clear of this project's other seed ranges (training 3-17, "
                          "eval_seed_base 12345, generate_random_routes 900000).")
    ap.add_argument("--force", action="store_true", help="Regenerate even if a city already exists.")
    args = ap.parse_args()

    specs = []
    for raw in args.specs.split(","):
        grid, _, drop = raw.strip().partition(":")
        nx, _, ny = grid.partition("x")
        specs.append((int(nx), int(ny), float(drop or 0.0)))

    print(f"Generating {len(specs)} cities into {args.out_root} "
          f"(configs -> {args.roster_dir}/)\n")
    summary = []
    for i, (nx, ny, drop) in enumerate(specs):
        tag = f"{nx}x{ny}" + (f"_drop{int(drop*100)}" if drop else "_full")
        city_name = f"grid_{tag}"
        # Distinct suffix if the same spec appears twice in one --specs list.
        if any(s[0] == city_name for s in summary):
            city_name = f"grid_{tag}_{i}"
        city_dir = os.path.join(args.out_root, city_name)
        seed = args.seed_base + i

        if os.path.exists(os.path.join(city_dir, f"{city_name}.rou.xml")) and not args.force:
            print(f"[skip] {city_name} already exists (use --force to regenerate)")
            continue

        print(f"[{i+1}/{len(specs)}] {city_name}: {nx}x{ny} grid, drop={drop:.0%}, seed={seed}")
        net_path, n_dropped, n_tls = generate_net(
            city_dir, city_name, nx, ny, drop, args.lanes, args.length, seed,
            tls_layout=args.tls_layout)
        route_path = generate_routes(
            city_dir, city_name, net_path, args.duration, args.insertion_rate, seed)
        write_config(args.roster_dir, city_name, net_path, route_path, args.duration)
        print(f"        -> {n_tls} traffic-light intersections ({n_dropped} node(s) dropped)")
        summary.append((city_name, n_tls, n_dropped))

    if summary:
        print(f"\n{len(summary)} cities generated. Validate with:")
        print(f"    python experiments/validate_sumo_cities.py --base_dir {args.roster_dir}")


if __name__ == "__main__":
    main()
