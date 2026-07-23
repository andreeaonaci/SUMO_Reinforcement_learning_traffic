"""Rewrite a city's route file so all traffic actually falls inside the
simulation window, instead of silently departing after the sim has ended.

Why this exists
----------------
Some route files (RESCO benchmarks especially -- ingolstadt7, cologne3, ...)
use real-world clock timestamps for `depart`/`begin`/`end` (e.g. 57600s =
16:00:00). If a city's config runs from begin_time=0 to num_seconds=3600,
every vehicle scheduled at t=57600 never appears -- the sim ends over 5
hours before any traffic would depart. `measure_approach_volume.py` then
reports 0 veh-s on every single edge, which looks like a broken network but
is actually just a time-window mismatch.

What this script does
----------------------
For every city config under `--base` (default: environments/):
  1. Read net_file / route_file / begin_time / num_seconds from config.yaml.
  2. Parse the route file, find every <vehicle>, <trip>, and <flow> element.
  3. Remap their depart/begin/end timestamps into [0, num_seconds] using
     one of two modes:
       - "rescale" (default): linear min-max rescale of the ORIGINAL
         depart distribution into the new window. Preserves the shape of
         the demand curve (rush-hour peaks stay peaks) -- this is what you
         want almost all the time.
       - "random": uniform-random redistribution within the window,
         ignoring the original timing entirely. Use this when the original
         schedule isn't meaningful for your purposes (e.g. you just want
         *some* traffic to test the pipeline).
  4. Re-sort elements by their new depart/begin time (SUMO expects route
     files sorted by departure time; skipping this produces a runtime
     warning and SUMO may silently drop or reorder vehicles).
  5. Write the result to a new file (never overwrites the original).
  6. Optionally (--update-config) point the city's config.yaml at the new
     file, keeping a timestamped backup of the original config.

Usage
-----
    # Preview what would change, for every city, without writing anything
    python diagnostics/fix_route_windows.py --dry-run

    # Fix every city's route file (rescale mode), write new .rou.xml files
    python diagnostics/fix_route_windows.py

    # Fix one city only, and update its config.yaml to use the new file
    python diagnostics/fix_route_windows.py --only city_6 --update-config

    # Fully randomize departure times instead of rescaling
    python diagnostics/fix_route_windows.py --mode random --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import yaml

TIME_TAGS = ("vehicle", "trip", "flow")
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("xsi", XSI_NS)


@dataclass
class WindowStats:
    old_min: float
    old_max: float
    new_min: float
    new_max: float
    n_elements: int


def _time_key(el: ET.Element) -> Optional[float]:
    """Return the element's scheduling time (depart for vehicle/trip,
    begin for flow), or None if it has neither."""
    val = el.get("depart", el.get("begin"))
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _collect_timed_elements(root: ET.Element) -> List[ET.Element]:
    return [el for el in root.iter() if el.tag in TIME_TAGS and _time_key(el) is not None]


def _rescale(value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
    if old_max <= old_min:
        return new_min
    frac = (value - old_min) / (old_max - old_min)
    return new_min + frac * (new_max - new_min)


def remap_route_file(
    route_path: str,
    num_seconds: float,
    begin_time: float = 0.0,
    mode: str = "rescale",
    rng: Optional[random.Random] = None,
) -> Tuple[ET.ElementTree, WindowStats]:
    """Parse `route_path` and return a new ElementTree with every
    vehicle/trip/flow's timing remapped into [begin_time, begin_time + num_seconds].

    Does not write anything to disk -- see `write_route_file`.
    """
    tree = ET.parse(route_path)
    root = tree.getroot()

    timed = _collect_timed_elements(root)
    if not timed:
        raise ValueError(f"No <vehicle>/<trip>/<flow> elements with a depart/begin found in {route_path}")

    old_times = [_time_key(el) for el in timed]
    old_min, old_max = min(old_times), max(old_times)
    new_min, new_max = begin_time, begin_time + num_seconds

    rng = rng or random.Random()

    for el in timed:
        if el.tag == "flow":
            old_begin = float(el.get("begin", old_min))
            old_end = float(el.get("end", old_begin))
            duration = max(old_end - old_begin, 1.0)

            if mode == "random":
                new_begin = rng.uniform(new_min, max(new_min, new_max - duration))
            else:
                new_begin = _rescale(old_begin, old_min, old_max, new_min, new_max)

            new_end = min(new_begin + duration, new_max)
            if new_end <= new_begin:
                new_end = new_begin + 1.0
            el.set("begin", f"{new_begin:.2f}")
            el.set("end", f"{new_end:.2f}")
        else:
            old_depart = float(el.get("depart"))
            if mode == "random":
                new_depart = rng.uniform(new_min, new_max)
            else:
                new_depart = _rescale(old_depart, old_min, old_max, new_min, new_max)
            el.set("depart", f"{new_depart:.2f}")
            # 'arrival' (if present) referred to the ORIGINAL schedule and
            # is now stale/misleading -- drop it rather than leave a wrong
            # value sitting in the file.
            if "arrival" in el.attrib:
                del el.attrib["arrival"]

    # SUMO expects route files sorted by departure/begin time.
    vtypes = [c for c in root if c.tag == "vType"]
    timed_children = [c for c in root if c.tag in TIME_TAGS]
    other = [c for c in root if c not in vtypes and c not in timed_children]
    timed_children.sort(key=lambda e: _time_key(e) or 0.0)

    for c in list(root):
        root.remove(c)
    for c in vtypes + other + timed_children:
        root.append(c)

    stats = WindowStats(
        old_min=old_min, old_max=old_max,
        new_min=new_min, new_max=new_max,
        n_elements=len(timed),
    )
    return tree, stats


def write_route_file(tree: ET.ElementTree, out_path: str) -> None:
    tree.write(out_path, xml_declaration=True, encoding="UTF-8")


# ---------------------------------------------------------------------------
# Driver over environments/*/config.yaml
# ---------------------------------------------------------------------------

def process_city(
    city_dir: str,
    base_dir: str,
    mode: str,
    suffix: str,
    update_config: bool,
    dry_run: bool,
    rng: random.Random,
    force_num_seconds: Optional[float],
    force_begin: Optional[float],
) -> None:
    cfg_path = os.path.join(base_dir, city_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    route_file = cfg.get("route_file")
    if not route_file or not os.path.exists(route_file):
        print(f"[{city_dir}] SKIP -- route_file not found: {route_file}")
        return

    num_seconds = float(force_num_seconds if force_num_seconds is not None else cfg.get("num_seconds", 3600))
    begin_time = float(force_begin if force_begin is not None else cfg.get("begin_time", 0))

    try:
        new_tree, stats = remap_route_file(
            route_file, num_seconds=num_seconds, begin_time=begin_time, mode=mode, rng=rng
        )
    except ValueError as e:
        print(f"[{city_dir}] SKIP -- {e}")
        return

    already_ok = (
        stats.old_min >= begin_time - 1e-6
        and stats.old_max <= begin_time + num_seconds + 1e-6
    )
    status = "already inside window" if already_ok else "OUT OF WINDOW -- needs fix"

    print(
        f"[{city_dir}] {route_file}\n"
        f"    elements={stats.n_elements}  original window=[{stats.old_min:.1f}, {stats.old_max:.1f}]"
        f"  sim window=[{begin_time:.1f}, {begin_time + num_seconds:.1f}]  -> {status}"
    )

    if already_ok:
        return  # nothing to fix, don't create a redundant file

    root, ext = os.path.splitext(route_file)
    out_path = f"{root}{suffix}{ext}"

    if dry_run:
        print(f"    [dry-run] would write: {out_path}")
        if update_config:
            print(f"    [dry-run] would update config.yaml route_file -> {out_path}")
        return

    write_route_file(new_tree, out_path)
    print(f"    wrote: {out_path}")

    if update_config:
        backup_path = cfg_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(cfg_path, backup_path)
        cfg["route_file"] = out_path
        cfg.setdefault("begin_time", begin_time)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"    updated config.yaml (backup: {backup_path})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="environments", help="Directory containing city_*/config.yaml folders.")
    parser.add_argument("--only", nargs="*", default=None, help="Only process these city folder names.")
    parser.add_argument("--mode", choices=["rescale", "random"], default="rescale",
                        help="rescale: preserve original demand shape, shifted into window. "
                             "random: uniform-random departs within the window.")
    parser.add_argument("--suffix", default="_shifted", help="Suffix appended to the output route filename.")
    parser.add_argument("--update-config", action="store_true",
                        help="Point config.yaml's route_file at the new file (backs up the original config first).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --mode random.")
    parser.add_argument("--num-seconds", type=float, default=None,
                        help="Override num_seconds from every config (otherwise read per-city).")
    parser.add_argument("--begin", type=float, default=None,
                        help="Override begin_time from every config (otherwise read per-city, default 0).")
    args = parser.parse_args()

    if not os.path.isdir(args.base):
        print(f"Base directory not found: {args.base}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    city_dirs = sorted(os.listdir(args.base))
    if args.only:
        city_dirs = [c for c in city_dirs if c in args.only]

    for city_dir in city_dirs:
        full_path = os.path.join(args.base, city_dir)
        if not os.path.isdir(full_path):
            continue
        process_city(
            city_dir=city_dir,
            base_dir=args.base,
            mode=args.mode,
            suffix=args.suffix,
            update_config=args.update_config,
            dry_run=args.dry_run,
            rng=rng,
            force_num_seconds=args.num_seconds,
            force_begin=args.begin,
        )


if __name__ == "__main__":
    main()
