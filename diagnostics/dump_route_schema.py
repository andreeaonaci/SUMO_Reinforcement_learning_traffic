"""Dump the tag names, attribute keys, and a few sample elements from a
SUMO route file, so we can see its actual schema instead of guessing.

Usage:
    python diagnostics/dump_route_schema.py environments/city_5_holdout/config.yaml
    # or point it straight at a .rou.xml:
    python diagnostics/dump_route_schema.py path/to/routes.rou.xml
"""
import argparse
import os
from collections import defaultdict
from xml.etree import ElementTree as ET

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="config.yaml OR a .rou.xml file directly")
    parser.add_argument("--samples-per-tag", type=int, default=3)
    args = parser.parse_args()

    if args.path.endswith(".yaml") or args.path.endswith(".yml"):
        with open(args.path) as f:
            cfg = yaml.safe_load(f)
        route_file = cfg["route_file"]
        if not os.path.isabs(route_file):
            # route_file paths in config.yaml are often relative to the
            # config's own directory
            route_file = os.path.join(route_file)
    else:
        route_file = args.path

    print(f"Route file: {route_file}\n")

    tree = ET.parse(route_file)
    root = tree.getroot()
    print(f"Root tag: <{root.tag}> attrs={root.attrib}\n")

    # Check for <include> — routes split across multiple files is common
    includes = root.findall("include") + list(root.iter("include"))
    if includes:
        print(f"!! Found <include> element(s) -- routes may live in other files:")
        for inc in includes:
            print(f"   href={inc.get('href')}")
        print()

    tag_attr_keys = defaultdict(set)
    tag_samples = defaultdict(list)
    tag_counts = defaultdict(int)

    for el in root.iter():
        tag_counts[el.tag] += 1
        tag_attr_keys[el.tag].update(el.attrib.keys())
        if len(tag_samples[el.tag]) < args.samples_per_tag:
            tag_samples[el.tag].append(dict(el.attrib))

    print(f"{'tag':<20} {'count':<8} attribute keys seen")
    print("-" * 80)
    for tag in sorted(tag_counts, key=lambda t: -tag_counts[t]):
        keys = ", ".join(sorted(tag_attr_keys[tag]))
        print(f"{tag:<20} {tag_counts[tag]:<8} {keys}")

    print("\nSample elements per tag:")
    for tag in ("flow", "trip", "vehicle", "route", "routeDistribution"):
        if tag in tag_samples:
            print(f"\n  <{tag}> samples:")
            for s in tag_samples[tag]:
                print(f"    {s}")


if __name__ == "__main__":
    main()
