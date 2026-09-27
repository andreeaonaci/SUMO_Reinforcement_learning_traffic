"""Build Braun-runnable SUMO scenarios from this project's RESCO benchmark files.

Run INSIDE Braun's venv (baselines/braun/bpy.sh), because it imports his code
(pinned snapshot ea47985, external path, never copied into this repo):

    baselines/braun/bpy.sh <repo>/baselines/braun/build_resco_scenarios.py

For every network of environments_rescofull it writes, under
baselines/braun/scenarios/<net>/:

  synth.tll.xml    Braun's own phase synthesis (scripts/build_network.py::_build_tll,
                   called verbatim) -> all maximal cliques of compatible atomic
                   movement groups, one static tlLogic per junction, programID
                   'movement_safe'.  ONE adaptation, applied after his function
                   returns: his builder names each tlLogic after the JUNCTION id
                   (true for his OSM builds, where netconvert gives tls id ==
                   node id).  RESCO's cologne3 / ingolstadt7 use TLS ids that
                   differ from the junction id (e.g. GS_cluster_..., gneJ143), so
                   the tlLogic id is remapped junction-id -> TLS-id.  Nothing
                   about the phases themselves is changed.
  synth.sumocfg    RESCO net + RESCO route file + synth.tll.xml  (arm "as published")
  native.sumocfg   RESCO net + RESCO route file, RESCO's OWN signal program
                   (arm B: Braun restricted to the existing green phases -- his
                   runtime extracts selectable phases from whatever program is
                   active, so no code change is needed for this arm)

Begin/end times are the evaluation windows of environments_rescofull/*/config.yaml
(read, not retyped), so both arms see exactly our benchmark windows.

It also writes results/braun/feasibility.json with, per network and TLS:
existing green-phase count, synthesized phase count, skip reason (if any),
and -- after actually starting SUMO through Braun's runtime -- the number of
junctions his runtime controls and whether his movement graph builds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

BRAUN_ROOT = Path('/home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAUN_ROOT))

import sumolib  # noqa: E402

from src.movement.extraction import is_selectable_green_state  # noqa: E402
from src.movement.graph import build_movement_graph  # noqa: E402
from src.movement.runtime import MovementControlRuntime  # noqa: E402
from src.movement.sumo_backend import SumoBackendKind  # noqa: E402


def _load_braun_build_network():
    spec = importlib.util.spec_from_file_location('braun_build_network', BRAUN_ROOT / 'scripts' / 'build_network.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['braun_build_network'] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


ROSTER = {
    # net key : environments_rescofull city dir
    'arterial4x4': 'city_1',
    'cologne3': 'city_4',
    'grid4x4': 'city_5_holdout',
    'ingolstadt7': 'city_6',
}
OUT_ROOT = REPO / 'baselines' / 'braun' / 'scenarios'


def _sumocfg(net: Path, rou: Path, begin: int, end: int, additional: Path | None) -> str:
    add = f'\n        <additional-files value="{additional}"/>' if additional is not None else ''
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{rou}"/>{add}
    </input>
    <time>
        <begin value="{begin}"/>
        <end value="{end}"/>
        <step-length value="1.0"/>
    </time>
    <report>
        <no-step-log value="true"/>
    </report>
</configuration>
"""


def _node_to_tls(net) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tls in net.getTrafficLights():
        nodes = {conn[0].getEdge().getToNode().getID() for conn in tls.getConnections()}
        if len(nodes) != 1:
            raise RuntimeError(f'TLS {tls.getID()} controls {len(nodes)} junctions; not supported by this adapter')
        mapping[nodes.pop()] = tls.getID()
    return mapping


def _existing_green_counts(net_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, elem in ET.iterparse(str(net_path), events=('end',)):
        if elem.tag == 'tlLogic':
            states = [p.get('state') for p in elem.findall('phase')]
            counts[elem.get('id')] = len({s for s in states if is_selectable_green_state(s)})
    return counts


def _runtime_check(cfg: Path, begin: int) -> dict:
    runtime = MovementControlRuntime(cfg_path=cfg, seed=42, backend_kind=SumoBackendKind.LIBSUMO)
    runtime.start()
    try:
        tl_api = runtime._backend.trafficlight
        active = {tls: tl_api.getProgram(tls) for tls in tl_api.getIDList()}
        t0 = runtime.simulation_api.getTime()
        graph = build_movement_graph(runtime.programs, net_path=_net_of(cfg))
        return {
            'sim_start_time': t0,
            'controlled_tls': sorted(runtime.programs),
            'n_controlled': len(runtime.programs),
            'active_program_ids': active,
            'phases_per_tls': {k: len(v.selectable_phases) for k, v in runtime.programs.items()},
            'movements_per_tls': {k: len(v.movements) for k, v in runtime.programs.items()},
            'graph_lane_groups': len(graph.lane_groups),
            'graph_movement_nodes': len(graph.movements),
            'graph_pass_through_tls': list(graph.pass_through_traffic_light_ids),
        }
    finally:
        runtime.close()


def _net_of(cfg: Path) -> Path:
    return Path(ET.parse(cfg).getroot().find('./input/net-file').attrib['value'])


def main() -> None:
    bn = _load_braun_build_network()
    report: dict = {'braun_commit': 'ea47985ccba2bbca273eb08139645399cf53ef23', 'networks': {}}
    for key, city in ROSTER.items():
        cfg = yaml.safe_load((REPO / 'environments_rescofull' / city / 'config.yaml').read_text())
        net_path = (REPO / cfg['net_file']).resolve()
        rou_path = (REPO / cfg['route_file']).resolve()
        begin = int(float(cfg.get('begin_time', 0) or 0))
        end = begin + int(cfg['num_seconds'])
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)

        net = sumolib.net.readNet(str(net_path), withConnections=True, withFoes=True)
        raw_tll = out_dir / 'synth.raw_nodeids.tll.xml'
        print(f'\n=== {key} ({city}) window [{begin}, {end})')
        bn._build_tll(net, net_path, raw_tll)  # Braun's function, verbatim

        node_to_tls = _node_to_tls(net)
        tree = ET.parse(raw_tll)
        remapped = {}
        for logic in tree.getroot().findall('tlLogic'):
            node_id = logic.get('id')
            tls_id = node_to_tls[node_id]
            if tls_id != node_id:
                remapped[node_id] = tls_id
            logic.set('id', tls_id)
        synth_tll = out_dir / 'synth.tll.xml'
        tree.write(synth_tll, encoding='utf-8', xml_declaration=True)
        raw_tll.unlink()

        synth_counts = {
            logic.get('id'): len(logic.findall('phase')) // 3  # green, yellow, all-red per phase
            for logic in ET.parse(synth_tll).getroot().findall('tlLogic')
        }
        existing = _existing_green_counts(net_path)

        # synthfb: Braun's synthesis, except at junctions where it leaves some
        # controlled link green in NO selectable phase (an atomic movement group
        # with internal SUMO foes is dropped by his synthesizer -- his docs say
        # such a junction "must be inspected").  For those junctions no tlLogic
        # is written, which is exactly what his own _build_tll does for every
        # junction it rejects: the net's default program stays active and his
        # runtime extracts that program's greens.
        fb_tree = ET.parse(synth_tll)
        fallback = []
        for logic in list(fb_tree.getroot().findall('tlLogic')):
            greens = [p.get('state') for p in logic.findall('phase') if is_selectable_green_state(p.get('state'))]
            never = [i for i in range(len(greens[0])) if not any(g[i] in 'Gg' for g in greens)]
            if never:
                fallback.append({'tls': logic.get('id'), 'never_green_link_indices': never})
                fb_tree.getroot().remove(logic)
        synthfb_tll = out_dir / 'synthfb.tll.xml'
        fb_tree.write(synthfb_tll, encoding='utf-8', xml_declaration=True)

        (out_dir / 'synth.sumocfg').write_text(_sumocfg(net_path, rou_path, begin, end, synth_tll))
        (out_dir / 'synthfb.sumocfg').write_text(_sumocfg(net_path, rou_path, begin, end, synthfb_tll))
        (out_dir / 'native.sumocfg').write_text(_sumocfg(net_path, rou_path, begin, end, None))

        entry = {
            'city_dir': city,
            'net_file': str(net_path.relative_to(REPO)),
            'route_file': str(rou_path.relative_to(REPO)),
            'window': [begin, end],
            'n_tls_in_net': len(existing),
            'existing_green_phases': existing,
            'synthesized_phases': synth_counts,
            'tls_not_synthesized': sorted(set(existing) - set(synth_counts)),
            'tlLogic_id_remap_node_to_tls': remapped,
            'synthfb_native_fallback': fallback,
        }
        for arm in ('synth', 'synthfb', 'native'):
            entry[f'runtime_{arm}'] = _runtime_check(out_dir / f'{arm}.sumocfg', begin)
        report['networks'][key] = entry
        print(json.dumps({k: v for k, v in entry.items() if k != 'runtime_native'}, indent=1)[:3000])

    out = REPO / 'results' / 'braun' / 'feasibility.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
