"""Why does Braun's synthesis never give green to links 6-9 of ingolstadt7/gneJ210?
Runs in Braun's venv (imports his synthesis internals, read-only)."""
import sys
from pathlib import Path

BRAUN_ROOT = Path('/home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAUN_ROOT))
import importlib.util  # noqa: E402

import sumolib  # noqa: E402

from src.movement import phase_synthesis as ps  # noqa: E402

spec = importlib.util.spec_from_file_location('bn', BRAUN_ROOT / 'scripts' / 'build_network.py')
bn = importlib.util.module_from_spec(spec)
sys.modules['bn'] = bn
spec.loader.exec_module(bn)

net = sumolib.net.readNet(str(REPO / 'sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.net.xml'), withConnections=True, withFoes=True)
node = next(n for n in net.getNodes() if n.getID() == 'cluster_371462086_469470779_98101387_cluster_371462067_371775459_371775468')
specs = bn._movement_link_specs(node)
for s in specs:
    print(s.traffic_light_link_index, s.request_index, s.incoming_lane_id, '->', s.outgoing_lane_id, s.outgoing_edge_id)
groups = ps._atomic_link_components(sorted(specs, key=lambda x: x.traffic_light_link_index))
print('\natomic components:')
for g in groups:
    idx = [x.traffic_light_link_index for x in g]
    internal = ps._has_sumo_or_outgoing_edge_conflict(list(g), node.areFoes)
    why = []
    for i, a in enumerate(g):
        for b in g[i + 1:]:
            if ps._sumo_requests_are_foes(a, b, node.areFoes):
                why.append((a.traffic_light_link_index, b.traffic_light_link_index, 'sumo-foes'))
            if ps._same_outgoing_edge_conflict(a, b):
                why.append((a.traffic_light_link_index, b.traffic_light_link_index, 'merge-from-different-approach'))
    print(idx, 'DROPPED (internal conflict)' if internal else 'ok', why)
