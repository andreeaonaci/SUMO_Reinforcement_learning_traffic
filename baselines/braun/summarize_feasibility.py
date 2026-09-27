"""Print a compact per-network table from results/braun/feasibility.json (any Python)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
rep = json.loads((REPO / 'results' / 'braun' / 'feasibility.json').read_text())
for key, e in rep['networks'].items():
    rs, rn = e['runtime_synth'], e['runtime_native']
    print(f"\n{key}: window {e['window']}  TLS in net {e['n_tls_in_net']}  "
          f"synth-controlled {rs['n_controlled']}  native-controlled {rn['n_controlled']}  "
          f"not synthesized: {e['tls_not_synthesized']}  remapped ids: {len(e['tlLogic_id_remap_node_to_tls'])}")
    print(f"   sim start t={rs['sim_start_time']}  graph(synth): lane_groups={rs['graph_lane_groups']} "
          f"movement_nodes={rs['graph_movement_nodes']}  pass_through={rs['graph_pass_through_tls']}")
    print(f"   active programs (synth run): {sorted(set(rs['active_program_ids'].values()))}  "
          f"(native run): {sorted(set(rn['active_program_ids'].values()))}")
    print(f"   {'tls':<24} {'existing':>8} {'synth':>6} {'rt_synth':>8} {'rt_native':>9} {'links':>6}")
    for tls in sorted(e['existing_green_phases']):
        print(f"   {tls[:24]:<24} {e['existing_green_phases'][tls]:>8} {e['synthesized_phases'].get(tls, '-'):>6} "
              f"{rs['phases_per_tls'].get(tls, '-'):>8} {rn['phases_per_tls'].get(tls, '-'):>9} "
              f"{rs['movements_per_tls'].get(tls, '-'):>6}")
