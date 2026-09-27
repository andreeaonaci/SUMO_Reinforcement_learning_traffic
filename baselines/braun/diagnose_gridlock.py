"""Where is traffic stuck when Braun's max-pressure runs on synthesized ingolstadt7 phases?
Braun venv.  Usage: diagnose_gridlock.py [arm] [policy] [scenario]"""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_braun import BraunShimEnv, SCENARIOS  # noqa: E402
import libsumo  # noqa: E402

arm = sys.argv[1] if len(sys.argv) > 1 else 'synthfb'
policy = sys.argv[2] if len(sys.argv) > 2 else 'max-pressure'
scen = sys.argv[3] if len(sys.argv) > 3 else 'ingolstadt7'
env = BraunShimEnv(cfg=SCENARIOS / scen / f'{arm}.sumocfg', policy=policy, checkpoint=None, sumo_seed=42,
                   sample_seed=0, decision_interval=10, min_green_steps=2, yellow_duration=3,
                   fixed_time_phase_duration=20, queue_pressure_phase_duration=10, device='cpu',
                   tripinfo_path=Path('/tmp/braun_diag_tripinfo.xml'))
env.reset()
done = False
t_marks = {}
while not done:
    _, _, d, _ = env.step({})
    done = d['__all__']
    k = env.k
    if k % 600 == 0:
        t_marks[k] = len(libsumo.vehicle.getIDList())
print('vehicles in network every 600 s:', t_marks)
halt = collections.Counter()
for v in libsumo.vehicle.getIDList():
    if libsumo.vehicle.getSpeed(v) < 0.1:
        halt[libsumo.vehicle.getLaneID(v)] += 1
rt = env.runtime
lane_to_tls = {}
for tls, prog in rt.programs.items():
    for m in prog.movements:
        lane_to_tls.setdefault(str(m.incoming_lane_id), set()).add((tls[:28], int(m.signal_index)))
print('top halting lanes at end:')
for lane, n in halt.most_common(12):
    print(f'  {lane:<28} {n:>4}  feeds {sorted(lane_to_tls.get(lane, []))}')
print('phase choice counts:')
for tls, c in env.phase_choice_counts.items():
    states = [p.state for p in rt.programs[tls].selectable_phases]
    print(f'  {tls[:28]:<28} {c}  {states}')
env.close()
