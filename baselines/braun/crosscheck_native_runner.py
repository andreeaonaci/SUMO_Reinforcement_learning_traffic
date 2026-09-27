"""Shim-fidelity check: run Braun's OWN run_evaluation_episode (his metrics, his tripinfo
parser) on the same checkpoint / scenario / seed and print his completed-trip time loss,
travel time and completion, to compare with eval_braun.py's numbers for the matching run
(eval_braun.py --sample-seed-base <seed+17213> reproduces his sampling RNG exactly).
Braun venv.  Usage: crosscheck_native_runner.py SCENARIO ARM POLICY SEED [CHECKPOINT]"""
import sys
from pathlib import Path

BRAUN_ROOT = Path('/home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BRAUN_ROOT))
from src.movement.evaluation.runner import (  # noqa: E402
    EvaluationPolicy, LearnedEvaluationActionMode, LearnedPolicyConfig, run_evaluation_episode)
from src.movement.sumo_backend import SumoBackendKind  # noqa: E402

scen, arm, pol, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ckpt = Path(sys.argv[5]) if len(sys.argv) > 5 else None
policy = {'learned-sample': EvaluationPolicy.LEARNED, 'learned-greedy': EvaluationPolicy.LEARNED_GREEDY,
          'max-pressure': EvaluationPolicy.MAX_PRESSURE}[pol]
lcfg = LearnedPolicyConfig(ckpt, 'cpu', LearnedEvaluationActionMode.SAMPLE, 1.0) if ckpt else None
m = run_evaluation_episode(cfg_path=HERE / 'scenarios' / scen / f'{arm}.sumocfg', policy=policy, seed=seed,
                           steps=3600, decision_interval=10, yellow_duration=3, min_green_steps=2,
                           learned_policy_config=lcfg, demand_scale=1.0, initial_occupancy_min=0.0,
                           initial_occupancy_max=0.0, time_to_teleport=-1, queue_pressure_phase_duration=10,
                           backend_kind=SumoBackendKind.LIBSUMO)
print(f'BRAUN-RUNNER {scen}/{arm}/{pol} seed {seed}: completed {m.completed_vehicles} departed {m.departed_vehicles} '
      f'time_loss {m.average_time_loss_s:.2f} travel {m.average_travel_time_s:.2f} wait {m.average_waiting_time_s:.2f} '
      f'throughput/h {m.throughput_per_hour:.1f}')
