"""Evaluate Braun's controllers on this project's benchmark, in THIS project's metrics.

Runs inside Braun's venv:

    baselines/braun/bpy.sh <repo>/baselines/braun/eval_braun.py \
        --scenario grid4x4 --arm synth --policy learned-sample \
        --checkpoint <ckpt> --episodes 3 --sumo-seed 12345 --out results/braun/eval/x.json

Control is Braun's own code end to end: his MovementControlRuntime (min-green,
yellow insertion, legal-action masks), his movement graph and features, his
policies (``src.movement.evaluation.runner._desired_states``: learned sampled /
learned greedy / max-pressure / queue / fixed-time), with the decision schedule
of his ``run_evaluation_episode`` (one warm-up simulation step, then a decision
every ``decision_interval`` seconds).  The only thing added is measurement:

* a thin env shim exposes reset()/step() in 5-simulated-second increments, so
  this project's ``run_episode`` (loaded verbatim by trip_metrics) polls the
  simulator at exactly the instants it polls our own env (begin+5k);
* SUMO's --tripinfo-output for the same run.

Nothing about Braun's control is modified.  Settings default to his native
ones (decision interval 10 s, min green 2 decisions, yellow 3 s, no teleport).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

BRAUN_ROOT = Path('/home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(BRAUN_ROOT))
sys.path.insert(0, str(HERE))

import libsumo  # noqa: E402

from src.movement.evaluation import runner as braun_runner  # noqa: E402
from src.movement.evaluation.runner import (  # noqa: E402
    EvaluationPolicy,
    LearnedEvaluationActionMode,
    LearnedPolicyConfig,
)
from src.movement.runtime import MovementControlRuntime  # noqa: E402
from src.movement.sumo_backend import SumoBackendKind  # noqa: E402

from trip_metrics import load_run_episode, parse_tripinfo, route_departures  # noqa: E402

SCENARIOS = HERE / 'scenarios'
POLICIES = {
    'learned-sample': (EvaluationPolicy.LEARNED, LearnedEvaluationActionMode.SAMPLE),
    'learned-greedy': (EvaluationPolicy.LEARNED_GREEDY, LearnedEvaluationActionMode.DETERMINISTIC),
    'max-pressure': (EvaluationPolicy.MAX_PRESSURE, None),
    'queue': (EvaluationPolicy.QUEUE, None),
    'fixed-time': (EvaluationPolicy.FIXED_TIME, None),
}


def _cfg_window(cfg: Path) -> tuple[float, float, Path]:
    import xml.etree.ElementTree as ET

    root = ET.parse(cfg).getroot()
    begin = float(root.find('./time/begin').attrib['value'])
    end = float(root.find('./time/end').attrib['value'])
    rou = Path(root.find('./input/route-files').attrib['value'])
    return begin, end, rou


class BraunShimEnv:
    """reset()/step() facade over one Braun evaluation episode (5 s per step)."""

    POLL_S = 5

    def __init__(self, cfg: Path, policy: str, checkpoint: Path | None, sumo_seed: int, sample_seed: int,
                 decision_interval: int, min_green_steps: int, yellow_duration: int,
                 fixed_time_phase_duration: int, queue_pressure_phase_duration: int, device: str,
                 tripinfo_path: Path):
        self.cfg = cfg
        self.policy, action_mode = POLICIES[policy]
        self.learned_cfg = (
            LearnedPolicyConfig(checkpoint_path=checkpoint, device=device, action_mode=action_mode, temperature=1.0)
            if checkpoint is not None and action_mode is not None else None
        )
        self.sumo_seed = sumo_seed
        self.sample_seed = sample_seed
        self.decision_interval = decision_interval
        self.min_green_steps = min_green_steps
        self.yellow_duration = yellow_duration
        self.fixed_time_decisions = fixed_time_phase_duration // decision_interval
        self.qp_decisions = queue_pressure_phase_duration // decision_interval
        self.tripinfo_path = tripinfo_path
        self.begin, self.end, _ = _cfg_window(cfg)
        self.runtime = None
        self.phase_choice_counts: dict[str, list[int]] = {}

    # ---- Braun's episode setup (mirrors runner.run_evaluation_episode) ----
    def reset(self):
        net_path = braun_runner.resolve_sumocfg_net_path(self.cfg)
        lane_ids_by_edge, lane_geometries = braun_runner.lane_inputs_from_net(net_path)
        self.runtime = MovementControlRuntime(
            cfg_path=self.cfg, gui=False, seed=self.sumo_seed,
            yellow_duration=self.yellow_duration, yellow_start_delay=0,
            min_green_steps=self.min_green_steps, time_to_teleport=-1,
            additional_sumo_args=('--tripinfo-output', str(self.tripinfo_path)),
            backend_kind=SumoBackendKind.LIBSUMO,
        )
        self.runtime.start()
        rt = self.runtime
        rt.step()  # Braun's runner advances one step before the first decision
        self.k = 1
        # Learned sampling RNG: Braun seeds it with (sumo seed + 17213); we pass
        # an explicit sample_seed so repeated episodes on ONE traffic realisation
        # draw different samples.
        self.learned_ctx = braun_runner._learned_context(
            policy=self.policy, learned_policy_config=self.learned_cfg, programs=rt.programs,
            lane_ids_by_edge=lane_ids_by_edge, lane_geometries=lane_geometries,
            decision_interval=self.decision_interval, net_path=net_path,
            vehicle_api=rt.vehicle_api, seed=self.sample_seed - 17_213,
        )
        self.baseline_ctx = braun_runner._baseline_context(
            policy=self.policy, programs=rt.programs, lane_ids_by_edge=lane_ids_by_edge,
            lane_geometries=lane_geometries, net_path=net_path, vehicle_api=rt.vehicle_api,
            seed=self.sumo_seed,
        )
        self.accepted: dict[str, str] = {}
        self.phase_choice_counts = {t: [0] * len(p.selectable_phases) for t, p in rt.programs.items()}
        return {tls: None for tls in rt.programs}, {}

    def _decide(self):
        rt = self.runtime
        desired = braun_runner._desired_states(
            policy=self.policy, runtime=rt, programs=rt.programs,
            baseline_context=self.baseline_ctx, learned_context=self.learned_ctx,
            accepted_targets=self.accepted, decision_index=(self.k - 1) // self.decision_interval,
            fixed_time_phase_decisions=self.fixed_time_decisions,
            queue_pressure_phase_decisions=self.qp_decisions,
        )
        braun_runner._record_phase_counts(self.phase_choice_counts, rt.programs, desired)
        self.accepted = dict(rt.request_targets(desired))

    def step(self, _actions):
        rt = self.runtime
        target_k = (self.k // self.POLL_S + 1) * self.POLL_S
        horizon = int(self.end - self.begin)
        while self.k < min(target_k, horizon):
            if (self.k - 1) % self.decision_interval == 0:
                self._decide()
            rt.step()
            self.k += 1
        info = self._system_info()
        done = self.k >= horizon or not rt.is_running()
        obs = {tls: None for tls in rt.programs}
        return obs, {}, {'__all__': done}, info

    def _system_info(self):
        # Same quantities, same formulas, as sumo_rl SumoEnvironment._get_system_info,
        # which is where run_episode's queue/wait columns come from for our env.
        veh = libsumo.vehicle
        ids = veh.getIDList()
        speeds = [veh.getSpeed(v) for v in ids]
        waits = [veh.getWaitingTime(v) for v in ids]
        return {
            'system_total_stopped': sum(int(s < 0.1) for s in speeds),
            'system_mean_waiting_time': 0.0 if not ids else statistics.fmean(waits),
        }

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None


class _NoAgent:
    def act_batch(self, obs, explore=False):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['grid4x4', 'cologne3', 'ingolstadt7', 'arterial4x4'])
    ap.add_argument('--arm', default='synthfb', choices=['synth', 'synthfb', 'native'])
    ap.add_argument('--policy', required=True, choices=sorted(POLICIES))
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--sumo-seed', type=int, required=True)
    ap.add_argument('--sample-seed-base', type=int, default=1000)
    ap.add_argument('--decision-interval', type=int, default=10)
    ap.add_argument('--min-green-steps', type=int, default=2)
    ap.add_argument('--yellow-duration', type=int, default=3)
    ap.add_argument('--fixed-time-phase-duration', type=int, default=20)
    ap.add_argument('--queue-pressure-phase-duration', type=int, default=10)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    if args.policy.startswith('learned') and args.checkpoint is None:
        raise SystemExit('--checkpoint is required for learned policies')
    cfg = SCENARIOS / args.scenario / f'{args.arm}.sumocfg'
    begin, end, rou = _cfg_window(cfg)
    departures = route_departures(rou, begin, end)
    run_episode = load_run_episode(None)  # getTimeLoss branch -- see trip_metrics docstring

    episodes = []
    for ep in range(args.episodes):
        with tempfile.NamedTemporaryFile(suffix='.xml', prefix='braun_tripinfo_', delete=False) as h:
            tripinfo = Path(h.name)
        env = BraunShimEnv(
            cfg=cfg, policy=args.policy, checkpoint=args.checkpoint, sumo_seed=args.sumo_seed,
            sample_seed=args.sample_seed_base + ep, decision_interval=args.decision_interval,
            min_green_steps=args.min_green_steps, yellow_duration=args.yellow_duration,
            fixed_time_phase_duration=args.fixed_time_phase_duration,
            queue_pressure_phase_duration=args.queue_pressure_phase_duration, device=args.device,
            tripinfo_path=tripinfo,
        )
        t0 = time.time()
        try:
            m = run_episode(env, _NoAgent(), libsumo)
            n_controlled = len(env.runtime.programs)
            phase_counts = env.phase_choice_counts
            sim_end = libsumo.simulation.getTime()
        finally:
            env.close()
        m.update(parse_tripinfo(tripinfo, begin, end))
        tripinfo.unlink(missing_ok=True)
        m.update({
            'episode': ep, 'sample_seed': args.sample_seed_base + ep, 'sim_end_time': sim_end,
            'departures_in_window': departures, 'completion': m['arrived'] / departures,
            'ti_completion': m['ti_arrived'] / departures, 'n_controlled_tls': n_controlled,
            'wall_s': round(time.time() - t0, 1), 'phase_choice_counts': phase_counts,
        })
        episodes.append(m)
        print(f"[{args.scenario}/{args.arm}/{args.policy}{args.tag} ep{ep}] delay {m['delay']:.2f} trip "
              f"{m['trip_time']:.2f} arrived {m['arrived']} ({100 * m['completion']:.1f}%) wait {m['wait']:.2f} | "
              f"tripinfo delay {m['ti_delay']:.2f} trip {m['ti_trip_time']:.2f} wait {m['ti_wait']:.2f} "
              f"arrived {m['ti_arrived']} | t_end {sim_end} wall {m['wall_s']}s", flush=True)

    out = {
        'controller': f'braun/{args.policy}', 'scenario': args.scenario, 'arm': args.arm,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None, 'tag': args.tag,
        'settings': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'braun_commit': 'ea47985ccba2bbca273eb08139645399cf53ef23',
        'window': [begin, end], 'episodes': episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
