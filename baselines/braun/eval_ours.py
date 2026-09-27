"""Evaluate OUR checkpoints / rule-based controllers for the Braun comparison.

Runs in this project's own (miniconda) environment.  It is
diagnostics/eval_paper_metrics.py's main loop with exactly two additions and
no behavioural change:

  1. SUMO --tripinfo-output is switched on per episode (by setting the sumo-rl
     env's ``additional_sumo_cmd`` before each reset) so every controller also
     gets tripinfo delay / trip time / per-vehicle waiting time, parsed by the
     same trip_metrics.parse_tripinfo used for Braun's rows;
  2. per-episode results are written to JSON (the original only prints means).

The episode itself is eval_paper_metrics.run_episode, imported (not copied),
and env / evaluator construction is the same code path: make_holdout_evaluator
for the zero-shot holdout (grid4x4, sumo seed 12345, comm dropout as in every
holdout number in the paper), HoldoutEvaluator over the roster city config for
in-distribution (--city), exactly as eval_paper_metrics --city does.

    python baselines/braun/eval_ours.py CKPT|max_pressure|fixed_time [...] \
        --base_dir environments_rescofull --pad_to_true_holdout --episodes 5 \
        [--city city_4] --out results/braun/eval/ours_<scenario>_<name>.json
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from diagnostics import eval_paper_metrics as epm  # noqa: E402
from experiments.federated_training import (  # noqa: E402
    make_holdout_evaluator,
    maybe_pad_action_dim_to_true_holdout,
    resolve_city_configs_and_dims,
)
from trip_metrics import parse_tripinfo, route_departures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('controllers', nargs='+')
    ap.add_argument('--base_dir', default='environments_rescofull')
    ap.add_argument('--pad_to_true_holdout', action='store_true')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--eval_sumo_seed', type=int, default=12345)
    ap.add_argument('--city', default=None)
    ap.add_argument('--out_dir', type=Path, required=True)
    args = ap.parse_args()

    import traci  # same import order as eval_paper_metrics.main

    _, (own_dim, neighbor_dim, k_max), action_dim, _ = resolve_city_configs_and_dims(args.base_dir)
    if args.pad_to_true_holdout:
        action_dim = maybe_pad_action_dim_to_true_holdout(action_dim, args.base_dir)

    city_dir = args.city or 'city_5_holdout'
    cfg = yaml.safe_load(open(os.path.join(args.base_dir, city_dir, 'config.yaml')))
    begin = float(cfg.get('begin_time', 0) or 0)
    end = begin + float(cfg['num_seconds'])
    departures = route_departures(REPO / cfg['route_file'], begin, end)
    scenario = Path(cfg['net_file']).name.split('.')[0]

    rule_based = {'max_pressure', 'fixed_time', 'always_zero', 'random', 'round_robin'}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ctrl in args.controllers:
        rule = ctrl if ctrl in rule_based else None
        if rule is None:
            state = torch.load(ctrl, map_location='cpu')
            state = state.get('model', state) if isinstance(state, dict) and 'model' in state else state
            agent, _ = epm.build_agent(state, k_max, action_dim)
            head = 'frap' if epm.detect_frap(state) else 'phase' if epm.detect_phase_relational(state) else 'indexed'
            name = Path(ctrl).parent.name
        else:
            agent, head, name = None, rule, rule

        if args.city:
            from environments.federated_env import ActionMaskPadder, build_federated_env
            from federated.evaluator import HoldoutEvaluator

            def _builder(cfg=cfg):
                return ActionMaskPadder(build_federated_env(cfg), action_dim)

            evaluator = HoldoutEvaluator(env_builder=_builder, episodes=1, eval_seed_base=args.eval_sumo_seed,
                                         eval_city_name=args.city)
        else:
            evaluator = make_holdout_evaluator(args.base_dir, (own_dim, neighbor_dim, k_max), action_dim,
                                               episodes=1, eval_sumo_seed=args.eval_sumo_seed)
            assert evaluator.eval_city_name == 'city_5_holdout' and evaluator.is_true_holdout, \
                'holdout fell back to a training city (sec 25 trap)'
        env = evaluator._get_env()
        if rule == 'fixed_time' and hasattr(env, 'fixed_ts'):
            env.fixed_ts = True
        base = evaluator._unwrap_base_env(env)
        policy = agent if rule is None else epm._RulePolicy(evaluator, rule)

        episodes, tripinfos = [], []
        for ep in range(args.episodes):
            fd, ti = tempfile.mkstemp(suffix='.xml', prefix='ours_tripinfo_')
            os.close(fd)
            tripinfos.append(ti)
            base.additional_sumo_cmd = f'--tripinfo-output {ti}'
            t0 = time.time()
            m = epm.run_episode(env, policy, traci)
            m['episode'] = ep
            m['wall_s'] = round(time.time() - t0, 1)
            m['sim_end_time'] = float(traci.simulation.getTime())
            episodes.append(m)
        evaluator.close()  # flushes the last tripinfo file
        for m, ti in zip(episodes, tripinfos):
            m.update(parse_tripinfo(ti, begin, end))
            os.unlink(ti)
            m['departures_in_window'] = departures
            m['completion'] = m['arrived'] / departures
            m['ti_completion'] = m['ti_arrived'] / departures
            print(f"[{scenario}/{head}/{name} ep{m['episode']}] delay {m['delay']:.2f} trip {m['trip_time']:.2f} "
                  f"arrived {m['arrived']} ({100 * m['completion']:.1f}%) wait {m['wait']:.2f} | tripinfo delay "
                  f"{m['ti_delay']:.2f} trip {m['ti_trip_time']:.2f} wait {m['ti_wait']:.2f} arrived {m['ti_arrived']}"
                  f" | t_end {m['sim_end_time']} wall {m['wall_s']}s", flush=True)
        out = {
            'controller': f'ours/{head}', 'scenario': scenario, 'name': name, 'checkpoint': ctrl if rule is None else None,
            'city_dir': city_dir, 'base_dir': args.base_dir, 'eval_sumo_seed': args.eval_sumo_seed,
            'window': [begin, end], 'episodes': episodes,
        }
        path = args.out_dir / f'ours_{scenario}_{head}_{name}.json'
        path.write_text(json.dumps(out, indent=1, default=float))
        mean = {k: statistics.fmean([e[k] for e in episodes if not np.isnan(e[k])] or [float('nan')])
                for k in ('delay', 'trip_time', 'arrived', 'wait', 'ti_delay', 'ti_wait')}
        print(f'  -> {path.name}: mean {mean}', flush=True)


if __name__ == '__main__':
    main()
