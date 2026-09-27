"""Aggregate results/braun/eval/*.json into per-scenario comparison tables (any Python 3).

    python3 baselines/braun/aggregate.py [--min-seeds 1]

Per-seed value = mean over that checkpoint's episodes; a row's mean/SE are over
seeds (SE = population std / sqrt(n), this project's convention).  Rule-based
controllers are deterministic given the SUMO seed, so they are single values
(n = 1, SE = 0).  |diff|/SE is unpaired: sqrt(SE_a^2 + SE_b^2).
Writes results/braun/summary.json and prints markdown tables.
"""
import argparse
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / 'results' / 'braun' / 'eval'
METRICS = ['delay', 'trip_time', 'completion', 'wait', 'ti_delay', 'ti_trip_time', 'ti_wait', 'ti_completion']


def ep_mean(episodes, k):
    xs = [e[k] for e in episodes if e.get(k) is not None and not (isinstance(e[k], float) and math.isnan(e[k]))]
    return statistics.fmean(xs) if xs else float('nan')


def load():
    rows = defaultdict(lambda: defaultdict(list))  # scenario -> row label -> list of per-seed dicts
    for f in sorted(glob.glob(str(EV / '*.json'))):
        d = json.load(open(f))
        name = Path(f).stem
        sc = d['scenario']
        per = {k: ep_mean(d['episodes'], k) for k in METRICS}
        per['n_episodes'] = len(d['episodes'])
        per['file'] = name
        if name.startswith('ours_'):
            label = {'phase': 'ours phase-relational (greedy)', 'max_pressure': 'ours max_pressure (existing phases)',
                     'fixed_time': 'ours fixed_time (RESCO program)'}[d['controller'].split('/')[1]]
        else:
            m = re.match(r'braun_[a-z0-9]+_(synthfb|native|synth)_s(\d+)_it(\d+)_(sample|greedy)$', name)
            if m:
                arm, seed, it, mode = m.groups()
                label = f'Braun PPO [{arm}] it{int(it)} {"sampled" if mode == "sample" else "greedy"}'
                per['seed'] = int(seed)
            else:
                m = re.match(r'braun_[a-z0-9]+_(synthfb|native|synth)_(max-pressure|queue|fixed-time)$', name)
                arm, pol = m.groups()
                label = f'Braun {pol} [{arm}]'
        rows[sc][label].append(per)
    return rows


def stats(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float('nan'), float('nan')
    mu = statistics.fmean(vals)
    se = statistics.pstdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mu, se


def zstat(a, b, k):
    ma, sa = stats([r[k] for r in a])
    mb, sb = stats([r[k] for r in b])
    se = math.sqrt(sa ** 2 + sb ** 2)
    return (ma - mb), (abs(ma - mb) / se if se > 0 else float('inf'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-seeds', type=int, default=1)
    args = ap.parse_args()
    rows = load()
    summary = {}
    order = ['grid4x4', 'cologne3', 'ingolstadt7']
    for sc in [s for s in order if s in rows] + [s for s in rows if s not in order]:
        print(f'\n### {sc}\n')
        print('| controller | delay s | trip s | completion % | wait s (tripinfo) | wait s (sumo-rl) | '
              'tripinfo delay s | n seeds x eps |')
        print('|---|---:|---:|---:|---:|---:|---:|---|')
        summary[sc] = {}
        for label in sorted(rows[sc], key=lambda s: (not s.startswith('Braun PPO'), not s.startswith('Braun'), s)):
            rs = rows[sc][label]
            if len(rs) < args.min_seeds and label.startswith('Braun PPO'):
                continue
            st = {k: stats([r[k] for r in rs]) for k in METRICS}
            n = len(rs)
            eps = sorted({r['n_episodes'] for r in rs})
            summary[sc][label] = {'n': n, 'episodes_per_seed': eps, **{k: {'mean': v[0], 'se': v[1]} for k, v in st.items()},
                                  'per_seed': rs}

            def f(k, scale=1.0, nd=1):
                mu, se = st[k]
                return f'{mu * scale:.{nd}f}' + (f' ± {se * scale:.{nd}f}' if n > 1 else '')
            print(f"| {label} | {f('delay')} | {f('trip_time')} | {f('completion', 100)} | {f('ti_wait')} | "
                  f"{f('wait')} | {f('ti_delay')} | {n} x {'/'.join(map(str, eps))} |")
        # significance
        ours = rows[sc].get('ours phase-relational (greedy)')
        comps = []
        for arm in ('synthfb', 'native'):
            for mode in ('sampled', 'greedy'):
                lab = f'Braun PPO [{arm}] it85 {mode}'
                if lab not in rows[sc]:
                    continue
                b = rows[sc][lab]
                if ours:
                    comps.append((lab, 'ours phase-relational', b, ours))
                mp = rows[sc].get(f'Braun max-pressure [{arm}]')
                if mp:
                    comps.append((lab, f'Braun max-pressure [{arm}]', b, mp))
        if comps:
            print('\n| A | B | Δdelay (A−B) | \\|Δ\\|/SE delay | Δcompletion pp | \\|Δ\\|/SE completion | n_A | n_B |')
            print('|---|---|---:|---:|---:|---:|---:|---:|')
            summary[sc]['significance'] = []
            for la, lb, a, b in comps:
                dd, zd = zstat(a, b, 'delay')
                dc, zc = zstat(a, b, 'completion')
                print(f'| {la} | {lb} | {dd:+.1f} | {zd:.2f} | {100 * dc:+.1f} | {zc:.2f} | {len(a)} | {len(b)} |')
                summary[sc]['significance'].append({'A': la, 'B': lb, 'd_delay': dd, 'z_delay': zd,
                                                    'd_completion': dc, 'z_completion': zc, 'nA': len(a), 'nB': len(b)})
    out = REPO / 'results' / 'braun' / 'summary.json'
    out.write_text(json.dumps(summary, indent=1, default=str))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
