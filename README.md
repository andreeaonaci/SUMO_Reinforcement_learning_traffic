# Federated RL for Traffic Signal Control

This repository implements a research-grade federated reinforcement learning pipeline for adaptive traffic signal control.

Quick start (mock, no SUMO required):

```bash
 python -m experiments.federated_training --parallel --rounds 10 --local_episodes 2 --aggregation_strategy fedavg --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5
python experiments/evaluate.py --model results/global_fed.pth --episodes 5
```

See `experiments/` for scenario scripts.
# SUMO Reinforcement Learning Traffic

`sumo-rl` provides reinforcement learning environments and training utilities for traffic signal control using SUMO.

## Overview

- Gymnasium-compatible SUMO traffic signal control environments
- PettingZoo-compatible multi-agent interfaces
- Example and research-ready scenarios for 2-way, grid, and arterial intersections
- Support for SUMO network and route files packaged with the repository

## Installation

1. Install SUMO and set `SUMO_HOME` in your environment.
2. Install the package and required Python dependencies:

```bash
pip install .
```

3. For optional rendering support:

```bash
pip install .[rendering]
```

4. For optional training and NACRL PyTorch models, make sure `torch` is installed. If `pip install .` does not install PyTorch on your platform, install it separately:

```bash
pip install torch
```

## Requirements

- Python 3.9+
- `gymnasium>=0.28`
- `pettingzoo>=1.24.3`
- `numpy`
- `pandas`
- `pillow`
- `sumolib>=1.14.0`
- `traci>=1.14.0`
- `torch>=1.13.1`

## Usage

```python
from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    use_gui=False,
    num_seconds=1000,
)

obs = env.reset()
print(obs)

state, reward, done, info = env.step(env.action_space.sample())
print(reward, done)
```

## Package Contents

- `sumo_rl/environment/` – SUMO environment wrappers and observation utilities
- `sumo_rl/agents/` – learning agents and policies
- `sumo_rl/nets/` – sample SUMO network and route files
- `experiments/` – example training scripts

## Research results: federated DQN baseline (2026 investigation)

This section summarizes the results of the project's first full research pass — a federated DQN
(`agents/dqn.py`) shared across SUMO-simulated cities of different intersection topologies via
FedAvg-family aggregation (`federated/aggregation.py`, `federated/aggregation_strategies.py`). The
exact commit this section describes is preserved on the `dqn_baseline_final` branch, before the
network was swapped out for a different algorithm (PPO — see the next section once available). Full
derivation, every number's source run, and 72 dated investigation sections live in
`fidings/divergence_investigation.md`; this is the condensed, paper-oriented version.

### Headline finding

**A trained DQN policy loses to simple rule-based control (`fixed_time`, `max_pressure`) by 3-4
orders of magnitude when evaluated on a truly unseen intersection topology — but that gap is
mostly a training-budget/evaluation-protocol artifact, not evidence DQN "doesn't work" for traffic
control.** Evaluated in-distribution (the same city it trained on, with a training budget matched
to the RESCO benchmark's own published numbers), the same architecture comes within 1.4x of
`max_pressure`. The gap that remains under true cross-topology evaluation is real, large,
characterized, and only partially closed by more training budget.

| condition | metric | trained DQN | `fixed_time` | `max_pressure` |
|---|---|---:|---:|---:|
| In-distribution, single city, budget-matched to RESCO | waiting time (s) | 37.4 | 230.6 | 27.3 |
| True holdout (unseen city), 2-city roster, 5 seeds | best-round reward mean | -5278.1 | -2.73 | -0.34 |
| True holdout, 7-city roster, 5 seeds | mean reward | -6918.4 | -2.25 | -0.044 |
| True holdout, extended budget (63 vs 20 rounds) | best-round reward mean | -2285.2 (from -5278.1) | — | — |

### Why: two compounding, independently-verified mechanisms

1. **Confident lock-in.** Training sometimes converges to a policy that repeats one action
   regardless of the actual traffic state — confirmed directly by re-evaluating checkpoints across
   30 different random seeds and finding byte-identical rewards every time. This happens whether or
   not federation is involved: turning federation off entirely (`--no_federation`, each city trains
   alone) reproduces the exact same failure at a statistically indistinguishable rate
   (7.0% vs 6.0% of checkpoints, |diff|/SE = 0.34) — it's a property of DQN training against this
   task, not an artifact of aggregating weights across cities.
2. **Cross-topology generalization gap.** The policy has to control intersection layouts it never
   saw during training; the rule-based baselines apply the same fixed formula everywhere and never
   face this problem. More training budget helps substantially (best-round mean -5278.1 → -2285.2
   going from 20 to 63 rounds) but the gap to `max_pressure` is still ~6,700-17,700x at the higher
   budget — it doesn't come close to closing within 1.25x of RESCO's own training budget.

### What was tried to close the gap, and what actually helped

No **training-time or aggregation-time** intervention closed it — all of the following were tested
at proper multi-seed (or matched-pair) rigor and came back null or negative against plain FedAvg:
alternative aggregation strategies (clustered/EMA-loss/EMA-alignment/gradient-survival/velocity-
novelty — best of these, `clustered_fedavg`, |diff|/SE = 0.85, not significant), architecture changes
beyond `--dueling --n_step 3`, an extra `max_pressure`-style input feature, reward shaping, and
widening the training roster from 3 to 14 cities / 26 to 116 intersections (|diff|/SE = 0.60, clean
null — ruling out "not enough training data/diversity" as the explanation).

One **test-time** lever did help, decisively: fine-tuning a trained checkpoint briefly on synthetic
randomized traffic on the target topology itself (never the real evaluation route file), then
evaluating on the real holdout traffic. Multi-seed replicated (seeds 3/7/11/17): |diff|/SE = 72.78
(best-of-5-rounds) / 32.18 (round-5-only) against zero-shot transfer, every seed/round beating
zero-shot by at least 3.9x. Still 20-3200x off the rule-based baselines in absolute terms — a real,
well-replicated, citable improvement, not a solved problem.

### The open, uncomfortable finding

A **randomly-initialized** network, fine-tuned on the holdout city with the identical protocol,
**beat** the federated-**pretrained** network fine-tuned the same way (-406.85 vs -693.84, single
seed — directional, multi-seed replication not yet done). Combined with the roster-widening null
above, this points at the sharpest form of the project's central negative result: **the binding
constraint is not the training data's quantity or diversity — it's the algorithm's failure to
retain and transfer what it learns.** This is the problem the architecture change (DQN → a
different network/algorithm, kept behind the same FedAvg-style aggregation) is aimed at.

### Algorithm swap: is DQN itself the bottleneck? (2026-09-04/05)

Tested directly: kept FedAvg untouched, swapped the local learning algorithm behind it. Two
alternatives were implemented, both matching `DQNAgent`'s interface exactly so the federated
plumbing needed no changes beyond a `--algo {dqn,ppo,munchausen}` flag:

- **PPO** (`agents/ppo.py`) — an on-policy actor-critic with an entropy-regularized stochastic
  policy, which structurally can't collapse to one repeating action the way `argmax(Q)` can.
- **Munchausen-DQN** (`agents/munchausen_dqn.py`, Vieillard et al. 2020) — keeps DQN's off-policy
  replay buffer (same sample efficiency, unlike PPO) but replaces the hard policy with a Boltzmann
  distribution over Q-values and an entropy-regularized soft-Bellman target.

**Methodology note worth keeping in mind when reading any single-seed result in this project:** a
promising-looking Munchausen configuration (`temp=0.01 + n_step=3`) beat DQN cleanly on one seed —
and then failed to replicate on three further seeds, and even on its original seed only *tied*
DQN once run to the full training budget instead of a short screen. A textbook single-seed
mirage, caught before being reported as a real result.

**The real, statistically-grounded answer** (3 seeds each, full 20-round budget,
`environments_c1_4_6`, true holdout): **DQN+q_entropy and Munchausen-DQN are statistically
indistinguishable** — |diff|/SE = 0.46 (best-round), 0.28 (mean), both far below this project's
own ≥2 significance bar. PPO stayed clearly behind both even after quadrupling its episode budget
to remove its on-policy sample-inefficiency disadvantage. Every network-capacity variant tried
(`d_model` 64/256/512, on both DQN and Munchausen) underperformed the existing default of 128.

**One lead for later, not yet confirmed:** Munchausen-DQN's seed-to-seed variance was 5-7x
smaller than DQN's (best-round std 332 vs. 2290) even though its mean wasn't better — DQN's one
good seed was doing most of the work of making it look ahead. A specific, testable
consistency-vs-peak-performance hypothesis for a future dedicated multi-seed variance study.

**Conclusion: no algorithm swap earned its keep.** DQN+q_entropy remains the standing default.
Full round-by-round data and every intermediate (including the mirage) is in
`fidings/divergence_investigation.md` §73.

### Paper framing

Not "federated DQN traffic control fails" (disproven by the in-distribution result above) — the
defensible claim is: *in-distribution, this approach is competitive with published numbers once
training-budget and evaluation-protocol confounds are controlled for; cross-topology generalization
has a real, characterized, partially budget-resistant gap; here is a rigorous mechanism
investigation (confident lock-in) of the instability underneath it, plus a working test-time
mitigation and a still-open transfer/retention anomaly.* Same category of contribution as the
RESCO benchmark paper itself, whose own headline finding is "published methods underperform simple
baselines in realistic scenarios."

## License

MIT License
