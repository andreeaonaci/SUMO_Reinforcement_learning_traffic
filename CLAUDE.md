# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline for **federated reinforcement learning of a single shared traffic-signal
policy** across multiple SUMO-simulated cities (intersections of different topologies: 3x3/4x4
grids, RESCO cologne3/ingolstadt7/grid4x4/arterial4x4). One DQN architecture — a
foundation model — controls every intersection in every city; topology differences (3-way vs
5-way, missing neighbors, etc.) are expressed entirely through `action_mask` / `neighbor_mask`,
never through per-topology code paths. `PROJECT_FLOW.md` has a detailed module-by-module trace
of the `--parallel` execution path (call hierarchy, class responsibilities, data flow) — read it
before making non-trivial changes to the federated training path.

## Setup

Requires SUMO installed with `SUMO_HOME` set:

```bash
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
pip install .            # or: pip install .[rendering] for pyvirtualdisplay support
```

`bash setup_wsl.sh` does a one-time WSL/Ubuntu setup (apt SUMO packages + pip deps).

## Common commands

Federated training (parallel = one worker process per city, recommended):
```bash
python -m experiments.federated_training --parallel --rounds 10 --local_episodes 2 \
    --aggregation_strategy fedavg --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5
```
Key flags: `--seed`, `--eval_every`, `--eval_episodes`, `--aggregation_strategy`
{fedavg, ema_alignment, clustered_fedavg, ...}, `--n_clusters` (for clustered_fedavg),
`--no_federation` (train each city independently, no aggregation), `--baseline_controller`
{fixed_time, max_pressure} (skip training, evaluate a rule-based controller instead),
`--tau`/`--target_update` (DQN target network), `--base_dir` (which `environments*/` roster to
use — see "City configs" below), `--disable_head_fix` (ablation: turn off masked-head
aggregation).

Each run creates a timestamped directory under `results/run_<timestamp>/` with the global model
checkpoint and `federated_history.json`.

Evaluate a trained model:
```bash
python experiments/evaluate.py --model results/global_fed.pth --episodes 5
```

Other experiment entry points in `experiments/`: `local_training.py` (single-city, no
federation), `centralized.py` (centralized baseline), `run_nacrl.py`, `sarsa_double.py`,
`sarsa_resco.py`, `sanity_check.py`, `validate_sumo_cities.py` (checks every city config loads),
`plot_convergence.py`, `analyze_phase0.py`, `analyze_phase1.py`.

Phase 1 ablation sweep (7-city roster x 5 seeds x multiple aggregation strategies + rule-based
baselines; skips runs already completed): `bash analyse/run_phase1_ablation.sh`. Output lands in
`results/phase1/<run_name>/`; summarize with
`python experiments/analyze_phase1.py --results_root results/phase1`.

**Default way to run any multi-run experiment batch** (seed sweeps, flag ablations, cheap
validation matrices — the `environments_c1_4`/`environments_c1_4_6` style small-batch testing
used throughout `fidings/divergence_investigation.md`): `analyse/run_concurrent_batch.sh`, not a
one-off sequential script. Runs jobs with bounded concurrency (default 3 at a time) instead of
one-at-a-time — empirically CPU is not the bottleneck for a single run (each city worker uses
~13-15% of one core; SUMO/libsumo per-tick stepping is the real constraint), RAM is (~2.5-3.5GB
per run). See the script's header for usage and the `run_dir` PID-suffix fix in
`experiments/federated_training.py::main()` it depends on for concurrent launches to not collide.

Tests:
```bash
pytest tests/                      # gym_test.py (Gymnasium API), pz_test.py (PettingZoo API)
```


## Architecture

### Two execution paths
- **Parallel** (`--parallel`, `federated/parallel_server.py::ParallelFederatedServer`): spawns
  one persistent worker process per city (multiprocessing, spawn context). Each worker keeps a
  warm SUMO environment + replay buffer alive across rounds; the main process only ships model
  state dicts back and forth each round. This is the path used for real training runs.
- **Sequential** (`federated/server.py::FederatedServer`, via `federated/client.py`): builds and
  tears down each city's environment every round in a single process. Simpler, used for
  quick/mock runs and debugging.

Both converge on the same round loop: broadcast global weights → each city trains locally for
`--local_episodes` episodes → collect updated state dicts + sample counts → aggregate → evaluate
on the holdout city → checkpoint.

### Observation/action contract (the thing that makes topology-agnostic RL work)
Defined and documented in `agents/networks.py` and built by
`environments/federated_env.py::MultiAgentFederatedWrapper`. Every intersection, in every city,
produces the same shape regardless of topology:
- `own_obs (D_own,)` — fixed-size own-intersection features (built by `LaneExtractor` →
  `LaneNormalizer`/`LaneSorter` → `TopKEncoder`)
- `neighbor_obs (K_MAX, D_nbr)` — zero-padded per-neighbor features (`NeighborGraphBuilder` finds
  K-hop neighbors from the SUMO net topology; `NeighborSummaryExtractor` summarizes each one)
- `neighbor_mask (K_MAX,)` — 1.0 valid neighbor this tick, 0.0 padded or comm-dropped
- `hop_dist (K_MAX,)` — hop distance per neighbor slot
- `action_mask (A_MAX,)` — 1.0 = real action for this intersection's actual phase count, 0.0 =
  padding. This replaces any hand-written per-topology phase mapping; `ActionSpaceInspector`
  discovers valid action counts by probing SUMO directly. `ActionMaskPadder` pads every city's
  action space up to the shared global width (`max_action_dim`) so one Q-head serves all cities.

The network (`agents/networks.py::NeighborAttentionQNetwork`) never sees which city/topology an
observation came from — everything topology-specific is expressed purely through the masks.

`CommDropoutWrapper` (`federated/comm_dropout.py`) sits around the environment during both
training and eval and corrupts `neighbor_mask`/`neighbor_obs` per `p_link`/`p_isolate`/
`p_hop_cutoff`, simulating unreliable inter-intersection communication.

### Aggregation strategies
`federated/aggregation_strategies.py` implements multiple pluggable strategies behind
`build_aggregation_strategy()`: `FedAvgStrategy` (sample-count-weighted average, not equal
weighting), `EMALossImprovementStrategy`, `EMAGradientAlignmentStrategy`,
`LearningVelocityNoveltyStrategy`, `GradientSurvivalStrategy`, `ClusteredFedAvgStrategy`
(clusters cities by `action_dim` and aggregates within-cluster). `federated/aggregation.py`
has the underlying `fed_avg`/`weighted_average`/`masked_head_weighted_average` primitives — the
masked-head variant only averages the Q-head slots that were actually active for each
contributing city, since action spaces differ in width.

### Agent
`agents/dqn.py::DQNAgent` is the single shared Q-learning agent class used both as the global
model and as each worker's local model — same class, same architecture, just different weights
in flight. Holds the online/target networks, `ReplayBuffer`, optimizer, and epsilon schedule.

### Evaluation
`federated/evaluator.py::HoldoutEvaluator` evaluates the aggregated global model (or a rule-based
baseline controller) on a held-out city not used in training (`city_5_holdout` in the default
7-city roster) — reward, waiting time, action distribution, and Q-gap diagnostics.

### City configs
Each city is a directory under `environments/` (`city_1` … `city_7`, `city_5_holdout`) holding a
`config.yaml` that points at a SUMO `.net.xml`/`.rou.xml` pair under `sumo_rl/nets/` plus
sim params (`delta_time`, `num_seconds`, `k_max`, `max_hops`, `use_libsumo`, ...).
`city_5_holdout` is auto-excluded from training and reserved for `HoldoutEvaluator` **only when its
action space (width 8) fits within the roster's global `action_dim`** — `make_holdout_evaluator`
(`experiments/federated_training.py`) silently falls back to the first compatible *training* city
otherwise (logged as `"Using '<city>' as evaluation city ... (compatibility fallback)"`). Confirmed
2026-08-13 (`fidings/divergence_investigation.md` §25) that every 2-city (`environments_c1_4`) run
in this project has actually been evaluating on `city_1`, one of its own two training cities, not a
true holdout — check the run's log for that warning before trusting any "generalizes to unseen
city" framing on a reduced roster; the 7-city (`environments`) roster does use the real holdout.
`environments_phase0/` and `environments_city1/` are alternate rosters made of symlinks back into
`environments/*` (selected via `--base_dir`) — used to scope which cities a given experiment run
sees, not separate environment implementations. `configs/default.yaml` holds the historical
default hyperparameters (rounds/lr/batch_size); most of these are now overridden via CLI flags in
`experiments/federated_training.py`.

### Diagnostics
`diagnostics/` has standalone one-off scripts for inspecting SUMO route/action-space data
(`inspect_action_spaces.py`, `route_traffic_balance.py`, `measure_approach_volume.py`,
`q_gap_trend.py`, `dump_route_schema.py`, `fix_route_windows.py`) — not part of the training
pipeline, run manually when debugging a specific city's data.

### `sumo_rl/` package
This is the underlying installable Gymnasium/PettingZoo package (`sumo_rl/environment/env.py` is
the single-agent/multi-agent SUMO wrapper it's built on). It's a dependency of the federated
pipeline above, not the pipeline itself — treat `environments/federated_env.py` as the layer that
adapts `sumo_rl` environments into the federated multi-city contract.

## Research plan status (paper track)

`PROJECT_NEXT_STEPS.md` is the source of truth for the phased research plan (Phase 0 infra
stabilization → Phase 1 cheap validation → Phase 2 full-scale validation → Phase 3 baselines →
Phase 4 clustering/related-work). `fidings/` holds dated investigation write-ups (what was
tested, what broke, what's still open) — read the latest one there before trusting any past run's
numbers at face value. As of 2026-08-02, audited against the actual code (not just the plan doc,
which had gone stale):

- **Phase 0 infra items are already implemented**, contrary to the plan doc's "in progress"
  status: target network + Double DQN (`agents/dqn.py::DQNAgent.optimize`), persistent
  optimizer/replay-buffer/agent across rounds (one `DQNAgent` per worker process, created once
  before the round loop — `federated/parallel_server.py`), no remaining per-city LR override
  (`--lr` controls every city uniformly, confirmed no `environments/*/config.yaml` sets `lr:`),
  reward clipping (`reward_clip=10.0`) + Huber loss for outlier robustness, gradient-norm
  clipping, process-based (not thread-based) `--parallel` parallelism, incremental
  checkpointing every round. Update the plan doc's Phase 0 status if you re-run this audit and
  it still holds.
- **Phase 0's decision gate is NOT cleanly passed**, despite the code being done: two full
  20-round runs with identical code/config/`--seed 42` produced completely different outcomes
  (one learned cleanly, one stayed flat the whole run) — see `fidings/divergence_investigation.md`
  §3. This run-to-run non-determinism (suspect: SUMO/TraCI or multiprocessing-worker scheduling
  not fully pinned by the Python-level seed) is the actual current blocker on the plan's critical
  path, not any of the originally-diagnosed infra bugs.
- **Phase 4's clustering strategy is already implemented and wired up correctly**
  (`ClusteredFedAvgStrategy` in `federated/aggregation_strategies.py`), including the per-cluster
  broadcast-routing the plan doc calls out as easy to get wrong — confirmed both
  `federated/server.py` and `federated/parallel_server.py` route each client its own cluster's
  aggregated state, not a naive single global broadcast. Only the *trustworthy comparison run*
  (multi-seed, full roster) is still outstanding, same as the plan doc says.
- **`federated/strategies.py::fed_prox` (a dead stub that delegated straight to plain `fed_avg`,
  never wired into the strategy registry) was deleted 2026-08-13** during a code-quality pass —
  confirmed zero references anywhere outside itself first. Don't confuse this with the *actually
  implemented and tested* FedProx proximal term, `DQNAgent.mu` — see next bullet — which is real
  and unaffected by this deletion.

## RESUME HERE (as of 2026-08-13 — check this is still current before trusting it)

Phase 1 is now complete at all three roster sizes (2/3/7-city, 5 seeds each). Read
`fidings/divergence_investigation.md` in full before doing anything non-trivial here — it's long
but every number is re-derivable and the reasoning matters. Short version:

- **Current best-known training config: `--dueling --n_step 3`** (dueling Q-head +
  n-step returns). Validated on 5 seeds, 2-city roster (§21): mean reward -2030.4 (std 515.0),
  no seed-outlier failure mode. Do not use `--fedprox_mu` (tested, no effect, §14) or
  `--server_momentum` combined with `--dueling` (tested, net-negative, §18 — a hard CLI check in
  `experiments/federated_training.py` blocks this combination). `--pseudo_grad_clip` and
  `--eval_ema_decay` are implemented but unconvincing (§19) — not part of the recommended config.
- **Phase 1's masked-head ablation, redone on `--dueling --n_step 3`, three roster sizes, now
  complete (§20 for 2/3-city, §23 for 7-city brought to 5 seeds on 2026-08-13):** mean-reward
  benefit shrinks monotonically with roster size and is fully gone by 7 cities — |diff|/SE: 3.42
  (2-city, clean win) → 0.71 (3-city, ambiguous) → 0.23 (7-city, genuine null, not a data gap).
  Best-round benefit is real at every roster size (fix-on reaches a meaningfully better peak at
  2/3/7-city) but also shrinks in relative terms: ~95% → 76% → 44% better best-round mean.
  **No more seeds needed on this question — Phase 1 is done.**
- **CRITICAL, as of 2026-08-13: the `fixed_time` rule-based baseline was broken project-wide, now
  fixed — see §24.** Three-layer bug (wrapper classes' `__getattr__` covers reads only, not
  writes, so `env.fixed_ts = True` silently shadowed instead of reaching the raw SUMO env; plus a
  missing multi-agent guard in `sumo_rl`'s `SumoEnvironment.step()`). Every `fixed_time` number
  ever cited in this project (going back to §1) measured a degenerate "never switch off the first
  phase" policy, not real fixed-time control. Fixed in `environments/federated_env.py`,
  `federated/comm_dropout.py`, `sumo_rl/environment/env.py`. **Consequence that matters far more
  than the number itself: with the fix, `fixed_time` (-2.73) and `max_pressure` (-0.34) both beat
  the 7-city trained DQN (mean -6918.4, best-round -2182.0) by a wide margin on the same holdout
  city** — invisible before today since the only baseline was itself broken. Not yet multi-seed,
  not yet checked on 2-city/3-city.
- **NEXT ACTION — two open decisions, not a compute task to just run:** (1) §24's finding that the
  trained DQN currently loses to simple rule-based control needs investigating properly (multi-seed
  baseline numbers, check other rosters) before spending more compute on Phase 2's
  aggregation-strategy comparison — that comparison matters less if no strategy beats a heuristic
  yet. (2) Separately, Phase 1's own decision-gate outcome is mixed (see `PROJECT_NEXT_STEPS.md`
  Phase 1/Phase 2 status) — 2-city passed cleanly but 7-city shows a null mean-reward result
  alongside a real best-round win. Per the plan's own instruction not to guess on an ambiguous
  gate, both of these need a call from the user on how to proceed. Don't unilaterally launch more
  Phase 2 compute without that decision.
- `analyse/run_concurrent_batch.sh` is the **default** way to run any multi-run batch (see
  "Common commands" above). §22 measured ~1.5x wall-clock speedup at `MAX_CONCURRENT=3` on
  2-city runs (each concurrent run individually slows ~60%, contention worsens over a run's
  duration). §23 found 7-city runs handle `MAX_CONCURRENT=2` fine despite §22's more conservative
  `MAX_CONCURRENT=1` assumption for that roster size (measured via `top`/`ps`: city workers are
  bursty, not steadily CPU-bound; RAM, not CPU, was the binding constraint at ~9GB/15.8GB with 2
  concurrent 7-city jobs) — don't assume `MAX_CONCURRENT=1` is required for 7-city, but watch RAM
  headroom if pushing higher. Depends on the `run_dir` PID-suffix fix in
  `experiments/federated_training.py::main()` (§22) — without it, concurrent launches within the
  same wall-clock second silently corrupt each other's output directories.
