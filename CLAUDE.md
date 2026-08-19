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

## RESUME HERE (as of 2026-08-19 night — check this is still current before trusting it)

**No internet available in this session — do not attempt WebFetch/WebSearch or assume network
access.** Everything needed to resume is local: this file, `fidings/divergence_investigation.md`,
and the `results/` run directories. If a task seems to need a web lookup (e.g. another literature
check like §35), flag it as blocked rather than guessing or fabricating a source.

**No experiment currently running as of this writeup** — the 5-seed baseline batch flagged in the
previous version of this note finished cleanly (all 4 exit=0, see §47) and nothing new has been
launched since. Safe to start something without first checking for a stale in-flight job, though
`ps aux | grep federated_training` costs nothing to confirm.

Phase 1 is complete at all three roster sizes (2/3/7-city, 5 seeds each). Read
`fidings/divergence_investigation.md` in full before doing anything non-trivial here — it's long
(47 sections as of this writeup) but every number is re-derivable and the reasoning matters. Short
version, newest first:

- **§47 corrects §46: the `--dueling --n_step 3` architecture recommendation's edge over plain
  FedAvg does NOT hold up at 5-seed rigor under true-holdout eval — |diff|/SE = 0.63 (best-round),
  0.56 (mean), both far below this project's ≥2 bar.** §46's single-seed (seed 3) finding that
  dueling+n_step clearly beat the baseline was itself a case of the standing "single-seed story
  doesn't replicate" pattern (§11→§12, §30→§31) — plain FedAvg's own seed 5 (best -3396.76) beat
  three of dueling+n_step's five seeds. **`--dueling --n_step 3` remains the best guess (it's still
  what §15/§19 validated for the in-distribution eval, a separate and still-true claim), but is NOT
  a confirmed win under true-holdout evaluation.** No architecture or aggregation-strategy
  comparison currently has a statistically supportable non-trivial baseline to build on, and none
  come close to `fixed_time`/`max_pressure` regardless (§45) — **this is a stronger reason than §46
  gave not to scale Phase 2 yet.**

- **CRITICAL, confirmed at full 5-seed rigor 2026-08-19: the 2-city "best-round beats baselines"
  claim (§21, §29) does not survive a genuine holdout — it was entirely an artifact of evaluating
  in-distribution (§43, confirmed §45).** Merged in `debugging_andreea`'s `--pad_to_true_holdout`
  flag (widens a reduced roster's Q-head so it can actually be evaluated on `city_5_holdout`
  instead of always falling back to `city_1`, one of its own training cities). 5-seed true-holdout
  2-city result: best-round mean -5278.1 (std 2335.2) vs. `max_pressure`'s -0.34 and `fixed_time`'s
  -2.73 — **|diff|/SE = 5.05 (best-round), 13.79 (mean reward)**, the cleanest, most decisive
  result in the whole investigation, every single seed's best round 3-4 orders of magnitude worse
  than either baseline. A 3-city pilot lands in the same range — adding a third training city
  doesn't help. **Every "2-city best-round beats baselines" statement anywhere in this file or in
  the fidings doc's §21/§29 (and the neighbor-attention thread §30/§31, which also evaluated on
  `city_1`) is superseded, not just caveated — correction notes added to those sections.** At every
  roster size and every seed now tested with a true holdout (2-city, 3-city pilot, 7-city since
  §24), the trained DQN loses decisively to both rule-based baselines, full stop. Two other new
  capabilities merged in the same commit: `--reward_shaping_wait_weight`/
  `--reward_shaping_stopped_weight` (training-only reward shaping, targets the 7-city
  queue-draining gap from §26/§28) — first pilot (1 seed, conservative weight) looked worse than
  the unshaped baseline, inconclusive rather than a rejection (§44) — and 6 real bugs fixed in
  `sumo_rl/nacrl/` (separate algorithm, "training never actually happened" was the worst one) plus
  one more found and fixed live (`SumoEnvironmentPZ.__init__` reading action/observation spaces
  before the env had ever been reset) — NACRL itself is still blocked on this sandbox missing the
  `pettingzoo` dependency (network too throttled to install it as of this writeup).
- **The "why does the trained DQN lose to rule-based baselines" investigation (2026-08-15 to
  2026-08-18, §30-§42) narrowed the mechanism a lot without fully resolving it.** Chain of
  elimination, each ruling out a candidate cause: weight-divergence/gradient-conflict between
  cities doesn't predict a round's crash (§32); the crashes are real, reproducible policy failures
  that survive 6x more eval episodes, not measurement noise (§33); crashed rounds are genuinely
  **confidently-locked degenerate policies** — byte-identical rewards across different SUMO seeds,
  `min_gap` (Q-value confidence) correlating -0.884 with reward within the one checkpoint with real
  gap variance — the network gets *sure* of a bad repeating action, and rare low-confidence moments
  are what let it escape (§34). A literature check against RESCO (the benchmark this project's city
  configs are drawn from, fetched and read directly) confirmed this project's pure-argmax-at-eval
  convention matches the field standard (§35) — the failure mode isn't a project-specific mistake.
  Two fixes were built and tested: **softmax(Q/0.2) at eval time recovers near-optimal episodes
  from a checkpoint pure argmax never once escaped**, but only partially (§36); **a short
  training-time exploration-reset burst durably fixed a moderately-locked checkpoint but not a
  severely-locked one** (§39/§40), and **turning that into a standing `--epsilon_reset_every N`
  training flag is a clean null across all 5 seeds of the standard 2-city config** (§41/§42,
  |diff|/SE ≈ 0.1-0.2) — not worth enabling by default, useful only as a targeted repair once a
  locked round is detected. A `pressure_norm` reward function was added and tested as an
  alternative to the default `diff-waiting-time` (§37/§38, single seed) — didn't help, and the same
  degenerate-lock-in signature reproduced under it too, evidence the lock-in isn't specific to the
  default reward design. **`agents/dqn.py`'s `_epsilon_action`/`act_batch` still only implement
  epsilon-greedy — the actual `federated/parallel_server.py`-level root cause of *why* aggregation
  produces this lock-in (§28's original question) is still open.** New reusable diagnostics from
  this stretch: `diagnostics/weight_divergence.py`, `diagnostics/reeval_checkpoint.py` (supports
  `--temperature` for softmax eval), `diagnostics/recovery_finetune.py`.
- **Also from this stretch: the 2-city masked-head/neighbor-attention ablation reversed on more
  seeds (§30→§31) — a cautionary, not a settled, result.** Single-seed found clean-comm attention
  (C) beating both rule-based baselines on every measure and mean-pooling (D) badly underperforming
  no-neighbor-info (B); on 5 seeds neither claim survived (B/C/D pairwise indistinguishable on
  best-round, |diff|/SE ≤ 0.73) — another instance of this project's standing pattern (§11→§12 was
  the first) where a good single-seed story doesn't reproduce. The `--disable_head_fix` /
  `--disable_neighbor_attention` code split itself (decoupling aggregation-time masked-head
  averaging from network-time attention-vs-pooling, previously conflated) is real and committed.
- **Current best-known training config, unchanged: `--dueling --n_step 3`.** Validated on 5 seeds,
  2-city roster (§21): mean reward -2030.4 (std 515.0), no seed-outlier failure mode. Do not use
  `--fedprox_mu` (§14, no effect) or `--server_momentum` with `--dueling` (§18, net-negative — a
  hard CLI check blocks this). `--pseudo_grad_clip`/`--eval_ema_decay` implemented but unconvincing
  (§19). `--epsilon_reset_every` (new, §41/§42) is implemented and safe (0 = exact no-op) but a
  clean null in aggregate — don't turn it on as a default, it's a targeted-repair tool only.
- **Phase 1's masked-head ablation across roster sizes (§20/§23), still the standing read:**
  mean-reward benefit shrinks monotonically with roster size, gone by 7 cities (|diff|/SE: 3.42 →
  0.71 → 0.23); best-round benefit real at every size but also shrinking in relative terms.
- **`fixed_time`/`max_pressure` rule-based baselines beat the trained DQN at every roster size now
  checked with a true holdout — 7-city since §24-§29/§32-§34, 2-city since §43.** The old "2-city
  best-round *does* beat both baselines" framing (§21/§29) only ever held on the `city_1`
  in-distribution fallback — see the §43 bullet above. Both roster sizes: DQN loses on mean and
  best-round once evaluated correctly.
- **NEXT ACTION — decision (1) below is now largely answered; decision (2) is still open:**
  (1) "does the trained DQN beat rule-based control" — **no, at any roster size, once evaluated on
  a true holdout (§43).** The mechanism behind the underlying instability is well-characterized
  (confidently-locked degenerate policy, §34) and several fixes were tested (softmax eval §36,
  recovery-finetune §39/§40, periodic-reset §41/§42, pressure reward §37/§38) but none is a clean,
  general win; the highest-leverage remaining thread is still §28's unanswered "why does federated
  aggregation itself produce this lock-in." Newly open sub-question from §43: does
  `--reward_shaping_wait_weight` (untested, 7-city pilot running as of this writeup) or a properly
  validated `--pad_to_true_holdout`-corrected multi-seed sweep change this picture at all, or is
  the gap simply too large for any tested intervention to close. (2) Phase 1's own decision-gate
  outcome is still mixed (2-city clean pass on the now-superseded in-distribution numbers, 7-city
  null mean-reward result) — per the plan's own instruction not to guess on an ambiguous gate, this
  needs a user call before scaling Phase 2 compute, and arguably needs re-litigating given §43.
- `analyse/run_concurrent_batch.sh` is the **default** way to run any multi-run batch (see
  "Common commands" above). §22 measured ~1.5x wall-clock speedup at `MAX_CONCURRENT=3` on
  2-city runs (each concurrent run individually slows ~60%, contention worsens over a run's
  duration). §23 found 7-city runs handle `MAX_CONCURRENT=2` fine despite §22's more conservative
  `MAX_CONCURRENT=1` assumption for that roster size (measured via `top`/`ps`: city workers are
  bursty, not steadily CPU-bound; RAM, not CPU, was the binding constraint at ~9GB/15.8GB with 2
  concurrent 7-city jobs) — don't assume `MAX_CONCURRENT=1` is required for 7-city, but watch RAM
  headroom if pushing higher. Depends on the `run_dir` PID-suffix fix in
  `experiments/federated_training.py::main()` (§22) — without it, concurrent launches within the
  same wall-clock second silently corrupt each other's output directories. **Host sleep during a
  long batch (§30, §42) freezes but does not kill a running job** — it resumes cleanly from
  wherever it left off once the machine wakes (confirmed twice now, hours-long gaps both times);
  don't assume a large wall-clock gap in a training log means the run needs restarting.
