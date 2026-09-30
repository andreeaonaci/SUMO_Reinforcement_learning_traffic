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

## RESUME HERE (as of 2026-09-26 — check this is still current before trusting it)

### DONE: Braun 2026 comparison at six seeds (§110, §110b) — nothing is running

Historical note below kept for how it was run.

Remaining queue as of 2026-09-27 17:00: `native` seeds 17/21/25 (ETA ≈ 22:00); `synthfb`
17/21/25 trained but not yet evaluated. Finish with
`baselines/braun/run_eval_braun.sh {grid4x4,cologne3,ingolstadt7}` then
`python3 baselines/braun/aggregate.py` (both skip-or-resume).

Run by a separate agent. Braun's code is pinned at commit `ea47985` (the snapshot his paper
cites), downloaded as a tarball to `/home/deea/external/GNN-Traffic-Signal-Control-ea47985...`
— **it has no LICENSE file, so it must never be copied into this repo**; adapters import from
that path. New files land in `baselines/braun/` and `results/braun/`, **uncommitted until
reviewed**. The design point that governs every comparison: **Braun discards the network's
signal program and synthesizes phases as every maximal conflict-free movement set**, a
different (usually larger) action space than RESCO, FRAP, max-pressure-as-we-run-it, or
`--phase_relational`. So the comparison needs his max-pressure over *his* synthesized phases
as a control, or learner and action space are confounded. Metrics must be simulator-level
(delay/trip/completion/wait via `eval_paper_metrics.py`'s computation) — our training reward
can't be produced by his code. Nothing else is running.

Everything below is committed and pushed to `origin/next_phases`. **The main checkout may be
behind — `git pull` first.** All work since 2026-09-09 lives in the worktree branch
`worktree-rescofull-writeup`, already merged into `origin/next_phases`.

### READ FIRST: concurrent work exists, found 2026-09-23

**Braun, "A graph-based control interface for traffic signals on heterogeneous road
networks," arXiv:2607.21831, July 2026.** A shared graph network scores individual
movements; each junction converts those scores into its own variable-sized legal
phase set through a deterministic incidence matrix, so parameter shapes are
independent of junction action count. **That is the same structural commitment as
`--phase_relational`, reached independently, two months before we found it.**

It does not scoop the contribution — different learner (policy gradient over a
movement graph vs. value-based per-phase descriptor), different construction, and
**no counterpart to §98's dead-row control**, which is what actually carries this
paper. But any submission on this topic that omits it looks negligent. Cited and
distinguished in Related Work as of `paper/main.tex`. Its reported sensitivity to a
**signal-coverage distribution shift independently corroborates §103b's coverage
account for FRAP**, which was flagged here as well-supported but unproven.

Also added and previously missing: X-Light (IJCAI 2024, cross-city TSC).

### THE HEADLINE, in one paragraph

The action representation is the binding constraint on cross-topology generalization.
`--phase_relational` (a movement-relational readout derived automatically from the simulator)
beats a positionally-indexed head by four orders of magnitude zero-shot on an unseen topology,
**confirmed at 6 seeds on five configurations**, and beats it in-distribution too. It also beats
`max_pressure` zero-shot. What it does NOT do is beat `max_pressure` in-distribution on every
scenario, and it is not a novel architecture — see the claim ledger.

### What was established 2026-09-09 → 09-16

| § | finding | rigor |
|---|---|---|
| 100 | phase-relational CONFIRMED on the fully RESCO-exact roster | 6 seeds, \|diff\|/SE 23.6/30.9/53.8, 6/6 |
| 100b | in-distribution it does NOT beat `max_pressure`: ties Cologne, loses Ingolstadt. Raw Cologne mean was a **survivorship artifact** | 6 seeds |
| 101 | **prior-art review: the architecture family, zero-shot claim, federated setting and curriculum are ALL already published.** What survives is the *automatic configuration* property | no compute |
| 101b | **4th RESCO mismatch: our `ingolstadt7` is missing a green phase** RESCO's net has, at 1 of 7 intersections | measured |
| 102 | **PCFT does not help on the fixed readout — it hurts.** Plain FedAvg beats both PCFT arms 6/6. Curriculum itself a null | 6 seeds, 3.27/3.34 |
| 103 | **budget objection DEAD** (indexed still 4 orders behind at 20 rounds, 5.57/13.32, 6/6). **FRAP works** (independent confirmation of §98). **Phase beats FRAP on best-round** (4.06, 6/6) *while FRAP held the oracle config* | 6 seeds |
| 103b | RESCO validation in literature metrics, three readouts, done correctly | 6 seeds |
| 104 | training-topology diversity: **no measurable effect, but the test is SATURATED** — do not cite as a clean null | 3 seeds |
| 105 | **fine-tuning REVERSES on the phase-relational head** — 0/6 runs beat their own zero-shot (5.40/8.89). The corpus's LARGEST prior effect (72.78 on indexed) | 3 seeds, screen |
| 106 | ensemble **SPLITS**: majority vote ties its best member and beats the member mean (2.20 SE); SWA weight-average **collapses** to -3.87 | 1 group of 6, screen |
| 107 | **pre-submission audit: a FABRICATED citation, rule 1 broken in the abstract, and the concurrent work above.** Numbers themselves verified sound | re-derived from raw |
| 105b | **fine-tune reversal CONFIRMED at 6 seeds**: 1 round -0.123→-0.250 (4.44, 0/6); 2 rounds →-0.205 (3.75, 1/6, +0.02). The one improver is the WORST zero-shot seed; r(zero-shot, change) = -0.91 in the 2-round arm — deficit recovery visible inside one experiment | **6 seeds** |
| 109 | **beats `max_pressure` on WAITING at 3 s benchmark timing, 6/6 seeds (§109b: TIED on delay, 41.4 vs 40.2 s)** (98.9% vs 99.2%; wait 0.26 s vs 2.83 s). References on the 3 s holdout: mp -0.380, ft -2.730. Previously UNMEASURED at this timing. Indexed completes only 16-20% (gridlock) | **6 seeds** |
| 112b | **unseen RESCO networks, zero-shot**: phase-relational ~98% completion on cologne1/cologne8 vs indexed ~62%; ties `max_pressure` delay on cologne8 (waiting lower 6/6); ingolstadt21 (21 signals) is the limit, 77.6% vs mp 89.0% | **6 seeds** |
| 113c | **one-episode fine-tune on those networks: nothing significant**; helps ingolstadt21 (delay 5/6), neutral cologne8, hurts cologne1 (one collapsed seed) — same deficit-recovery ordering as §105b | 6 seeds, screen |
| 108 | **training-topology diversity does NOT help — CONFIRMED (§108b).** §104's saturation fixed on the congestion holdout; 0.58/0.20/1.60, div nominally *worse* on all three, ≤2/6 seeds favour it; detectable effect 0.049 vs a 1.082 range. "Hurts" is NOT claimable (crosses 2 only if base's worst seed is dropped) | **6 seeds** |

### The claim ledger — what can and cannot be said

**CAN claim (fully supported):** the action representation is the binding constraint (§98's
dead-rows control: indexed still gridlocks with every usable row fully trained); phase-relational
beats `max_pressure` zero-shot, 6 seeds × 5 configurations — **at the benchmark's 3 s timing on
WAITING time, 6/6 seeds at matched throughput (§109), but TIED on average delay (41.4 vs 40.2 s,
§109b) — always name the metric**; beats Braun 2026's released code on every RESCO scenario in
delay (§110, 3-seed screen, not RESCO-comparable for his synthesized-phase arm); training-topology diversity does not help
(§108b); the gap survives 2.5-4x budget (§103);
the gap holds in-distribution too, *while arriving more traffic*, so the true gap is wider (§103b);
phase-relational is far more stable (per-seed delay 19.7-23.0 vs indexed's 25.7-232.2); ~20 prior
interventions were floor effects — **now TWO measurements, not an inference** (§102 curriculum,
§105 fine-tuning — and §105 was the largest effect in the entire corpus); six evaluation
artifacts, each of which produced a plausible wrong number (§25, §95a, §95b, §99, §100b, §101b).

**CAN claim with the caveat in the same sentence:** "we require no per-intersection configuration
where the closest prior art does" — this is a **capability** claim; §103's best-round win over FRAP
is the parity evidence, but §103's final-round comparison does **not** clear the bar.

**CANNOT claim:** better than SOTA; a novel architecture (FRAP 2019, AttendLight 2020); novel
zero-shot transfer (MuJAM 2022, TransferLight Dec 2024); novel federated TSC or clustered
aggregation (HFRL 2025); PCFT as a contribution (ICCV 2023 owns client curricula, and §102 shows it
hurts); in-distribution competitiveness with `max_pressure`; any absolute Ingolstadt number without
§101b's missing-phase disclosure.

### Three rules that must survive

1. **Never report trip time or delay without `arrived` beside it.** `eval_paper_metrics.py`
   computes both over ARRIVED vehicles only, so stranding traffic *improves* them. This produced a
   wrong headline once already (§100b).
2. **A citation written from memory is a draft, not a reference.** §107 caught
   `TransferLight` attributed to an author who does not exist, `hfrl` with no authors
   at all, and two more with wrong author order — all written from memory in one
   pass. Verify authors and venue against the source before a bibliography is done,
   and drop volume/issue numbers you cannot verify rather than guessing.
3. **Never quote a number from `environments_c1_4_6` against RESCO.** That roster still carries
   §99's three mismatches. Only `environments_rescofull` is RESCO-exact. This is the error that
   retracted §59.

### New this stretch, and reusable

- **`--frap_head`** — MPLight's FRAP ported from RESCO's source as a baseline arm, with its
  competition mask asserted equal to their `build_comp_mask` verbatim. 60 tests.
- **`/priorart` skill** — the claim-level counterpart to `/benchmark`. Would have saved a week.
- `/lever`, `/launch`, `/runstatus` extended with this session's bugs (see their own docs).
- `environments_divwide`, `environments_wide_clean` — leak-free diversity rosters.
- `analyse/`: `run_resco_validation.sh`, `run_diversity.sh`, `run_rescofull_frap.sh`,
  `run_budget_sensitivity.sh`, `run_pcft_phase6.sh` — all skip-or-resume, safe to stop and relaunch.

### THE PAPER EXISTS — `paper/main.tex`, and it is the current deliverable

A complete IEEEtran draft, **9 pages** (restructured 2026-09-30 into standard IEEE order:
Introduction, Related Work, Problem Formulation, Method, Experimental Setup incl. Baselines and
Metrics, Results incl. §VI-F comparison with Braun at six seeds (`tab:braun`), §VI-G unseen RESCO networks
cologne1/cologne8/ingolstadt21 zero-shot (`tab:unseen`, §112b) and one-episode fine-tune
(`tab:unseen_ft`, §113c), Analysis incl. the
re-evaluation of the 41 interventions, Discussion and Limitations, Conclusion). The narrative
"Experimental Programme" / "Intervention Corpus" chapters were removed; their facts live in
Analysis. Keep new text in objective IEEE register, no story telling. 31 verified references,
compiles clean from the repo root or `paper/` (`pdflatex` twice, no undefined refs).
Title: *Phase-Relational Q-Learning: Configuration-Free Traffic Signal Control Across
Heterogeneous Intersection Topologies*. It carries the full experimental programme, three TikZ
system diagrams, the setup/provenance tables, the RESCO in-distribution comparisons, the
41-intervention inventory (the prose once said "thirty"; §107 addendum), the
evaluation-artifact section and the limitations. **Do not start a
new paper file** — extend this one.

Structural rules it already follows, which must be preserved:
- **`tab:config` is the provenance table** (refer to it by label — its number moved when tables
  were merged). Every results table has a row giving its roster, yellow interval and whether it may
  be set beside published numbers. **A new table is not finished until it has a `tab:config`
  row.** The column takes three values: yes, no, and *internal only* (benchmark timing, but a
  within-study comparison with no published counterpart).
- Only `tab:zeroshot` (its upper, 3 s block) and `tab:indist` are ever set against published
  figures. Everything else is fenced in its own caption as an internal comparison.
- Every delay/trip-time figure is accompanied by a completion percentage against the departure
  total, because the metric is computed over arrived vehicles only (rule 1 below).

### Paper decision, 2026-09-20: the 2s-yellow results STAY

User's call, asked and answered. Deleting all 2s content would have removed the
dead-rows control (§98 — the single strongest piece of evidence for the central
claim), the budget table (§103), the curriculum reversal (§102), the triple-demand
result (§97) and the entire 30-intervention inventory. They are kept and fenced
instead: `paper/main.tex`'s Table IV (`tab:config`) records the yellow interval and
external-comparability of **every** table in the paper, and each 2s table's own
caption repeats "internal comparison, not to be set against published numbers."
Rule 3 above is unaffected — fencing is what makes keeping them legitimate, and any
NEW table must be added to `tab:config` when it is added to the paper.

### NEXT, in priority order

1. ~~Re-run §104 on a scenario with headroom.~~ **DONE and CONFIRMED at 6 seeds, §108/§108b.**
   Diversity does not help (0.58/0.20/1.60, ≤2/6 seeds). Paper updated to the 6-seed numbers.
2. ~~Re-test §93's ensemble on the phase-relational head.~~ **DONE 2026-09-20, §106 — it SPLITS.**
   The majority vote survives but weaker than §93 (ties its best member at -0.12, beats the member
   mean by 2.20x member-level SE; its value is *selection* — best-member performance without
   needing to know which seed is best). The SWA weight-average **reverses hard**: -3.87, an order of
   magnitude below the *worst* member, because independently seeded runs sit in different loss
   basins and parameter averaging is only defined up to permutation symmetry. **This does NOT
   implicate FedAvg** (its clients are broadcast from a common point each round and never leave a
   shared basin) — say so explicitly anywhere §106 is cited, or a reader takes it as an indictment
   of the method. Screen: one group of six, the §70 trap. **Still open: replicate on a disjoint seed
   group** — §93 called for exactly this and it has never been run.
3. **The paper needs no new compute.** §101's positioning decision stands: write the mechanism +
   evaluation-artifacts paper, with phase-relational as constructive validation rather than the
   novelty claim. Do NOT chase a performance claim against TransferLight.
4. Open and untested: why phase-relational beats FRAP — §103 suggests *stability*, not capacity.
   §103b's pressure-coverage hypothesis (FRAP's results track its % of movements with downstream
   lanes across all four cities: 75/75/39/33) is well-supported but unproven.

---

---

## Strategic context (2026-08-27/29) — preserved because `fidings/` does NOT cover it

The historical RESUME HERE blocks through 2026-09-07 were removed from this file on
2026-09-26; their §-numbered experimental detail lives in
`fidings/divergence_investigation.md` (§1–§108), which is the source of truth. The one
thing that was *not* mirrored there is reproduced verbatim below — it is the
publishability / don't-start-over discussion, and a check against `fidings/` confirmed
the "start over" verdict and its reasoning appear nowhere else.

**STRATEGIC CONTEXT from the 2026-08-27/29 session (paper-worthiness discussion + the
clustered-federation decision rule) — read this first, it's not captured anywhere else and won't
survive if only the experimental sections below are read.**

The user asked directly whether this project is worth publishing and whether it's worth continuing
vs. starting over in a fresh repo. Answers given, for continuity if this thread is lost:
- **Publishability verdict:** not yet paper-ready as of §57 (no resolved mechanism, no
  reconciliation with RESCO's own numbers), but §58-§61's separation of training-budget vs.
  cross-topology-generalization effects meaningfully improved the story. The realistic framing for
  a paper is **not** "federated DQN traffic control fails" (disproven by §59) — it's "in-distribution
  this approach is competitive with published numbers once budget/protocol confounds are controlled
  for; cross-topology generalization has a real, characterized, partially budget-resistant gap;
  here is a rigorous mechanism investigation (confident lock-in, §32-34/§51-57) of the instability
  underneath it." That is judged a legitimate, citable contribution (methodology bug-finding +
  mechanism + budget/generalization separation) even if the reward number never goes positive — same
  category as RESCO's own paper, whose headline finding is also "published methods underperform
  simple baselines in realistic scenarios."
- **"Is it worth continuing, or is everything garbage, start over in a new repo?" — explicit verdict:
  not garbage, do not start over.** Reasoning on record: the infrastructure is audited-correct
  (Phase 0), §59 proves the pipeline works in-distribution (matches RESCO's own published numbers),
  and every hard-won bug fix (holdout fallback, `fixed_time`, Adam/weight-decay on masked heads,
  `run_dir` collision, `--resume` LR-reset) would need to be rediscovered from scratch elsewhere,
  while the actual unresolved problem (cross-topology generalization) is a research problem that
  would follow to any new repo doing similar RL, not a defect specific to this code.
- **Agreed plan: one more bounded, targeted experiment (clustered federation) with an explicit
  stopping rule, not open-ended fishing.** `ClusteredFedAvgStrategy` already exists in this
  codebase and directly targets the actual named problem (extreme topology heterogeneity forced
  into one shared policy) — cheapest remaining lever. **Turned out NOT to be "no new code needed"
  after all — see §65: a real bug (`ClusteredFedAvgStrategy` never actually clustered by genuine
  per-city differences, silently degenerated to an arbitrary alphabetical split) was found and
  fixed before launching, or this whole experiment would have tested the wrong thing.** **Decision
  rule, agreed with the user: if this does not show a real improvement (this project's own
  |diff|/SE ≥ 2 bar, once done at proper multi-seed rigor) over plain FedAvg on the true holdout,
  that is the signal to stop chasing reward-improving interventions and pivot fully to writing up
  the characterized-gap paper described above — not a reason to try yet another lever
  indefinitely.**
- **Why `environments_c1_4` (2-city) can't test this and `environments_c1_4_6` (3-city) is used
  instead:** `ClusteredFedAvgStrategy` clusters cities by `action_dim` via `federated/clustering.py
  ::cluster_cities` (deterministic sorted bucketing). Verified directly: with exactly 2 cities and
  `n_clusters=2`, clustering always degenerates to one city per cluster — identical to
  `--no_federation`, already tested (§49/§50/§64) with a null result. With 3 cities
  (`environments_c1_4_6` = arterial4x4/`city_1`, cologne3/`city_4`, ingolstadt7/`city_6`,
  action_dims 5/4/3 respectively), `n_clusters=2` produces a genuine, non-trivial split
  (`city_4`+`city_6` cluster together, `city_1` alone — verified by calling `cluster_cities`
  directly) — this is the roster being used for the pilot below.
- **Prior data point, not yet at proper rigor:** `clustered_fedavg` was tested once before, on the
  7-city roster at the original 20-round budget (5 seeds): mean -6494.5, the *best* mean of every
  aggregation strategy tested there, but |diff|/SE=0.85 vs. plain `fedavg` — not significant. A
  lead, never followed up on with more budget or a different roster until now.
- **Host-sleep note (already established elsewhere in this file, repeated here since the user
  raised it directly this session):** if the machine sleeps mid-run, the training job freezes but
  does not die — it resumes cleanly once the machine wakes (confirmed multiple times: §30, §42,
  and implicitly by this session's own multi-hour unattended batches). This document (this file +
  `fidings/divergence_investigation.md`) is the durable record if the session itself doesn't
  survive; the training compute itself is not at risk from sleep either way.

---

*Removed 2026-09-26: three `SUPERSEDED (kept for detail)` blocks, 57,569 chars. Recover
them from git history (`git show HEAD:CLAUDE.md`) or read `fidings/divergence_investigation.md`.*
