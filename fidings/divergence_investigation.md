# Federated Training Divergence — Investigation Findings

Log of what we tested, what broke, what we fixed, and what's still open. Numbers below are
pulled directly from `results/**/federated_history.json` and `training.log` for the runs named —
re-derivable from those files if anything here looks off.

## TL;DR

- A real divergence bug existed (unnormalized occupancy feature, plain Adam decaying masked-out
  Q-head rows, a masked-aggregation fallback that used the wrong reference weights). Fixed in
  `dfadab5` ("Fix training divergence sources...").
- Post-fix, **one** 20-round run improved substantially (waiting time 1632s → 609s). **Another**
  20-round run with the *exact same code, config, and seed* stayed flat/bad the whole time. Root
  cause found and fixed 2026-08-04 — see [§5](#5-root-cause-of-the-run-to-run-non-determinism-parallel-workers-were-never-seeded).
- **`--disable_head_fix` has never worked under `--parallel`** (the path every real training run
  uses) until fixed 2026-08-05 — see
  [§10](#10-9s-own-follow-up-found-a-real-bug---disable_head_fix-never-worked-under---parallel).
  Every past "fix off" ablation (§1, §8) actually ran with masked-head aggregation still on;
  their conclusions about the fix aren't evidence about the fix. Rerun correctly in §11 (3 seeds,
  looked like a clean win) then §12 (5 seeds, 2 more break that story) — **net result: ambiguous,
  not a clean win.** Masked-head aggregation is confirmed (via direct weight-delta measurement) to
  let untouched-by-some-clients rows move ~2.2-2.3x faster than naive averaging, which raises
  variance in both directions — usually a much better achievable peak, but seed4 is an outright
  failure that naive averaging never produces. Mean reward: not statistically distinguishable
  between conditions (5 seeds).
- **Seed4's failure minimally reproduced — see [§13](#13-minimal-reproduction-of-seed4s-failure-2-cities-is-already-enough).**
  `city_1` alone (same seed): stable, near-optimal. `city_1`+`city_4` (2 cities, the minimum
  possible federation): severe instability immediately, worse than the original 3-city case.
  Confirms the failure is federation/client-drift itself, not `city_1`'s own training or a
  specific 3-city combination — direct experimental support for the FedProx diagnosis already on
  the `new_ideeas` list, now the top-priority item there.
- **Phase 1 masked-head ablation redone on the validated `dueling+n_step` config, three roster
  sizes, 2026-08-11/12 — see
  [§20](#20-phase-1-masked-head-ablation-redone-on-duelingn_step-three-roster-sizes-the-fixs-benefit-shrinks-as-the-roster-grows).**
  First-ever clean, statistically unambiguous mean-reward win for the fix, on 2-city (5 seeds,
  |diff|/SE=3.42). 3-city reproduces the older "peak wins, mean ambiguous" pattern. 7-city (only
  2 seeds so far) looks unclear-to-negative — needs more seeds before trusting. The fix's benefit
  appears to shrink as roster size grows; a real, reportable pattern either way it resolves.
- **FedProx tested, 2026-08-06 — negative result, see [§14](#14-fedprox-swept-across-mu-and-a-3rd-city-no-stabilizing-effect).**
  Swept `mu` in {0, 0.01, 0.03, 0.1} on the `city_1`+`city_4` repro, plus `mu=0.01` on the 3-city
  roster, all seed 4. No mu tested reduces the oscillation; `mu=0.1` is measurably worse on every
  metric (mean/best/worst); `mu=0.01` on 3 cities reproduces the undamped failure almost exactly.
  FedProx is not the fix for this instability. Dueling network head (next item on `new_ideeas`) is
  now the priority follow-up.
- Across every post-fix run, `city_1` (the single-intersection city) has the worst, most
  persistent local loss growth of any city in the roster. Swapped it to a bigger, structurally
  distinct topology (`RESCO/arterial4x4`) this session. Validated with a real 10-round training
  run 2026-08-04 (`no_federation_vs_federated_comparison.md`) — city_1's new map is not a
  training-time outlier.

## Timeline

| date | commit / event | what happened |
|---|---|---|
| 2026-06-17 → 07-26 | `589275f` … `9ab1329` | Initial working pipeline, Phase 0 sweep completed |
| 2026-07-30 | `3a269ce` "strange divergence, trying with local episodes=4" | Divergence first noticed and named |
| 2026-07-30/31 | Phase 1 ablation (`fedavg_with_fix_seed1` / `fedavg_without_fix_seed1`) | Masked-head-aggregation ablation, seed 1 only |
| 2026-08-01 14:12 | `dfadab5` "Fix training divergence sources, add perf wins, remove dead code" | Root-cause fixes (see below) |
| 2026-08-01 18:37 | `ed2ae4d` merge PR #2 | Fix merged into `next_phases` |
| 2026-08-01 18:43 | `run_2026_08_01-18_43_01` | Post-fix 20-round run — **did not improve** |
| 2026-08-02 02:38 | `run_2026_08_02-02_38_41` | Post-fix 20-round run, same config/seed — **improved substantially** |
| 2026-08-02 (this session) | city_1 topology swap | Config change only, not yet trained |

## 1. Masked-head aggregation ablation (Phase 1, seed 1)

`masked_head_weighted_average` only averages the Q-head rows a city actually touched that round,
instead of blending in untouched rows from other cities' locally-drifted copies.
`fedavg_without_fix_seed1` ran with `--disable_head_fix` (plain full-head averaging) as the
ablation control. 7-city roster, 10 rounds, `local_episodes=2`, `lr=3e-4`.

| run | reward (round 1 → 10, best) | waiting_time (round 1 → 10, best) |
|---|---|---|
| `fedavg_with_fix_seed1` (masked-head ON) | -10247 → -9782 (best -3457) | 1967s → 1660s (best 947s) |
| `fedavg_without_fix_seed1` (masked-head OFF) | -9759 → **-7371** (best -6992) | 1913s → **1517s** (best 1475s) |

**Honest caveat:** on this single seed, masked-head aggregation did *not* come out ahead — the
ablation (`without_fix`) actually finished with better reward and waiting time. The ablation
script (`analyse/run_phase1_ablation.sh`) is written for 5 seeds per variant; only seed 1
completed for both. One seed is not enough to conclude masked-head aggregation hurts — it's
equally likely this seed is noise — but it means **the masked-head aggregation win is not yet
demonstrated**, only implemented and reasoned about from first principles. Seeds 2-5 need to run
before trusting either direction.

Rule-based reference points from the same holdout city:
- `baseline_fixed_time` (always phase 0): reward -9996, waiting_time 1900s
- `baseline_max_pressure`: reward -0.34, waiting_time 2.9s — **implausible relative to every
  other number in this doc**, almost certainly an artifact of the baseline-controller eval path
  (e.g. degenerate/near-empty scenario or a units mismatch), not a real max-pressure result.
  Flagging rather than trusting it; needs its own investigation before citing.

## 2. Divergence fix (`dfadab5`, 2026-08-01)

Three real bugs found tracing the pipeline end to end:

1. **Unnormalized occupancy feature.** `LaneEncoder.FEATURES` fed raw `traci` occupancy (0-100)
   into the network unscaled, next to every other feature normalized to ~0-1 — a ~100x scale
   outlier on one input dimension, silently regressed relative to `environments/cologne.py`
   which already divided by 100.
2. **Plain Adam + weight_decay on masked-out Q-head rows.** Adam folds `weight_decay` into the
   gradient *before* the moment estimates. A Q-head row for an action a low-action-count city
   never takes gets ~zero true gradient every step, so Adam's running estimates lock onto the
   `weight_decay*param` term as a near-constant signal — the effective update collapses to
   `-lr*sign(param)`, decaying that weight at a rate set by the *learning rate*, not the tiny
   `weight_decay` coefficient. Hundreds of optimizer steps per local round erased untouched
   action rows. Fixed by switching to `AdamW` (decoupled decay).
3. **Wrong fallback reference in masked-head aggregation.** When no client touched a given
   action row, the fallback used `state_dicts[0]` (an arbitrary client's locally-drifted copy)
   instead of the actual previous global weights the docstring claimed. Threaded
   `previous_global_state` through `aggregate_round` / `masked_head_weighted_average` and both
   callers (`federated/server.py`, `federated/parallel_server.py`) so untouched rows genuinely
   stay frozen instead of drifting toward whichever client happened to be first in the list.

Plus perf wins (per-tick lane-extraction cache, conditional reset sleep only on the socket/traci
path) and dead-code removal (`environments/cologne.py`, `environments/federated_preprocessing.py`
— both unimportable/unreferenced).

## 3. Post-fix verification: two runs, same everything, different outcomes

Both runs: `next_phases` @ `ed2ae4d` (post-merge), `base_dir=environments`, `aggregation_strategy=fedavg`,
`disable_head_fix=False`, `lr=3e-4`, `lr_decay=0.97`, `min_lr=1e-5`, `local_episodes=2`,
`parallel=True`, `eval_episodes=3`, `rounds=20`, **`seed=42`** — confirmed identical from each
run's logged argparse `Namespace`.

| | `run_2026_08_01-18_43_01` | `run_2026_08_02-02_38_41` |
|---|---|---|
| reward round 1 | -9082 | -9277 |
| reward round 20 | -8106 | **-2776** |
| reward trend | flat/noisy the whole run (-8100 to -10600) | net improvement, ends near best |
| waiting_time round 1 | 1781s | 1632s |
| waiting_time round 20 | 1618s | **609s (best of the run)** |
| waiting_time trend | stays 1500-1900s throughout, never breaks out | drops steadily, best value at the very end |

**The 18:43 run never learns; the 02:38 run learns cleanly** — with nothing different in code,
config, or seed between them. See [Open questions](#open-questions--next-steps).

### Per-city loss shape (from the improved run, `run_2026_08_02-02_38_41`)

Every city's local loss rises through the early/mid rounds, peaks around round 9-11, then
declines or plateaus — the signature of epsilon/LR decay settling out noisy early training,
*not* runaway divergence:

| city | round 1 | peak (round) | round 20 |
|---|---|---|---|
| city_1 (single-intersection) | 0.95 | 3.75 (13) | 3.79 — plateaus high, never really comes down |
| city_2 (3x3grid) | 0.99 | 2.24 (11) | 1.77 |
| city_3 (4x4-Lucas) | 0.006 | 0.026 (13/20) | 0.018 — negligible throughout |
| city_4 (RESCO cologne3) | 0.78 | 1.14 (11) | 0.57 |
| city_6 (RESCO ingolstadt7) | 0.046 | 0.15 (10) | 0.077 |
| city_7 (RESCO grid4x4 dense) | 0.055 | 0.75 (11) | 0.54 |

city_1 stands out: highest absolute loss of any city by a wide margin, and the only one whose
loss doesn't meaningfully come back down by round 20 (it plateaus in a 3.6-3.9 band instead of
continuing to climb, so it's *capped*, but never recovers the way every other city does).

## 4. city_1 topology swap (this session)

**Hypothesis:** city_1 uses `2way-single-intersection` — 1 traffic light, 720 steps/round vs.
thousands for the grid/RESCO cities — so its local training does far fewer, higher-variance
gradient updates per round while getting aggregated with the same weight as 16-intersection
cities. Its outsized, non-recovering loss is consistent with this small-sample-size effect.

**Change made:** `environments/city_1/config.yaml` net/route swapped from
`2way-single-intersection` to `sumo_rl/nets/RESCO/arterial4x4` (16 traffic-light-controlled
intersections, ~2500-vehicle route file, 0-2992s departure window — fits the existing 3600s
`num_seconds` with no shifting needed).

**Why this net and not another:** surveyed every unused net bundled in the repo
(`2x2grid`, `4x4loop`, `Nguyen`, `OW`, `RESCO/{cologne1,cologne8,ingolstadt1,ingolstadt21,arterial4x4}`)
against the existing roster to avoid duplicating structure already covered:

| already in roster | structure |
|---|---|
| city_2 `3x3grid`, city_3 `4x4-Lucas`, city_5_holdout/city_7 `grid4x4` | regular grid |
| city_4 `cologne3`, city_6 `ingolstadt7` | real-world extract |

`arterial4x4` is a synthetic arterial-corridor layout (irregular node naming, not a rectangular
grid) — distinct from every net already in use, same "well-known RESCO benchmark" provenance as
the cities already in the roster, and the size class the swap was aimed at (16 intersections,
matching the "bigger than 2x2, 4x4-ish" ask).

**Caveat on roster balance:** city_1/city_3/city_5_holdout/city_7 are now all 16-intersection
cities (different topologies, same size). If size diversity matters as much as topology
diversity, `RESCO/cologne8` (8 TLS, real-world) or the Nguyen network (8 TLS, classic academic
test net) are the other unused, structurally-distinct options sitting in the repo.

**Status:** config changed, smoke-tested through `environments.federated_env.build_federated_env`
(the actual factory training uses) — 16 intersections detected, `own`/`neighbor`/`action_mask`
shapes match the shared contract (`own: (115,)`, `neighbors: (8,3)`, `action_mask: (5,)`),
non-zero rewards flowing by step 20. **No actual training run has used this config yet** — the
20-round run analyzed in §3 above started before this swap and used the old single-intersection
city_1 the whole time (confirmed: its action_counts never exceed 4 actions, the old city's phase
count, never arterial4x4's 5).

## 5. Root cause of the run-to-run non-determinism: parallel workers were never seeded

**Diagnosed and fixed 2026-08-04**, prompted by the same noise showing up again in a fresh 10-round
A/B comparison (`no_federation_vs_federated_comparison.md`) — B (federated) was much noisier
round-to-round than A (std 925 vs 370), and neither run showed a clean trend, so the open question
from §3/#1 above got picked back up.

**Root cause:** `ParallelFederatedServer` starts one worker per city via
`mp.get_context("spawn")` (`federated/parallel_server.py`). `spawn` launches a **brand-new Python
interpreter** per worker — unlike `fork`, it does not inherit the parent process's memory or RNG
state. `set_seed(args.seed)` (`experiments/federated_training.py:389`) runs exactly once, in the
main process, *before* the workers are spawned — so it never reaches them. Every worker's:

- epsilon-greedy exploration (`agents/dqn.py:228,232` — `random.random()`, `random.choice()`)
- replay-buffer minibatch sampling (`agents/dqn.py:42` — `random.sample()`)
- greedy-action tie-breaking (`agents/dqn.py:207` — `np.random.choice()`)
- training-side comm-dropout pattern (`CommDropoutWrapper` built in `_client_worker` with no
  `seed` key in `DEFAULT_COMM_DROPOUT`, so it defaulted to `seed=None` → OS entropy)

...was drawing from each worker's own **unseeded, OS-entropy-initialized** global `random`/
`numpy.random` state — different every process launch, regardless of `--seed`. This fully explains
§3: identical code/config/`--seed 42` can and did produce completely different training
trajectories, because the actual stochastic decisions during training were never under seed
control in the first place. (The sequential path, `federated/server.py`/`federated/client.py`,
runs in the *same* process as `set_seed()`, so it was never affected — this bug is specific to
`--parallel`, which is the path used for all real training runs per `CLAUDE.md`.) SUMO's own
traffic randomness was already correctly pinned via each city's `sumo_seed:` config key — not
part of the bug.

**Fix (`federated/parallel_server.py`, `experiments/federated_training.py`):** `_client_worker`
now seeds `random`, `numpy`, and `torch` explicitly at process start; `ParallelFederatedServer`
takes a `seed` param and derives a distinct-but-deterministic per-city seed (`seed + city_index`)
so cities don't all explore identically; the derived seed is also threaded into the training-side
`CommDropoutWrapper`. `args.seed` is now passed through from `federated_training.py`.

**Verified with a smoke test**, not just by inspection: built a temporary 2-city roster
(`city_4`/cologne3, `city_6`/ingolstadt7, `city_5_holdout`/grid4x4, `num_seconds` shortened to 300
for speed), ran the same `--seed 123`, `--rounds 2` config twice. Before this fix, two such runs
were guaranteed to differ (per the mechanism above). After the fix: **identical
`federated_history.json` eval metrics round-by-round, and byte-identical `global_round_002.pth`
checkpoints** (`torch.equal` true on every tensor) — full determinism confirmed under `--parallel`.

## 6. First real multi-seed run post-fix: loss is clean, holdout reward is noisy (confounded)

**2026-08-04, same session as §5.** Cheap validation of the seeding fix: 3 seeds (1, 2, 3), 20
rounds each, `--parallel`, plain `fedavg`, on a deliberately small 2-city roster
(`environments_multiseed/`: `city_4`/cologne3 [3 intersections] + `city_6`/ingolstadt7 [7
intersections]) — chosen as the two cheapest cities in the roster so 3 seeds x 20 rounds is
affordable. `city_5_holdout` (grid4x4, 16 intersections) picked up via the existing base_dir
fallback for eval, same as always. Full round-by-round numbers in
`results/run_2026_08_04-{14_15_27,14_57_31,15_22_52}/federated_history.json` (seeds 1/2/3
respectively).

**Holdout reward: still noisy, no clean trend in any of the 3 seeds** (round-20 reward: seed1
-644.9, seed2 -576.9, seed3 -2.3; std across rounds 277-518 depending on seed). At first glance
this looks like the same unresolved problem as §3 — but there's a real confound this time: the
model was trained *only* on two small cities (3 and 7 intersections) and evaluated on a
16-intersection holdout it never saw anything structurally similar to during training. That
mismatch alone is enough to produce erratic holdout behavior independent of whether local
training itself is stable. This setup wasn't built to isolate that — worth fixing before reading
too much into the holdout numbers here (see open question below).

**Per-city local training loss (the metric Phase 0's gate actually asks about, and not subject to
the holdout-mismatch confound): clean and consistent across all 3 seeds.**
- `city_4`: monotonic-ish downward trend in every seed — round 1 loss 0.60-0.84, round 20 loss
  0.42-0.46, roughly halving over the run, same shape in all three seeds.
- `city_6`: stays bounded and low throughout (0.03-0.07) in every seed, no blow-up, no runaway
  growth, mild wobble but no divergence.

No sign of the city_1-style "high loss, never recovers" pattern from §3 in either city, in any
seed. This is a meaningfully better result than anything pre-fix: three *different* seeds now
produce three *differently-detailed-but-qualitatively-matching* loss curves — exactly what
"reproducible but not identical" should look like, as opposed to §3's "identical seed, wildly
different outcome."

**Reading:** the core training-stability question Phase 0 asks (does local loss trend down /
stay bounded) looks genuinely healthy and reproducible now. The holdout-reward volatility is real
but likely dominated by the scale mismatch in this particular cheap setup, not by leftover
training instability — needs a same-scale-ish holdout (or accepting this roster can only speak to
loss, not holdout reward) before treating it as a second open problem.

## 7. Speed optimizations, then the real test: city_1 (the hardest case) revalidated post-fix

**2026-08-04, same session.** §6 deliberately tested the two *easiest* cities (city_4, city_6 —
neither had ever shown instability, even pre-fix). Before spending more compute redoing that with
`city_1` — the city with the worst, most persistent divergence in every prior run (§3) — added two
speed optimizations, both verified risk-free before trusting them on a real run:

- **Torch thread pinning** (`federated/parallel_server.py`): each city worker now runs with
  `torch.set_num_threads(1)`. Previously every worker defaulted to 6 intra-op threads; with 6-7
  cities training concurrently on a 12-core box that's up to 36-42 threads fighting over 12 cores.
  Each worker is already its own OS process — the thread pool was pure redundant oversubscription
  on top of that, not additional real parallelism.
- **Batched per-tick action selection** (`agents/dqn.py`): `train()`'s action-selection loop
  previously called the network once per intersection per tick (`agents/dqn.py`'s old
  `{ts_id: self.act(o, ...) for ts_id, o in obs_dict.items()}`) — 16 separate batch-1 forward
  passes per tick for a 16-intersection city. New `act_batch()`/`_greedy_action_batch()` batch the
  network forward pass across every intersection in one call (batch size = however many
  intersections that city has that tick, not a fixed constant), while preserving the *exact* same
  per-intersection RNG draw order/semantics (explore-vs-greedy still decided independently per
  intersection from the same two RNG streams) — verified genuinely **byte-identical** output
  (not just close) on a city_4+city_6, `--seed 777`, 2-round before/after comparison:
  `federated_history.json` eval metrics matched exactly and `global_round_002.pth` checkpoints
  were `torch.equal`-identical on every tensor. `optimize()`'s replay-buffer training step was
  already properly batched — this only affects action selection.

**The real test**: `city_1` (`arterial4x4`, 16 intersections) + `city_4` + `city_6`, 3 seeds
(1/2/3), 20 rounds, `--parallel`, plain `fedavg`. `city_1` being 16 intersections (same scale as
the holdout) also removes §6's scale-mismatch confound.

**`city_1`'s loss — the actual thing being re-tested — is now well-behaved in all 3 seeds:**

| seed | round 1 | round 20 | range across all 20 rounds |
|---|---|---|---|
| 1 | 0.49 | 0.33 | 0.24 – 0.63 |
| 2 | 0.42 | 0.17 | 0.16 – 0.42 (clean downward trend, more than halved) |
| 3 | 0.42 | 0.38 | 0.25 – 0.42 |

Compare to §3's pre-fix `city_1`: peaked at 3.75 (round 13) and *never recovered*, plateauing at
3.6-3.9 through round 20 — roughly **an order of magnitude worse** than anything seen here.
`city_4` and `city_6` again show the same healthy, consistent-shape-across-seeds pattern as §6
(`city_4`: 0.57-0.83 → 0.43-0.47, downward in all 3 seeds; `city_6`: bounded 0.04-0.10 throughout).
No sign of the old city_1-specific failure mode in any seed.

**Holdout reward/waiting time: still noisy overall, but with a striking new pattern — all three
seeds independently converge to near-optimal waiting times in the same round window, then regress
at round 20:**

| seed | round 15 | round 16 | round 17 | round 18 | round 19 | round 20 |
|---|---|---|---|---|---|---|
| 1 waiting(s) | 145.9 | 664.9 | 1284.8 | **9.2** | 474.6 | 829.8 |
| 2 waiting(s) | 1990.2 | 1466.7 | **10.2** | **3.8** | **10.7** | 223.2 |
| 3 waiting(s) | 1601.2 | **10.1** | **10.0** | 13.3 | 255.8 | 1499.0 |

Every seed hits single-to-low-double-digit waiting time (i.e. a genuinely near-optimal signal
control policy) somewhere in rounds 16-19, then every seed regresses at round 20. This is not
random noise — three independent seeds landing on the same round window for both the good phase
and the round-20 regression is a real, reproducible pattern, not coincidence. Not yet explained;
candidates: `explore_fraction=0.5` means epsilon hits its floor around round 10 of 20, so rounds
16-19 are pure exploitation of an already-good policy, and something about the last round
specifically (LR near its floor, one more aggregation round, a target-network sync) may be
knocking the policy off a sharp optimum. Needs its own look before trusting round-20 numbers as
"final performance" in any future run of this length.

**Reading:** this is the actual Phase 0 hard-case test that was missing — city_1's specific,
long-standing failure mode is gone post-fix, confirmed across 3 seeds, not just 1. Combined with
§6, per-city training loss (Phase 0's core "does local training even work" question) now looks
solid across every city in the roster tested so far. The reward-variance side of Phase 0's gate
is more nuanced than "resolved" — there's clear evidence of real learning (the rounds 16-19
near-optimum) but also a new, specific, reproducible round-20 regression that wasn't visible
before because no prior run had reproducible seeds to compare against. That's arguably a much
better place to be than before (a narrow, analyzable pattern vs. total unpredictability), but it
means "reward variance shrinking over time" still isn't a clean yes.

## 8. Phase 1 masked-head ablation, redone properly: 3 seeds, post-fix, still ambiguous

**2026-08-04/05, same session as §7.** §1's ablation was 1 seed, pre-seeding-fix, 7-city roster.
Redone here with the seeding bug actually fixed: `city_1`+`city_4`+`city_6` (action_dim 5/4/3 —
real heterogeneity, the mechanism `masked_head_weighted_average` specifically targets), 3 seeds
(1/2/3), 20 rounds, plain `fedavg`, fix-on vs. `--disable_head_fix`. Fix-on reuses §7's runs
directly; fix-off is 3 new matched runs
(`results/run_2026_08_04-{21_38_15,22_26_24,23_14_29}`).

| seed | condition | mean reward | std reward | round 20 reward | best-round reward | mean wait(s) | round 20 wait(s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fix_on  | -2741.2 | 1463.4 | -2568.2 | -120.9 | 1311.7 | 829.8 |
| 1 | fix_off | -2051.2 | 1082.8 |  -634.4 | -541.9 | 1345.3 |  53.9 |
| 2 | fix_on  | -2575.9 | 1376.5 | -1135.2 |  -65.5 | 1217.5 | 223.2 |
| 2 | fix_off | -2568.6 | 1295.0 |  -313.5 | -133.0 | 1375.8 | 440.7 |
| 3 | fix_on  | -2198.7 | 1490.2 | -4080.8 |  -73.8 | 1411.3 | 1499.0 |
| 3 | fix_off | -2313.4 | 1089.2 | -2640.4 | -534.5 | 1414.9 | 1138.7 |

**No clean separation, but a real and consistent pattern in *how* the two conditions differ:**
- **Mean reward across all 20 rounds**: fix_on avg -2505.3, fix_off avg -2311.0 — a ~200-point
  gap that's small relative to either condition's own round-to-round std (~1100-1490). Not
  distinguishable from noise on 3 seeds.
- **Best single round**: fix_on hits a strikingly consistent, near-optimal best round in *every*
  seed (-120.9, -65.5, -73.8 — all within a tight band). fix_off's best rounds are more variable
  and never get as close to zero (-541.9, -133.0, -534.5). Fix-on reaches a better peak, reliably.
- **Round 20 (final)**: the opposite pattern — fix_off's last round is less bad than fix_on's in
  *all three* seeds (-634/-314/-2640 vs -2568/-1135/-4081). This lines up with §7's finding that
  fix_on's round-20 regression (the still-unexplained dip after the rounds-16-19 near-optimum) is
  itself part of what's being measured here — fix_off's smoother-but-lower-peak curve just doesn't
  have as far to fall.
- **Per-city training loss** (the actual mechanism under test): checked `city_1` (action_dim=5,
  most exposed to the head-row-masking effect on this roster) and `city_6` (action_dim=3, most
  masked-out rows relative to the shared 5-wide head) in detail — **no meaningful separation in
  either city, any seed.** Loss curves for fix_on and fix_off sit within each other's seed-to-seed
  spread, both bounded in the same ranges (`city_1`: ~0.16-0.63 either way; `city_6`: ~0.04-0.09
  either way).

**Reading, per the plan's own decision-gate categories: this is the "ambiguous / overlapping
distributions" case**, not a clean win or loss for the fix. Unlike §1's single-seed result (which
looked like a clean loss for the fix, but was one sample), 3 seeds now show the fix trading peak
performance for final-round performance rather than being strictly better or worse — a more
nuanced, more interesting result than either "fix wins" or "fix loses," but not yet a basis for a
paper claim. Per the plan: don't scale to Phase 2 on this; either more seeds, or investigate
whether the round-20 regression (§7) is entangled with this comparison (both conditions might be
riding the same underlying end-of-schedule effect, in which case comparing final-round numbers
between conditions is comparing two samples of the same unexplained artifact, not the head-fix
mechanism itself).

## 9. Digging into the round-20 regression: it's not generic noise, it's specific to the fix

**2026-08-05, same session.** §7 flagged a round-20 regression as unexplained; §8's fix-off data
(collected for an unrelated purpose) turned out to contain the control condition needed to test
whether it's generic FedAvg noise or something the masked-head fix itself causes. No new training
needed — this used data already on disk.

**Where does each condition's best round actually land?**

| seed | fix_on best round | fix_off best round |
|---|---|---|
| 1 | **18** (reward -120.9) | 3 (reward -541.9) |
| 2 | **17** (reward -65.5) | 18 (reward -133.0) |
| 3 | **16** (reward -73.8) | 13 (reward -534.5) |

**fix_on's best round clusters tightly at 16-18 in all three seeds — practically the same 3-round
window every time.** `fix_off`'s best round is scattered with no pattern (3, 13, 18). A generic
"training is just noisy" explanation predicts scattered best-rounds in *both* conditions, the same
way `fix_off` actually looks. Getting the *same narrow window* independently in 3 different seeds,
only when the fix is on, is a specific, reproducible effect of the fix itself, not a property of
FedAvg/replay-buffer training in general on this roster.

**Per-city training loss doesn't explain the eval swing.** Checked `city_1` at rounds 19→20 for
each `fix_on` seed specifically (where the regression happens): seed3's loss does jump alongside
its reward regression (0.287→0.380), but seed2's reward collapses from -71.9 to -1135.2 while
`city_1`'s loss barely moves (0.1717→0.1734), and neither `city_4` nor `city_6` show a
corresponding spike for seed2 either. **No single city's local loss reliably explains the
holdout-reward collapse** — the regression shows up in the *aggregated* model's behavior without
a matching spike in any individual client's local training signal.

**Working hypothesis, not yet confirmed:** `masked_head_weighted_average` deliberately does *not*
smooth every city's Q-head row toward a common average every round — only the rows a city actually
touched get updated by that city's contribution (that's the whole point of the fix, see §2). Naive
full-head averaging (`fix_off`) implicitly regularizes the shared head every round, pulling all
cities' head weights toward each other regardless of use, which trades away peak performance for
smoother, more scattered results. The masked-head fix removes that regularization, letting the
head specialize more sharply per city — plausibly enough to reach a genuinely better policy (the
16-18 peak) — but with less of a stabilizing pull back toward a shared consensus, so one round's
per-city local updates landing slightly differently in weight-space can knock the *aggregated*
model off that peak without any individual city's own loss curve showing distress. This is the
same client-drift-in-weight-space failure mode the plan's `new_ideeas` list already names FedProx
as a targeted fix for — this investigation is independent evidence for that diagnosis, not a new
guess.

**Practical implication for §8:** the round-20 comparison between `fix_on` and `fix_off` isn't
comparing two samples of the same generic artifact (ruling out one of §8's two proposed
explanations) — the regression is real and fix-specific. That doesn't resolve §8's "ambiguous"
verdict on its own (fix_on still trades peak for stability, which is a genuine tradeoff either way
this mechanism is framed), but it does mean the round-20 numbers in §8 are measuring something
mechanistically connected to the fix, not noise — worth stating as such rather than discounting
them.

**Not yet done:** confirming the hypothesis directly would mean inspecting the actual Q-head
weight deltas between rounds 18-20 (do the untouched-row values actually swing more under
masked-head aggregation than under naive averaging, as the hypothesis predicts?) rather than
inferring it from eval/loss curves alone.

## 10. §9's own follow-up found a real bug: `--disable_head_fix` never worked under `--parallel`

**2026-08-05, same session, immediately after §9.** Went to confirm §9's hypothesis directly by
loading `head.4.weight`/`head.4.bias` from consecutive round checkpoints and measuring per-row
delta norms for `fix_on` vs `fix_off` around rounds 16-20. Result: **row deltas were statistically
indistinguishable between the two conditions** (both ~0.03-0.09 across all rows, no systematic
gap) — not what §9's hypothesis predicted. That non-result was the tell.

**Root cause: `ParallelFederatedServer` (`federated/parallel_server.py`) never gated its
aggregation call on `head_fix` at all.** It accepted a `head_fix` constructor argument, but only
ever forwarded it into each city worker's local network construction (toggling
`NeighborAttentionQNetwork` between full attention and a simplified neighbor-mean-pooling
architecture — an unrelated architectural knob). The actual aggregation line in `run()` called
`masked_head_weighted_average(...)` **unconditionally**, with no `if self.head_fix` branch
anywhere in the file. The sequential path (`federated/server.py:323`,
`aggregate_round(..., use_masked_head=self.use_masked_head, ...)`) has always done this
correctly — the parallel path was simply missing the equivalent gate.

**Blast radius: every `--disable_head_fix` ablation ever run through `--parallel` — which is all
of them.** `analyse/run_phase1_ablation.sh` (the project's standard Phase 1 script) passes
`--parallel` (confirmed, line 54). §1's `fedavg_without_fix_seed1` and §8's entire 3-seed `fix_off`
condition both used `--parallel`. None of these ever actually ran naive full-head averaging —
`masked_head_weighted_average` was active in every single "fix off" run this project has produced.
The only real variable those ablations tested was attention-vs-pooling network architecture, not
the aggregation mechanism the flag claims to control. **§1 and §8's conclusions about the
masked-head fix's effect are not evidence about that fix at all** — they're an accidental
architecture ablation, mislabeled.

**Fix applied** (`federated/parallel_server.py`): store `self.head_fix = bool(head_fix)` in
`__init__`; the aggregation branch now does
`masked_head_weighted_average(...) if self.head_fix else weighted_average(...)`, mirroring the
sequential path exactly.

**Verified mechanically, not just by inspection**, before trusting it: 2-city roster (`city_4`
action_dim=4, `city_6` action_dim=3 — `city_6` never touches row 3), 1 round, same seed, once with
the fix on and once off.
- `fix_on`: aggregated `head.4.weight[3] == city_4`'s local row exactly (100% from the only city
  that touched it) — correct masked-head behavior.
- `fix_off`: aggregated `head.4.weight[3] == 0.5*(city_4's row + city_6's row)` exactly — correct
  naive full-head behavior, and now *actually different* from the fix-on result, which it never
  was before.

**Status:** the `fix_off` condition of §8's ablation is being rerun correctly now
(`city_1`+`city_4`+`city_6`, 3 seeds, 20 rounds, otherwise identical config) — `fix_on`'s existing
data from §7/§8 doesn't need rerunning, since `masked_head_weighted_average` being the unconditional
default meant `fix_on` runs were always correct. §8's table and its "ambiguous, peak-vs-stability
tradeoff" conclusion should be treated as **retracted pending the corrected rerun** — not because
the numbers were wrong, but because they answered a different question than the one asked.

## 11. Phase 1 ablation redone correctly: a real, mechanistically-confirmed effect

**2026-08-05, same session, immediately after §10's fix.** `fix_off` rerun completed (3 seeds, 20
rounds, `city_1`+`city_4`+`city_6`, aggregation now genuinely disabled per §10's fix).

| seed | condition | mean reward | std reward | best-round reward | best round |
|---|---|---:|---:|---:|---:|
| 1 | fix_on  | -2741.2 | 1463.4 |  -120.9 | 18 |
| 1 | fix_off | -2544.1 |  822.2 |  -971.5 | 11 |
| 2 | fix_on  | -2575.9 | 1376.5 |   -65.5 | 17 |
| 2 | fix_off | -2669.5 |  956.4 |  -724.4 | 19 |
| 3 | fix_on  | -2198.7 | 1490.2 |   -73.8 | 16 |
| 3 | fix_off | -1728.6 | 1097.0 |  -147.4 | 19 |

**Mean reward across all 20 rounds: still not cleanly separated** (fix_on avg -2505.3, fix_off avg
-2314.1 — a ~190-point gap, small next to either condition's own std). On this metric alone, still
"ambiguous" by the plan's own gate categories.

**Best-achievable-round reward: now cleanly, non-overlappingly separated in fix_on's favor.**
fix_on's best round in every seed falls in a tight band (-120.9 to -65.5). fix_off's best round in
every seed is markedly worse (-971.5 to -147.4) — **fix_off's best result across all 3 seeds
(-147.4) is still worse than fix_on's worst best-result (-120.9).** Zero overlap. This is a real
effect, not noise from 3 samples.

**Q-head weight-delta magnitudes (rounds 16-20, averaged across all 3 seeds and every transition)
now confirm the mechanism directly — this is the check §9 called for and §10 caught as invalid:**

| head row | who touches it | fix_on mean Δ-norm | fix_off mean Δ-norm | ratio |
|---|---|---:|---:|---:|
| 0-2 | all 3 cities | 0.036-0.037 | 0.031-0.033 | ~1.1-1.2x |
| 3 | `city_1` + `city_4` only | 0.058 | 0.025 | **2.33x** |
| 4 | `city_1` only | 0.058 | 0.026 | **2.20x** |

Exactly what §9's original (pre-correction) hypothesis predicted, now properly confirmed: rows
shared by every client move at roughly the same rate under both conditions (~1.1-1.2x), but rows
touched by fewer clients move **more than twice as fast** under masked-head aggregation — because
`fix_off` dilutes every round's real update with 1-2 other clients' stale, untouched copies of
that row, while `fix_on` lets the touching client's full signal through undamped. This directly
explains the whole pattern: `fix_on`'s specialized rows can converge faster to a better value
(explaining the sharp, tight 16-18 best-round peak) but are also more exposed to a noisy round's
update swinging the aggregate model since there's no cross-client averaging to damp it (explaining
the higher `std_reward` in every seed: 1463/1377/1490 for `fix_on` vs 822/956/1097 for `fix_off`).

**Reading:** this is a real, reproducible, mechanistically-understood tradeoff — not noise, and
not the same "ambiguous, don't know why" result as the invalid §8 comparison. Masked-head
aggregation lets the shared head specialize enough to reach a substantially better *achievable*
policy (confirmed 3/3 seeds, no overlap), at the cost of higher round-to-round variance (confirmed
3/3 seeds) driven by a directly-measured ~2.2-2.3x faster effective update rate on the exact rows
the fix targets. Whether this counts as "fix wins" per Phase 1's gate depends on which metric the
paper cares about: mean-reward says ambiguous, best-achievable-policy says a clear, understood win
for the fix. Per-city loss (`city_1` checked in detail) still shows no separation between
conditions in either direction — expected, since loss reflects local TD-error fit, not the
resulting policy's quality, and isn't the right lens for this particular question.

## 12. §11 partially retracted: 2 more seeds break the "clean win" story

**2026-08-05, same session.** Ran 2 more seeds (4, 5) on both conditions — 5 total each, matching
the ablation script's original design. §11's "best-achievable-round is cleanly, non-overlappingly
separated in `fix_on`'s favor" **does not hold up.**

| seed | fix_on mean reward | fix_on best (round) | fix_off mean reward | fix_off best (round) |
|---|---:|---:|---:|---:|
| 1 | -2741.2 | -120.9 (18) | -2544.1 | -971.5 (11) |
| 2 | -2575.9 |  -65.5 (17) | -2669.5 | -724.4 (19) |
| 3 | -2198.7 |  -73.8 (16) | -1728.6 | -147.4 (19) |
| 4 | **-3821.7** | **-1936.1 (1)** | -2758.6 | -546.5 (13) |
| 5 | -2158.0 |  -61.6 (5) | -2712.5 | -490.3 (20) |

`fix_on` seed4 is a genuine outlier: reward gets *worse* from round 1 (-1936) through its worst
point around round 9-10 (-5061), only partially recovers by round 20 (-3400), and never
approaches the near-optimal range seeds 1/2/3/5 reach — while `city_1`'s local training loss
stays completely unremarkable throughout (0.20-0.45, no divergence signal at all). This failure
mode is **invisible to loss monitoring**; only the holdout-reward curve shows it. seed5 also
breaks the earlier "best round clusters at 16-18" pattern — its best round is round 5, not the
16-19 window — confirming that clustering was a 3-seed coincidence, not a structural effect of
training progress.

**Statistics with 5 seeds:**
- Mean reward: `fix_on` avg -2699.1 (stdev 603.4) vs `fix_off` avg -2482.6 (stdev 383.7).
  Difference -216.4 against an approximate standard error of ~320 → **|diff|/SE ≈ 0.68, not
  distinguishable from noise.** Same conclusion as before, now on firmer statistical footing.
- Best-round: still favors `fix_on` on the **median** (-73.8 vs -546.5) — 4 of 5 `fix_on` seeds
  still beat every `fix_off` seed on this metric — but no longer *cleanly, non-overlappingly* so,
  because of the seed4 outlier.

**What's still true, and doesn't depend on how many seeds we ran:** the Q-head weight-delta
measurement in §11 (rows touched by fewer clients move ~2.2-2.3x faster under masked-head
aggregation than under naive averaging) is a structural fact about the two aggregation functions,
confirmed directly from the actual saved weights — not a claim about outcomes across seeds, so it
isn't affected by this correction. It explains *why* `fix_on` is higher-variance in both
directions (better peaks **and** a real chance of a seed like 4 that fails outright), which is a
more accurate summary than §11's "clear win."

**Corrected reading:** this is the plan's "ambiguous / overlapping distributions" case for Phase 1
— a real, mechanistically-understood tradeoff (higher ceiling, higher variance, occasional
outright failure) rather than a clean win or loss. Two new open items worth tracking separately
from the fix-vs-no-fix question: (a) `fix_on` seed4's failure mode — reward degrading over 20
rounds of apparently-healthy local training is itself concerning and worth its own look,
independent of the ablation; (b) this is exactly the "single/few seeds can mislead" lesson the
project has hit repeatedly (§1's single pre-fix seed, the original run-to-run non-determinism) —
worth being more conservative about trusting n=3 conclusions going forward, even post-fix.

## 13. Minimal reproduction of seed4's failure: 2 cities is already enough

**2026-08-05, same session, immediately after §12.** Isolated the minimum roster that reproduces
`fix_on` seed4's failure mode, by reusing exactly the per-city seed that produced it: in that run,
`--seed 4` on a 3-city roster gives `city_1` seed 4, `city_4` seed 5, `city_6` seed 6 (confirmed
from the `[parallel] city=... seed=...` log lines; `ParallelFederatedServer` derives
`city_seed = seed + city_index`, §5). Reran with matching seeds on smaller rosters.

**`city_1` completely alone (seed 4, no federation at all — a single client is a no-op for
aggregation):** stable and excellent. Converges to near-optimal by round 3 (reward -31.1, waiting
5.1s) and stays there for nearly the whole 20-round run (mostly -15 to -85 reward, ~2000-2140
vehicles arriving consistently). Loss trends cleanly downward (0.45 → 0.17-0.33). Round 1's local
loss (0.445468) matches the 3-city run's round-1 `city_1` loss exactly, confirming per-city seeding
is deterministic regardless of roster composition — so anything that diverges after round 1 is a
federation effect, not a fluke of which cities happen to be present.

**`city_1` + `city_4` (seeds 4/5 — the minimal possible federation, 2 clients):** severe,
sustained oscillation for the *entire* run, not a one-time dip. Swings repeatedly between
near-optimal (round 4: -27.3, round 6: -25.3, round 12: -29.3, round 17: -30.3) and catastrophic
(round 2: -2911.3, round 5: **-5369.7**, round 10: -4581.0, round 16: -4943.9) — worse amplitude
than the original 3-city failure. `city_1`'s local loss trends *upward* over the run (0.45 → 1.01
at round 19), unlike either the 1-city case (trends down) or the original 3-city case (stays flat
0.20-0.45) — a third, distinct loss signature for a third roster size.

**Conclusion: the instability requires federation, but not any particular combination or count of
cities beyond the minimum of two.** `city_1` in isolation is fine; the moment a second city enters
the picture at all, aggregation-driven instability appears immediately, and 2 cities produces
*more* violent swings than 3 did. This rules out "something specific about the city_1+city_4+city_6
combination" and confirms the mechanism is federation/aggregation itself — directly supporting the
client-drift-in-weight-space diagnosis from §11 (the `new_ideeas` list already names FedProx as a
targeted fix for exactly this failure mode). Practical upshot: `city_1`+`city_4` (2 cities, not 3)
is now the cheapest available test bed for any follow-up work on this specific instability — same
cost as before (`city_1` at 16 intersections is the expensive part either way) but genuinely
minimal, which matters for isolating causes cleanly.

## 14. FedProx swept across mu and a 3rd city: no stabilizing effect

**2026-08-06.** §13 minimally reproduced seed4's failure and named FedProx (the `mu` proximal-term
mechanism already implemented in `agents/dqn.py::DQNAgent` — `start_round()` snapshots the
round-start global weights, `optimize()` adds `mu/2 * ||w - w_global||^2` to the loss, wired up
via `--fedprox_mu`) as the top-priority candidate fix, on the theory that it directly caps the
client-drift-in-weight-space mechanism §11 confirmed via weight-delta measurement. Tested it here
for the first time — every prior run had `fedprox_mu=0.0` (a no-op).

**Setup:** same seed-4 repro as §13, `--parallel`, `fedavg`, `lr=3e-4`, 20 rounds,
`local_episodes=2`. Four `mu` values on the 2-city roster (`environments_c1_4`:
`city_1`+`city_4`), plus one `mu` value on the 3-city roster (`environments_c1_4_6`:
`city_1`+`city_4`+`city_6`) to check whether any stabilizing effect seen would generalize past the
minimal repro.

| roster | mu | mean reward | best round | worst round |
|---|---:|---:|---:|---:|
| 2-city | 0.0 (baseline) | -3095.3 | -168.6 | -5031.3 |
| 2-city | 0.01 | -3148.9 | -29.0 | -4818.7 |
| 2-city | 0.03 | -2810.6 | -49.7 | -4301.1 |
| 2-city | 0.1 | -3483.7 | -1866.0 | -5874.2 |
| 3-city | 0.0 (from §12) | -3821.7 | -1936.1 | — |
| 3-city | 0.01 | -3791.9 | -1912.6 | -4762.3 |

**No monotonic dose-response, no stabilization at any tested strength:**
- `mu=0.01` is statistically indistinguishable from `mu=0.0` on the 2-city roster (mean -3149 vs
  -3095, within round-to-round noise) — matches the theoretical expectation that 0.01 is a very
  weak pull toward the round-start weights, but shows the mechanism isn't doing anything even at a
  level that should be safe.
- `mu=0.03` is the best of the four (mean -2811), but still swings from -50 to -4301 across the
  same 20 rounds — nowhere close to resolving the oscillation, and the improvement over baseline
  is small relative to that same run's own round-to-round spread.
- `mu=0.1` is clearly *worse* on every metric (mean, best, and worst all worse than baseline) —
  the classic FedProx failure mode of the proximal term being strong enough to fight useful local
  adaptation, not just excess drift.
- **The 3-city generalization check is the clearest negative result.** `mu=0.01` on
  `city_1`+`city_4`+`city_6` reproduces the §12 `mu=0.0` failure almost exactly (mean -3791.9 vs
  -3821.7, best -1912.6 vs -1936.1) — if the proximal term were damping client drift at all, this
  is precisely the case that should have shown it, and it produced no detectable difference from
  doing nothing.

**Caveat:** all of the above is a single seed (4), deliberately the known-hardest case from §13 —
suggestive, not a multi-seed statistical result. But it's a *consistent* null across 5 runs and
two roster sizes with no cherry-picking, not a borderline single-sample call.

**Reading:** FedProx, as implemented, is not the fix for this instability. The code is harmless to
keep (`mu=0.0` is a true no-op, matching every run before this one) but should not be relied on or
cited as a stability fix. The instability itself is still open — see `new_ideeas`' next item
(dueling network head), which targets the same action-indexed-head symptom structurally (a value
stream that aggregates cleanly regardless of action_dim) rather than penalizing drift after the
fact.

## 15. Dueling network head: the first intervention that actually helps

**2026-08-06/07.** Implemented the dueling head from `new_ideeas` (`agents/networks.py`:
`NeighborAttentionQNetwork(dueling=True)` splits the final layer into `value_head` (scalar, no
action_mask, aggregates cleanly across every city regardless of `action_dim`) + `advantage_head`
(still action-indexed, still masked-head-aggregated same as the plain head), combined as
`Q = V + A - mean(A)`. Wired through `DQNAgent`/`--dueling`/both servers; masked-head aggregation
updated to target `advantage_head.weight`/`.bias` instead of `head.4.weight`/`.bias` when dueling
is on (falls back to plain averaging otherwise, per `masked_head_weighted_average`'s existing
"key not found" guard -- verified this doesn't silently happen by checking the key names directly
before running the real experiment). Same seed-4 repro as sec 14, both rosters.

| roster | condition | mean reward | best round | worst round |
|---|---|---:|---:|---:|
| 2-city | baseline (mu=0.0, sec 14) | -3095.3 | -168.6 | -5031.3 |
| 2-city | dueling | -2281.1 | -40.8 | -4693.8 |
| 3-city | baseline (mu=0.0, sec 12) | -3821.7 | -1936.1 | — |
| 3-city | dueling | -2469.4 | -244.2 | -4335.2 |

**Consistent, generalizing improvement on both mean and best-round, on both roster sizes** --
unlike FedProx (sec 14, no effect at any strength) this is a real, structural effect: ~26% better
mean and ~5x better best round on 2-city; ~35% better mean and ~8x better best round on 3-city.
Does not eliminate the oscillation (worst-case rounds are still bad, -4694/-4335), but shifts the
whole distribution up consistently rather than trading peak for stability the way masked-head
aggregation itself does (sec 11).

**Caveat:** the 3-city baseline (sec 12) used `eval_episodes=3`; every run in this session
(sec 14 sweep, this section, sec 16) used `eval_episodes=5` (the code's default, unchanged this
session) -- more eval episodes generally means a less noisy mean estimate, so that one comparison
has a minor apples-to-apples asterisk. The 2-city comparison has no such issue (both runs this
session, same protocol). The 3-city gap (-2469 vs -3822) is far too large to be explained by eval
noise alone.

**Reading:** the first intervention this session that's a clean, structural win rather than a
wash or a tradeoff. Worth keeping and building on -- see sec 17 for a natural next step
(combining this with server-side momentum, sec 16).

## 16. Server-side momentum (FedAvgM-style): modest, mixed benefit

**2026-08-06/07.** Implemented `federated/aggregation.py::apply_server_momentum` -- treats
`agg_state - global_state_before` as this round's pseudo-gradient and applies it through an
exponentially-weighted velocity buffer (`velocity = beta*velocity_prev + delta; new_state =
global_state_before + velocity`) instead of jumping straight to the raw aggregate every round.
`beta<=0` is an exact no-op (unit-verified). Targets a different symptom than dueling or FedProx:
the *aggregated* model swinging sharply round to round with no matching spike in any individual
client's local loss (sec 9), by damping the applied update itself at the server rather than
anything client-side or architectural. Tested `beta=0.9`, same seed-4 repro, both rosters.

| roster | condition | mean reward | best round | worst round |
|---|---|---:|---:|---:|
| 2-city | baseline | -3095.3 | -168.6 | -5031.3 |
| 2-city | momentum (0.9) | -2826.2 | -819.8 | -4344.7 |
| 3-city | baseline (sec 12) | -3821.7 | -1936.1 | — |
| 3-city | momentum (0.9) | -3434.7 | -933.4 | -5416.3 |

**Small, consistent mean improvement on both rosters (~9-10%), but a mixed effect on the
extremes.** 3-city best round improves ~2x (-933 vs -1936); 2-city best round gets *worse* (-820
vs -169) -- momentum damps the sharp near-optimal peaks along with the crashes, so it doesn't come
free. Worst-case on 3-city (-5416) is also worse than baseline, so this run's worst-case isn't
actually being damped there either, just the mean is pulled up a bit overall.

**Reading:** a real but much weaker effect than dueling (sec 15) -- worth keeping as an available
knob (default off, `beta=0` no-op) but not a standalone fix. This session's note: the 3-city run
took an unusual wall-clock span (2026-08-06 18:24 -> 2026-08-07 17:11) because the machine went to
sleep for ~19h between rounds 17 and 18 (confirmed from `training.log` timestamps) -- the process
itself resumed cleanly with no errors when the machine woke, this is a wall-clock artifact of the
host machine, not a training issue.

## 17. Where this leaves things: dueling wins, next candidate is combining it with momentum

Ranking of everything tested against the seed-4 repro this session: **dueling (sec 15) > momentum
(sec 16) > FedProx (sec 14, no effect) / naive full-head averaging (sec 11-13, trades peak for
stability but doesn't fix the underlying oscillation)**. Dueling and momentum target different
mechanisms (architecture vs. server-side update damping) and aren't mutually exclusive -- both
flags (`--dueling`, `--server_momentum`) can be set on the same run. Combining them is the
obvious next experiment: does momentum's mild worst-case damping stack with dueling's mean/best
improvement, or does damping dueling's now-larger, more-useful updates just eat back the gain
dueling provides on its own? Not yet run.

## 18. Dueling + momentum combined: net-negative interaction, confirmed on both rosters

**2026-08-07/09.** Direct test of §17's open question: `--dueling --server_momentum 0.9`
together, same seed-4 repro, both rosters (3-city leg interrupted by a ~2-day host-machine sleep
gap mid-run, same as sec 16 — resumed cleanly, no errors).

| roster | condition | mean reward | best round | worst round |
|---|---|---:|---:|---:|
| 2-city | baseline | -3095.3 | -168.6 | -5031.3 |
| 2-city | dueling alone (sec 15) | -2281.1 | -40.8 | -4693.8 |
| 2-city | momentum alone (sec 16) | -2826.2 | -819.8 | -4344.7 |
| 2-city | combined | -3163.7 | -1277.7 | -4621.7 |
| 3-city | baseline (sec 12) | -3821.7 | -1936.1 | — |
| 3-city | dueling alone (sec 15) | -2469.4 | -244.2 | -4335.2 |
| 3-city | momentum alone (sec 16) | -3434.7 | -933.4 | -5416.3 |
| 3-city | combined | -2777.7 | -1256.3 | -4160.4 |

**Consistent finding across both rosters: combined is always worse than dueling alone on mean and
best round.** On 2-city the interaction is unambiguously bad — combined loses to *both*
individual interventions on mean and best, and even to plain baseline. On 3-city the picture is
more nuanced (combined beats momentum-alone and baseline on mean, and has the mildest worst-case
of the three non-baseline conditions there), but the one comparison that holds in both cases is
the one that actually matters for picking a method: **dueling alone always beats
dueling+momentum, on both mean and best round, on both rosters.** Momentum never adds value on
top of dueling here, and on the smaller roster actively erases most of dueling's gain.

**Reading, confirms sec 17's hypothesis:** dueling's advantage comes from letting the
advantage-head rows move fast and undamped (same mechanism sec 11 identified for masked-head
aggregation generally, now shown to extend to the dueling architecture too) — server-side
momentum's whole design is to damp exactly that kind of fast movement, so stacking it on top of
dueling works against the thing that makes dueling effective. **Practical recommendation: use
`--dueling` alone. Do not combine it with `--server_momentum`** — at best neutral, at worst it
gives back most of dueling's improvement, with no case observed where it helps.

## 19. Three more `new_ideeas`, alone and combined with dueling: n-step is the new headline result

**2026-08-09/10.** Implemented the remaining `new_ideeas`/Feature-Development candidates —
n-step returns (`--n_step`, per-intersection sliding-window return accumulation in
`agents/dqn.py`, bootstraps with `gamma**n`), server-side pseudo-gradient clipping
(`--pseudo_grad_clip`, `federated/aggregation.py::clip_pseudo_gradient`, threshold 1.5 chosen
from real round-to-round delta norms measured on an existing dueling checkpoint sequence,
~1.2-3.3), and an EMA-averaged eval snapshot (`--eval_ema_decay`, evaluation-only, never touches
what's broadcast to clients). All three default to an exact no-op, unit-tested. 12-run matrix:
each alone, and each combined with dueling (the sec 15 winner), both rosters, same seed-4
methodology.

**Full results, both rosters, sorted best-to-worst mean reward:**

| roster | condition | mean | best | worst |
|---|---|---:|---:|---:|
| 2-city | **dueling+n_step** | **-1327.5** | **-20.8** | **-3541.2** |
| 2-city | n_step alone | -2134.0 | -23.3 | -4007.5 |
| 2-city | dueling alone (sec 15) | -2281.1 | -40.8 | -4693.8 |
| 2-city | dueling+ema_eval | -2314.7 | -116.9 | -3604.8 |
| 2-city | dueling+gradclip | -2494.8 | -71.2 | -4858.3 |
| 2-city | gradclip alone | -2551.3 | -56.6 | -4738.8 |
| 2-city | momentum alone (sec 16) | -2826.2 | -819.8 | -4344.7 |
| 2-city | ema_eval alone | -2897.7 | -2336.2 | -3495.3 |
| 2-city | baseline | -3095.3 | -168.6 | -5031.3 |
| 2-city | dueling+momentum (sec 18) | -3163.7 | -1277.7 | -4621.7 |
| 3-city | **dueling+n_step** | **-2300.7** | **-18.2** | **-4074.0** |
| 3-city | dueling alone (sec 15) | -2469.4 | -244.2 | -4335.2 |
| 3-city | dueling+gradclip | -2538.3 | -25.3 | -5356.0 |
| 3-city | ema_eval alone | -2763.0 | -2249.0 | -3678.6 |
| 3-city | dueling+momentum (sec 18) | -2777.7 | -1256.3 | -4160.4 |
| 3-city | n_step alone | -2845.8 | -23.0 | -4245.3 |
| 3-city | dueling+ema_eval | -3064.2 | -2233.2 | -3852.9 |
| 3-city | momentum alone (sec 16) | -3434.7 | -933.4 | -5416.3 |
| 3-city | gradclip alone | -3679.2 | -1768.5 | -5086.1 |
| 3-city | baseline (sec 12) | -3821.7 | -1936.1 | — |

**Headline result: `dueling+n_step` is a genuine, generalizing synergy — #1 on mean AND best
round on both rosters, not just an average of its two ingredients.** On 2-city its mean
(-1327.5) beats *both* dueling alone (-2281.1) and n_step alone (-2134.0) by a wide margin, not
just splits the difference — same qualitative pattern on 3-city (-2300.7 vs -2469.4 / -2845.8).
Best round on 3-city (-18.2) is the best single number recorded anywhere in this entire
investigation, across every condition tested. This is the opposite outcome from
dueling+momentum (sec 18): two mechanisms that stack constructively instead of fighting.
**Mechanistic read:** n-step returns give a cleaner, less-noisy training signal per transition
(the credit-assignment problem n-step is designed to fix), while dueling lets the head
specialize fast per action; the two don't compete for the same "resource" the way momentum's
damping directly opposes dueling's fast-movement mechanism (sec 18's explanation) — plausible
they're complementary because one improves what gets learned per step and the other improves how
fast the network can act on it.

**n-step alone is also the strongest single new-idea result** — better than dueling alone on
every metric on 2-city, and the best best-round of any single (non-combined) intervention on
3-city (-23.0). Cheap, no architecture change, and combines the best of anything tested.

**gradclip and ema_eval, alone or combined with dueling, are unconvincing:**
- `pseudo_grad_clip=1.5` alone helps meaningfully on 2-city but is barely distinguishable from
  baseline on 3-city — the threshold was calibrated from a 2-city checkpoint sequence and
  evidently doesn't transfer; a fixed clip norm doesn't generalize across roster sizes the way
  dueling and n-step's mechanisms do. Combined with dueling: roughly neutral on 2-city, mixed
  (much better best-round, worse mean/worst) on 3-city — no clean win either way.
- `eval_ema_decay=0.9` alone does exactly what its mechanism predicts and nothing more: compresses
  the *reported* variance (far-and-away best worst-case on both rosters, far-and-away worst
  best-case on both rosters) without changing training at all, since it's evaluation-only by
  design (see the implementation note in `PROJECT_NEXT_STEPS.md`). Combined with dueling: decent
  on 2-city (keeps ~dueling's mean, damps the worst-case, doesn't sacrifice best-case nearly as
  much as ema_eval alone), but net-negative on 3-city (worse mean than either ingredient alone,
  best-case collapses back to ema_eval-alone levels). Not reliable enough to recommend generally.

**Practical recommendation, updated: use `--dueling --n_step 3` together.** This supersedes the
sec 15 "use dueling alone" recommendation — n-step wasn't tested yet when that was written, and
the combination is unambiguously better than dueling alone on every metric, on both rosters
tested. `--server_momentum` and `--pseudo_grad_clip` combined with dueling remain not
recommended (sec 18, and gradclip's inconsistency above); `--fedprox_mu` remains not recommended
(sec 14).

## 20. Phase 1 masked-head ablation redone on `dueling+n_step`, three roster sizes: the fix's
    benefit shrinks as the roster grows

**2026-08-11/12, run unattended overnight.** Every prior Phase 1 ablation (§1, §8, §11, §12) was
measured on the noisy pre-dueling/pre-n_step baseline. With §19's validated best config
(`--dueling --n_step 3`) now established, redid the fix-on vs `--disable_head_fix` comparison on
top of it — 5 seeds each on the 2-city and 3-city rosters (matching the project's standard 5-seed
methodology), plus a first-ever look at the full 7-city paper roster (2 seeds, `base_dir
environments`) since no dueling/n_step data existed there at any seed count. All runs `--parallel`,
`fedavg`, `lr=3e-4`, 20 rounds, `local_episodes=2`. 2-city fix-on reuses the 5 seeds from §19's
validation batch; every other condition is fresh.

**Summary (mean/std across seeds):**

| roster | condition | mean reward | std (across seeds) | best-round mean |
|---|---|---:|---:|---:|
| 2-city (n=5) | fix-ON | -2030.4 | 515.0 | -43.9 |
| 2-city (n=5) | fix-OFF | -3004.0 | 374.0 | -906.3 |
| 3-city (n=5) | fix-ON | -2590.6 | 498.9 | -172.7 |
| 3-city (n=5) | fix-OFF | -2772.5 | 286.2 | -726.8 |
| 7-city (n=2) | fix-ON | -7602.8 | 483.3 | -4038.9 |
| 7-city (n=2) | fix-OFF | -6895.1 | 629.1 | -3762.5 |

**Per-seed numbers (2-city fix-ON reuses §21's 5-seed validation set):**

| roster | condition | seed1 | seed2 | seed3 | seed4 | seed5 |
|---|---|---:|---:|---:|---:|---:|
| 2-city | fix-ON mean | -2455.2 | -1667.1 | -1954.8 | -1327.5 | -2747.4 |
| 2-city | fix-ON best | -13.5 | -18.9 | -142.8 | -20.8 | -23.5 |
| 2-city | fix-OFF mean | -3399.4 | -2471.7 | -2792.0 | -3455.9 | -2901.2 |
| 2-city | fix-OFF best | -753.5 | -52.3 | -1333.0 | -2352.4 | -40.2 |
| 3-city | fix-ON mean | -2729.2 | -3368.2 | -1864.8 | -2300.7 | -2690.2 |
| 3-city | fix-ON best | -32.8 | -790.1 | -10.2 | -18.2 | -12.0 |
| 3-city | fix-OFF mean | -2526.2 | -2602.4 | -3163.0 | -3074.4 | -2496.4 |
| 3-city | fix-OFF best | -750.9 | -1229.0 | -14.7 | -1613.1 | -26.1 |
| 7-city | fix-ON mean | -7119.5 | -8086.1 | — | — | — |
| 7-city | fix-ON best | -3125.3 | -4952.5 | — | — | — |
| 7-city | fix-OFF mean | -6266.0 | -7524.1 | — | — | — |
| 7-city | fix-OFF best | -1877.2 | -5647.8 | — | — | — |

(3-city fix-ON seed4 reused from §19's matrix, `results/run_2026_08_10-07_16_31`; every other cell
is a fresh run from this batch, run directories listed in §20's companion notes below.)

**2-city: a real, statistically clean win for the fix, for the first time in this project's
history.** Mean-reward gap (974) against combined standard error (~285) gives **|diff|/SE = 3.42**
— compare to §12's final ambiguous read of 0.68 on the pre-dueling/n_step baseline. Best-round
also wins decisively (-44 vs -906). This is the first time the masked-head fix has shown an
unambiguous *mean*-reward win, not just a peak-vs-stability tradeoff — strongly suggests the
ambiguity in every earlier ablation (§8, §11, §12) was substantially the underlying training
instability adding noise on top of the fix's real effect, now that dueling+n_step has cleaned that
noise up.

**3-city: mean reward is back to ambiguous (|diff|/SE = 0.71, same order as the old inconclusive
result), but best-round still wins clearly** (-173 vs -727). This reproduces the *original* §11
pattern almost exactly (fix reaches a better peak reliably, doesn't move the round-to-round mean
outside noise) — on a much cleaner baseline than §11 had, which strengthens confidence this
specific pattern (peak-win, mean-ambiguous) is a real property of the 3-city roster/mechanism
combination, not just noise from the old instability.

**7-city: fix-off looks numerically better on this data (|diff|/SE = 1.26) — but this is not a
trustworthy read, only a flag.** n=2 seeds gives essentially no statistical power; a ratio of 1.26
on 2 samples is not meaningfully different from n=5's noise floor. Checked whether the two runs
hitting near -10000 (`fixon_seed2`, `fixoff_seed1`) were a reward-clipping/gridlock-ceiling
artifact — they are not: `fixon_seed2` spends nearly the *entire* 20-round run in the -5000 to
-10000 band (not a one-round spike), and `fixoff_seed1`'s -10012.6 is only its first round, which
then recovers substantially (down to -1877 by round 5). Real signal, just very noisy and far from
converged — 20 rounds is evidently nowhere near enough for the full 7-city roster the way it is
for 2-3 cities. Absolute reward scale here (-6000 to -10000 range throughout) also dwarfs the
2-city/3-city runs' typical range, consistent with CLAUDE.md's note that 7-city Phase 1 needs its
full standard treatment (5 seeds, likely more rounds) before trusting any number from it.

**Reading:** the masked-head fix's benefit appears to shrink as roster size grows — clean mean-
reward win at 2 cities, peak-only win at 3 cities, unclear-to-negative at 7 (on statistically thin
data). This is a genuinely interesting pattern worth reporting either way it resolves: possibly the
row-sparsity mechanism the fix targets (§2, §9) matters more when action-space heterogeneity is
concentrated across fewer clients, and gets diluted/complicated as more, more-varied clients enter
the aggregation. **Not yet a basis for a paper claim at 7-city** — needs 3+ more seeds (bringing it
to the standard 5) and likely more rounds before the 7-city number can be trusted at all. The
2-city and 3-city results, by contrast, are on solid footing (5 seeds each, matching project
standard) and are ready to report as-is.

**Run directories (for re-derivation):**
`2city_fixoff`: seed1 `run_2026_08_11-10_23_10_1199787`, seed2 `..._1199786`, seed3 `..._1199788`,
seed4 `run_2026_08_11-11_52_39_1220777`, seed5 `run_2026_08_11-11_55_34_1224641`.
`3city_fixon`: seed1 `run_2026_08_11-13_21_10_1250573`, seed2 `..._1250574`, seed3
`run_2026_08_11-15_16_22_1280250`, seed5 `run_2026_08_11-15_16_27_1280445` (seed4 per above).
`3city_fixoff`: seed1 `run_2026_08_11-17_10_58_1312435`, seed2 `..._1312436`, seed3
`run_2026_08_11-18_38_24_1339061`, seed4 `run_2026_08_11-18_38_30_1339260`, seed5
`run_2026_08_11-19_23_58_1358493`. `7city_fixon`: seed1 `run_2026_08_11-20_46_18_1384725`, seed2
`run_2026_08_11-22_54_47_1419099`. `7city_fixoff`: seed1 `run_2026_08_12-01_03_53_1448113`, seed2
`run_2026_08_12-02_44_57_1472126`. All under `results/`.

## 21. `dueling+n_step` 5-seed validation (2-city): holds up, no seed4-style outlier

> **⚠ CORRECTED BY §43 (2026-08-18):** the best-round numbers below were evaluated on `city_1`,
> one of this roster's own training cities (in-distribution, not a true holdout — see §25/§29's
> caveat, confirmed load-bearing by §43). On the actual `city_5_holdout`, best-round losses badly
> to `fixed_time`/`max_pressure` instead of beating them. The mean-reward numbers here are
> unaffected by this (mean was never the claim being made), but any "beats baselines" framing
> below should be read as "beats baselines in-distribution," not as evidence of generalization.

**2026-08-10, run between §19 and §20, written up here for completeness.** Before trusting
`dueling+n_step` as the new recommended config (§19 found it on a single seed, 4 — deliberately
the known-hardest case per §13, but still one sample), validated it across 5 seeds on the 2-city
roster (`environments_c1_4`), same 20-round/`fedavg`/`lr=3e-4` methodology throughout. Seed 4
reused from §19's matrix (`results/run_2026_08_10-05_37_47`, mean -1327.5); seeds 1/2/3/5 run
fresh.

| seed | mean reward | best round | worst round |
|---|---:|---:|---:|
| 1 | -2455.2 | -13.5 | -5440.3 |
| 2 | -1667.1 | -18.9 | -4787.1 |
| 3 | -1954.8 | -142.8 | -4265.1 |
| 4 | -1327.5 | -20.8 | -3541.2 |
| 5 | -2747.4 | -23.5 | -5003.3 |

**Mean across 5 seeds: -2030.4 (std 515.0).** Every seed individually beats the single-seed
baseline (-3095.3), dueling-alone (-2281.1), and n_step-alone (-2134.0) numbers from §15/§19 —
no seed behaves like §12's seed4 outlier that reversed the FedProx-era ablation's conclusion.
Confirms this is a genuine, seed-robust result, not a lucky draw — the number this doc and
`PROJECT_NEXT_STEPS.md` cite as `dueling+n_step`'s validated performance. This 5-seed set is also
what §20's Phase 1 re-ablation's 2-city fix-ON arm reuses directly (no need to rerun it there).

## 22. Infrastructure: concurrent batch runner (`analyse/run_concurrent_batch.sh`) — a real but
    more modest speedup than first measured, plus a real bug caught and fixed

**2026-08-10/11.** Investigated whether training was leaving CPU/RAM headroom unused. Measured
(single job running): each city worker uses only ~13-15% of one core (SUMO/libsumo per-tick
stepping is the bottleneck, not CPU/PyTorch compute) and ~2.5-3.5GB RAM per run — on a 12-core/
15GB-RAM dev machine, both facts said an idle-time win was available by running multiple
independent jobs (seed sweeps, flag ablations) concurrently instead of the sequential-only pattern
every earlier batch in this doc used.

**Real bug found and fixed while wiring this up:** `experiments/federated_training.py::main()`
computed `run_dir` as `results/run_<timestamp>` with only second-granularity timestamps. Two
processes launched within the same wall-clock second (exactly what a batch script does) computed
the *identical* directory string; `os.makedirs(run_dir, exist_ok=True)` then let the second
process silently write into the first's directory — interleaved checkpoints/logs/history from two
different seeds merged into one folder, no error, no warning. Caught this by accident (three
concurrently-launched seed runs collided; two of the three failed with a
`RuntimeError: Parent directory .../clients does not exist` from the third process's directory
having been deleted out from under them) before it could silently corrupt a real result. **Fix:**
suffix `run_dir` with `os.getpid()` and make directory creation strict (`exist_ok=False`), so a
collision fails loudly instead of merging data. Verified with a real 2-process concurrent launch
(two `federated_training` processes started in the same second correctly got distinct
`run_dir`s) before trusting the mechanism on any real experiment.

**Speedup — measured end-to-end, not estimated.** Batch: seeds 1/2/3 concurrent (3-way) + seed 5
solo (trailing, once a slot freed), all `dueling+n_step` 2-city, `MAX_CONCURRENT=3`.

| | duration | pace |
|---|---:|---:|
| seed1 (concurrent) | 161.6 min | 8.08 min/round |
| seed3 (concurrent) | 160.9 min | 8.04 min/round |
| seed2 (concurrent) | 158.4 min | 7.92 min/round |
| seed5 (solo, trailing) | 100.6 min | 5.03 min/round |

Solo pace (5.03 min/round) matches every prior solo run in this doc almost exactly — confirms the
concurrent runs' slowdown (~8.0 min/round, **~60% slower per run**) is real per-run contention, not
a measurement artifact. An early same-batch sample (first ~2 rounds only) had suggested only ~20%
slowdown — contention evidently worsens as a run progresses, plausibly from growing replay-buffer
memory pressure across the 3 concurrent processes. **Net result over the whole 4-job batch: 259
min actual wall-clock vs. ~400 min estimated if run fully sequentially at the solo baseline — a
real ~1.5x speedup**, meaningfully better than sequential despite the per-run tax, but well short of
the near-linear speedup the idle-CPU/RAM-headroom framing alone would have predicted.

**Resulting artifact:** `analyse/run_concurrent_batch.sh` (persisted in the repo, not scratchpad;
documented in `CLAUDE.md` under "Common commands") is now the project's default way to run any
multi-run batch — `MAX_CONCURRENT=3` default, override per-machine/per-roster-size. Every batch
from §19's 5-seed validation onward that used it is noted as such; every batch through §18 predates
it and ran strictly sequentially.

## 23. 7-city Phase 1 ablation brought to 5 seeds: the fix's mean-reward benefit doesn't just
    shrink with roster size, it vanishes — but the best-round win survives at every scale tested

**2026-08-13.** Ran the 3 remaining seeds per arm (seeds 3/4/5, fix-on and fix-off, 6 runs total)
flagged as the standing next action in §20/item 6 above, `--dueling --n_step 3`, 7-city roster
(`base_dir environments`), `--parallel fedavg lr=3e-4`, 20 rounds. Used
`analyse/run_concurrent_batch.sh` at `MAX_CONCURRENT=2` (not the originally-planned 1) — real-time
`top`/`ps` monitoring during the first job showed the 6 city-worker processes are bursty (heavy
during backprop, idle during SUMO/libsumo per-tick stepping), not steadily CPU-bound the way §22's
`MAX_CONCURRENT=1` recommendation for 7-city assumed; load average stayed around 4-5 of 12 cores
with 2 concurrent 7-city jobs running. RAM was the tighter constraint (~9GB/15.8GB used with 2 jobs
active, 12 city workers total) but never hit swap. Net result: all 6 runs finished in ~7h wall-clock
(11:59 → 19:00) rather than the ~12h sequential estimate — a real, measured speedup, consistent with
§22's general finding that concurrency helps even when a single job looks compute-heavy on paper.

**Per-seed numbers (new seeds 3/4/5; seeds 1/2 from §20, reused not rerun):**

| condition | seed1 | seed2 | seed3 | seed4 | seed5 |
|---|---:|---:|---:|---:|---:|
| fix-ON mean | -7119.5 | -8086.1 | -7052.6 | -5330.2 | -7003.5 |
| fix-ON best | -3125.3 | -4952.5 | -1404.0 | -409.9 | -1018.4 |
| fix-OFF mean | -6266.0 | -7524.1 | -7855.9 | -6375.4 | -7137.0 |
| fix-OFF best | -1877.2 | -5647.8 | -4927.3 | -3399.6 | -3670.1 |

**Full 5-seed summary:**

| condition | mean reward | std (across seeds) | best-round mean |
|---|---:|---:|---:|
| fix-ON | -6918.4 | 889.0 | -2182.0 |
| fix-OFF | -7031.7 | 624.5 | -3904.4 |

**Mean reward: no signal at all, |diff|/SE = 0.23** — weaker than even 3-city's already-ambiguous
0.71 (§20), and a long way from 2-city's clean 3.42. The 2-seed read that triggered this batch
(fix-off looking *better*, |diff|/SE=1.26, §20) does not survive more data — it was noise, as
flagged at the time. With the full 5 seeds the direction is back to fix-on nominally ahead (-6918
vs -7032) but the gap is small relative to the per-seed spread on both arms; this is a genuine null
result on mean reward at 7-city, not a coin-flip that happens to have landed near zero.

**Best-round: fix-on still wins clearly, gap 1722 (-2182 vs -3904).** This is the third roster size
in a row (2-city, 3-city, 7-city) where the fix reaches a meaningfully better peak, even as its
effect on the round-to-round mean disappears. In relative terms the peak win *also* shrinks with
scale — roughly 95% better at 2-city, 76% at 3-city, 44% at 7-city (comparing each roster's fix-on
vs fix-off best-round mean) — but it never crosses over to a wash or a loss at any scale tested.

**Reading — this resolves §20's open question and sharpens the paper's honest framing.** The
masked-head fix's benefit on *mean* reward monotonically shrinks with roster size (3.42 → 0.71 →
0.23) and is genuinely gone by 7 cities, not just harder to see. But its benefit on *best achieved
policy* is real and persists at every roster size tested, shrinking in magnitude but never
disappearing. The honest claim this data supports: **the masked-head fix reliably helps a
federated run reach a better peak policy, at any roster size tested, but its effect on
average/expected round-to-round performance is roster-size-dependent and not distinguishable from
noise at the full 7-city scale.** This is a substantive, reportable finding on its own — a
conditional result, not a clean "fix wins," consistent with the plan's own guidance (`PROJECT_NEXT_STEPS.md`
Phase 1 decision gate) for treating this as legitimate rather than something to keep chasing more
seeds to reverse. All three roster sizes (2/3/7-city) now have the standard 5 seeds per arm — no
roster in the Phase 1 ablation needs more data to be trustworthy.

**Run directories (new seeds 3/4/5 only; seeds 1/2 listed in §20):**
`fixon7_seed3` `run_2026_08_13-12_01_38_3778`, `fixon7_seed4` `run_2026_08_13-12_01_38_3779`,
`fixon7_seed5` `run_2026_08_13-14_37_48_33224`, `fixoff7_seed3` `run_2026_08_13-14_38_01_33418`,
`fixoff7_seed4` `run_2026_08_13-16_36_52_54700`, `fixoff7_seed5` `run_2026_08_13-17_07_13_59988`.
All under `results/`. Batch log: `results/phase1_7city_seeds345.log`.

## 24. Major bug found while starting Phase 2: the `fixed_time` rule-based baseline never actually
    ran fixed-time control — every `fixed_time` number in this project's history is invalid

**2026-08-13.** Picked up item 5 from "Open questions" (`baseline_max_pressure`'s implausible
`reward=-0.34, waiting_time=2.9s` numbers, flagged since §1, 7-city, never investigated) while
Phase 2's 7-city strategy-comparison batch was being scoped. Reproduced cheaply via
`--baseline_controller max_pressure` (no training needed, ~15s). Then compared against
`--baseline_controller fixed_time` under the *exact same* code path (same holdout city, same
episode count) rather than cross-referencing old numbers from a different run configuration —
`fixed_time` came back `reward=-9995.52, waiting_time=1900.21s, arrived=241`, i.e. near-total
gridlock, a ~4-order-of-magnitude gap from `max_pressure`. Both numbers individually matched
historical citations (§1's `baseline_fixed_time`: reward -9996), so this wasn't a fluke of this
session — it's what the project has been reporting all along.

**Root cause (three-layer bug, traced with a debug call-counter, not guesswork):**
`HoldoutEvaluator._evaluate_policy` (`federated/evaluator.py`) toggles rule-based-baseline mode via
`if hasattr(env, "fixed_ts"): env.fixed_ts = (policy_name == "fixed_time")` — this is meant to make
the underlying `sumo_rl.SumoEnvironment` skip applying any external action and let SUMO run its own
native default signal-timing program, which is what "fixed-time control" is supposed to mean.
Three independent bugs combined to make this toggle a complete no-op on the real pipeline:
1. **`MultiAgentFederatedWrapper`, `ActionMaskPadder`, `CommDropoutWrapper`, and
   `RewardShapingWrapper`** (`environments/federated_env.py`, `federated/comm_dropout.py`) each
   wrap the raw SUMO env and each define `__getattr__` to delegate attribute *reads* down to
   `self.env` — but none of them delegates attribute *writes*. `env.fixed_ts = True` on any of
   these wrapper objects therefore just creates a new same-named instance attribute on the
   wrapper's own `__dict__`, shadowing all future reads (so `hasattr`/`getattr` on the wrapper
   look correct) while never reaching the real `SumoEnvironment` object underneath. The holdout
   evaluator's env is `CommDropoutWrapper(ActionMaskPadder(MultiAgentFederatedWrapper(raw_env)))`
   (see `make_holdout_evaluator`'s `build_holdout_env` closure) — three layers deep, so this bug
   fired at every layer simultaneously, and the flag never once reached `raw_env.fixed_ts`.
2. Even had the flag reached the raw env, `sumo_rl.environment.env.SumoEnvironment.step()`'s
   multi-agent branch (used by this entire federated pipeline; the single-agent branch is legacy)
   applied actions unconditionally — `if self.traffic_signals[ts].time_to_act: self._apply_actions(...)`
   — with no `if not self.fixed_ts:` guard, unlike the single-agent branch a few lines above it
   which has one. Multi-agent `fixed_ts` was silently a dead flag at the environment level too.
3. Net effect: for `policy_name == "fixed_time"`, `HoldoutEvaluator._policy_action`'s fallback
   (`return int(valid[0])`, i.e. "always the first valid action") was the *only* thing determining
   behavior — a "never switch off the first phase" degenerate policy, not real fixed-time signal
   timing. This produces exactly the catastrophic gridlock numbers on record.

**Fix:** added a `fixed_ts` `@property` (get + set, forwarding to `self.env.fixed_ts`) to all four
wrapper classes, and added the missing `if not self.fixed_ts:` guard around the multi-agent
action-application loop in `SumoEnvironment.step()`, mirroring the existing single-agent guard.
Verified with a debug call-counter on `_apply_actions` that it now fires exactly 0 times across a
full 720-step fixed-time episode (was 11520 = 16 ts × 720 steps before the fix), and that a
manually-driven probe env shows the SUMO traffic light actually cycling through many distinct
phase indices under `fixed_ts=True`, not frozen on one. Confirmed the fix doesn't touch the
trained-model path: `fixed_ts` defaults to `False`, the new guard's `if not self.fixed_ts` branch
still applies actions exactly as before, and a 1-round real training smoke test
(`environments_c1_4`, `--dueling --n_step 3`) completed normally with healthy varied action
distributions after the fix. (Note: `pytest tests/` has 3 pre-existing failures unrelated to this
fix — they import `sumo_rl` from a different, stale checkout at
`/mnt/c/Users/Deea/SUMO/SUMO_Reinforcement_learning_traffic` that this machine also has installed
editable, not this repo's local `sumo_rl/`; confirmed by the traceback's file paths. Pre-existing
environment quirk, not something this session introduced or fixed.)

**Numbers, same holdout city / episode count, before vs. after the fix:**

| controller | reward (before → after) | waiting_time (before → after) | arrived (before → after) |
|---|---:|---:|---:|
| `fixed_time` | -9995.52 → **-2.73** | 1900.21s → **6.97s** | 241 → **1439** |
| `max_pressure` | -0.34 (unaffected — this controller's own logic was always correct) | 2.91s | 1462 |

**Why this matters far beyond fixing one number.** Real fixed-time control on this network turns
out to be *good* — nearly matching `max_pressure`, not four orders of magnitude worse. Every
`fixed_time` citation anywhere in this project's history (§1's Phase 1 rule-based reference,
Phase 3's planned "trained policies should beat fixed-time" sanity floor) was measuring the broken
degenerate policy, not real fixed-time control, and needs to be treated as invalid, not just
imprecise. **More importantly: with a correct `fixed_time` number now available, the current
7-city trained federated DQN (§23: mean reward -6918.4, best-round -2182.0) is dramatically
*worse* than both rule-based baselines** (`fixed_time` -2.73, `max_pressure` -0.34) **on this same
holdout city.** This was structurally invisible before today, since the only baseline comparison
point was itself broken in a way that made the DQN look relatively less bad than it apparently is.
This is a significant, previously-hidden finding that argues for treating "does the trained policy
even beat simple rule-based control" as a live open question requiring its own investigation
before sinking more compute into aggregation-strategy comparisons that implicitly assume the
trained model is already in a competitive regime.

**Not yet done:** the `fixed_time`/`max_pressure` numbers above are single-episode, non-seeded
(deterministic eval, `deterministic_eval=True` means episodes only vary if `eval_seed_base` offset
changes, and neither rule-based controller has any stochastic component, so std=0.0 here is
expected, not a bug) — fine for confirming the bug and its fix, not yet run with proper multi-seed
rigor or folded into a full baseline table. Also not yet checked: whether this same bug affected
any *other* code path relying on `fixed_ts` (grep shows only the evaluator uses it), or whether the
2-city/3-city Phase 1 rosters would show the same DQN-loses-to-baseline pattern (not yet measured
there).

## 25. Multi-seed follow-up on §24: the DQN-loses-to-baseline finding is real and seed-robust on
    7-city, and the picture is different (and more nuanced) at 2-city

**2026-08-13, same session as §24.** Varying `--eval_sumo_seed` (1-5) *does* perturb the
deterministic rule-based controllers slightly (confirms SUMO's own seed affects something even
though the route file's vehicle demand is otherwise deterministic — small but real spread, not
identical runs), so this is a legitimate multi-seed check, not 5 copies of the same number.

**7-city holdout (`city_5_holdout`, the real paper holdout), 5 seeds each:**

| controller | mean reward | std |
|---|---:|---:|
| `fixed_time` | -2.250 | 0.505 |
| `max_pressure` | -0.044 | 0.016 |
| trained DQN (§23, 5 seeds) | -6918.4 | 889.0 |

Gap is enormous and the baselines' variance is tiny relative to it — this is not a noisy read. **On
7-city, the trained DQN does not beat either rule-based baseline, not even close, at any seed
tested, mean or best-round** (DQN best-round -2182.0 is still ~550x worse than `fixed_time`'s worst
seed).

**2-city (`environments_c1_4`) — important methodological caveat first:** the baseline-controller
path's holdout-city selection (`make_holdout_evaluator` in `experiments/federated_training.py`)
falls back to a *compatible* city when the true `city_5_holdout`'s action space is wider than the
roster's global action_dim — for the 2-city roster (`city_1`+`city_4`, global action_dim capped at
5) it silently evaluates on `city_1` instead, logged as `"Using 'city_1' as evaluation city ...
(compatibility fallback)"`. **`city_1` is one of the 2-city roster's own *training* cities, not a
held-out one** — this fallback affects the *trained* model's own per-round eval during real 2-city
training runs too (worth independently confirming this is really what §19/§20/§21's 2-city numbers
were evaluated against, since "holdout" in this project's vocabulary may not mean what it's assumed
to mean for the smaller rosters — flagging as a new open item, not yet confirmed).

With that caveat, on `city_1` as the eval city: `fixed_time` reward=-563.77, `max_pressure`
reward=-256.93 (single seed only, not yet a 5-seed check). Compare against the 2-city trained DQN's
own 5-seed numbers on this same fallback city (§21): **mean -2030.4 (worse than both baselines,
same direction as 7-city) but best-round -43.9 (better than both baselines — the opposite of
7-city).**

**Reading:** the roster-size pattern from §20/§23 (masked-head fix's benefit shrinking with scale)
now has a companion pattern for absolute competitiveness against simple heuristics: **at 2-city,
the trained DQN's peak performance can beat simple rule-based control even though its average
performance can't (consistent with a policy that's still unstable round-to-round but capable of
good policies); at 7-city, the DQN does not beat rule-based control even at its best round out of
20 — it isn't just noisier, it's uniformly worse.** This is consistent with (though not proof of)
a scaling/sample-efficiency problem — 20 rounds may simply be far short of enough for a 7-city,
16-way-heterogeneous-action-space shared policy to reach competence, whereas 2 cities is a much
easier learning problem. Not yet root-caused further (e.g. not checked: does more rounds close the
gap on 7-city, or is there a structural ceiling).

**Open item (1) confirmed, not just suspected:** checked `results/run_2026_08_10-02_20_32/training.log`
(a real 2-city `environments_c1_4` training run, not just the baseline-controller probe) —
`WARNING | Holdout evaluator city is incompatible (dims_match=True, holdout_action_dim=8,
global_action_dim=5). Falling back to a compatible city from base dirs.` followed by `Using
'city_1' as evaluation city for this run (compatibility fallback).` **Every 2-city (and, unverified
but structurally identical, likely every 3-city) result anywhere in this document — §6 onward, all
of §15/§16/§18/§19/§20/§21's dueling/n_step/momentum/masked-head validations on the 2-city roster —
was evaluated in-distribution on `city_1`, one of the 2-city roster's own training cities, not a
genuinely unseen holdout.** This does **not** invalidate the fix-on/fix-off or config ablation
*comparisons* themselves (both arms of every such comparison used the identical fallback city, so
the relative read is still fair) — but any framing of those results as "generalizes to an unseen
city" is incorrect and should be corrected to "evaluated in-distribution on a training city" until
this is fixed (e.g. by capping the global action_dim padding differently, or accepting the
holdout's wider action space and padding other cities up to it instead of down). `CLAUDE.md`'s
architecture section states city_5_holdout is "auto-excluded from training and reserved for
HoldoutEvaluator" without this caveat — needs updating.

**Remaining open items:** ~~(2) 3-city roster's fallback eval city and DQN-vs-baseline comparison
not yet run.~~ **Done — see §29: the 3-city roster falls back to the identical `city_1` eval city
as 2-city (confirmed on all 20/20 runs of a dedicated check), so it adds no distinct holdout
endpoint.** ~~(3) whether more training rounds closes the 7-city gap is untested~~ **Done — see
§28: it doesn't close, the model regresses past round 20 instead.** (4) whether the 7-city roster
itself (which does use the true `city_5_holdout`, action_dim=8 matching) is the only trustworthy
"real holdout" data in this entire document — worth an explicit audit of which past results used a
true holdout vs. the `city_1` fallback before writing the paper's evaluation-methodology section.

## 26. Mechanism dig on §25's 7-city gap: not a collapsed/degenerate policy, not a confidence
    problem — reward decomposition points at residual end-of-episode congestion, cause still open

**2026-08-13, zero new training compute — reused existing checkpoints.** Before spending hours on
longer 7-city runs, loaded the actual best-ever 7-city checkpoint on disk (`fixon7_seed4` from
§23/§25, `results/run_2026_08_13-12_01_38_3779/global_round_014.pth`, round 14's -409.9 was that
run's best of 20) and re-ran it through the exact same `HoldoutEvaluator` path used in training, to
directly inspect what the policy is actually doing rather than only looking at the scalar reward.

**Ruled out: policy collapse.** Action distribution across all 16 intersections is diverse (e.g.
`A0: {4: 198, 0: 436, 7: 1, 5: 71, 1: 14}`) — not stuck on one action like §24's pre-fix bug. Vehicle
throughput is close to the heuristics': 1384 arrived vs. `max_pressure`'s 1462 / `fixed_time`'s
~1439-1441 on the same city. The policy is functioning, not broken.

**Ruled out (surprisingly): low Q-value confidence as the scale-specific cause.** Hypothesis going
in: 7-city's shared network sees more context diversity per round, so maybe its Q-values are less
differentiated (smaller top1-vs-top2 gap) than a 2-city model's, i.e. it hasn't learned confident
preferences yet. Tested by loading the single best 2-city checkpoint on disk too
(`results/run_2026_08_10-17_05_38_1001848/global_round_008.pth`, round 8, -13.5, the best 2-city
result recorded anywhere in this document) through the identical inspection. **Result: 7-city's
Q-gap mean (0.140) is actually slightly *higher* (more confident) than 2-city's best checkpoint
(0.096), not lower.** Whatever separates 2-city's near-heuristic-beating performance from 7-city's
catastrophic gap, it is not "the network hasn't learned to be confident yet."

**What the numbers actually show:** the reward function is a telescoping sum of per-step waiting-time
*decreases* (§25's derivation), so a bad total-episode reward means the policy leaves a lot of
accumulated waiting time unresolved by episode end, not that it fails to move vehicles at all.
7-city trained: waiting_time=426s, stopped=81 vs. `max_pressure`'s waiting_time=2.9s, stopped~0-1 —
despite near-equal arrival counts. **Reading: the policy handles steady-state throughput
reasonably but is far worse than the heuristics at draining queues down to zero rather than merely
keeping them from exploding** — a qualitatively different failure mode than gridlock, more like
"good enough to avoid collapse, not good enough to be efficient."

**Not yet resolved — two live hypotheses, not yet distinguished:** (a) **aggregation dilution**:
7-city's FedAvg step blends gradients from 6 heterogeneous clients every round instead of 2,
plausibly making each round's *net* movement noisier/slower to converge even though each city still
gets the same `local_episodes=2` of local training — consistent with §20/§23's independent finding
that the masked-head fix's own benefit shrinks with roster size, a second data point for "more
clients per round makes aggregation-based learning slower/noisier in this codebase," not proof by
itself. (b) **genuine undertraining**: `fixon7_seed4`'s full 20-round reward trace oscillates
without a visible convergence trend (round1 -10068, round6 -632, round9 -988, round14 -410 [best],
round20 -7746) — consistent with a run that simply hasn't reached a stable basin yet, not with one
that converged to a ceiling and stayed there. Distinguishing (a) from (b) needs an actual longer
run (the "test more rounds" experiment from the fork this section was written instead of running) —
this section's zero-cost analysis narrows *what* to look for once that run exists (does the
oscillation dampen, and does the residual-waiting/queue-clearing gap specifically shrink) but
cannot resolve the fork on its own.

## 27. Phase 2 aggregation-strategy comparison, 7-city, auto-generated 2026-08-15

**Auto-generated by `experiments/analyze_phase2_strategies.py`** once the overnight strategy-comparison batch (`ema_loss`, `ema_alignment`, `velocity_novelty`, `gradient_survival`, `clustered_fedavg`, all `--dueling --n_step 3`, 7-city roster, masked-head fix on) finished. Compares each strategy's per-seed mean/best-round eval reward (loaded straight from each run's `federated_history.json`) against the known `fedavg` reference (§23: 5 seeds, mean -6918.4 std 889.0, best-round mean -2182.0).

| strategy | seeds | mean reward | std | best-round mean | vs fedavg | \|diff\|/SE |
|---|---:|---:|---:|---:|---:|---:|
| `clustered_fedavg` | 5 | -6494.5 | 681.5 | -4053.9 | +423.9 | 0.85 |
| `gradient_survival` | 5 | -6933.2 | 576.4 | -1566.6 | -14.8 | 0.03 |
| `ema_loss` | 3 | -8022.6 | 586.1 | -4390.0 | -1104.2 | 2.11 |
| `velocity_novelty` | 5 | -8045.0 | 793.6 | -4687.7 | -1126.6 | 2.11 |
| `ema_alignment` | 3 | -8419.7 | 587.5 | -4785.3 | -1501.3 | 2.87 |
| `fedavg` *(known, §23)* | 5 | -6918.4 | 889.0 | -2182.0 | — | — |

**Best mean reward: `clustered_fedavg`** (-6494.5, 5 seeds, |diff|/SE=0.85 vs fedavg). This does **not** clear the |diff|/SE >= 2 bar this project has used elsewhere for a real (not noise-level) signal -- treat as a lead, not a settled result.

**No strategy clears |diff|/SE >= 2 over plain `fedavg` on mean reward.** Phase 2's core question ("does any smarter aggregation strategy beat plain FedAvg") reads as a negative/null result so far, on the seed counts gathered.

**The bigger unresolved issue is unchanged by this batch.** Even the best strategy here (`clustered_fedavg`, mean -6494.5) is still far below both rule-based baselines (`fixed_time` -2.73, `max_pressure` -0.34, §24 -- single-episode, not yet multi-seed). Comparing aggregation strategies against each other doesn't touch this gap; all of them lose to trivial heuristics by 3-4 orders of magnitude. Open item 7 in this file's "Open questions / next steps" list is still the higher-priority open question.

**Recommended next step (not auto-executed -- needs a decision, same as item 7 already flagged):** before spending more compute on further strategy seeds or Phase 2 scale-up, investigate *why* every trained-DQN configuration loses to `fixed_time`/`max_pressure` on the 7-city holdout. §26's mechanism dig (residual end-of-episode congestion, not policy collapse) narrows this but doesn't resolve it -- the standing fork from §26 (aggregation dilution vs. genuine undertraining) is still open. The 2026-08-14 40-round `fedavg` mechanism-test run (`results/run_2026_08_14-00_59_53_6995`) exists as data toward that fork and hasn't been read yet as of this writeup.

## 28. The 40-round `fedavg` mechanism-test run resolves §26's fork: not undertraining — the model
    regresses, not converges, past round 20

**2026-08-15.** Read `results/run_2026_08_14-00_59_53_6995` (7-city, `fedavg`, `--dueling
--n_step 3`, seed 6, run to 40 rounds instead of the usual 20 specifically to test §26's open
fork: does the DQN-loses-to-baseline gap close with more training (undertraining), or does it
persist regardless of round count (a structural/aggregation ceiling)?

**Round-by-round eval_reward, first half vs. second half:**

| window | mean | std | best | worst |
|---|---:|---:|---:|---:|
| rounds 1-20 (usual budget) | -4895.7 | 3051.7 | -1154.3 (r13) | -10033.1 (r5) |
| rounds 6-20 (post-warmup) | -3347.2 | — | -1154.3 (r13) | -5666.8 |
| rounds 21-40 (extension) | -5623.5 | 2249.0 | -679.4 (r35) | -8549.6 (r31) |

5-round rolling means make the shape clearest: -9541 (r1-5, warmup) → -4987 (r6-10) → **-1741.5
(r11-15, the best window of the entire run)** → -3313 (r16-20) → -5672 (r21-25) → -5557 (r26-30)
→ -4843 (r31-35) → **-6421 (r36-40, second-worst window of the entire run, nearly back to
warmup-level badness)**.

**This is a regression, not a plateau or continued improvement.** The run's best sustained
performance (rounds 11-15) happens *inside* the normal 20-round budget; every 5-round window
after round 20 is worse than that peak, and the final window (36-40) is nearly as bad as the
initial warmup. A whole-run linear fit gives a nominally positive slope (+27.8 reward/round) only
because it's dragged by the catastrophic first 5 rounds — it does not reflect a "still climbing"
trend in the region that matters (post-round-20). Round 35 does set a new single-round best
(-679.4, beating round 13's -1154.3) but round 36 immediately crashes to -7943.8 — a one-round
spike, not a new stable level, consistent with the same round-to-round volatility seen throughout
the run rather than convergence.

**Answers §26's fork: undertraining is not supported by this data; the aggregation-dilution /
instability hypothesis is.** More rounds did not close the gap to rule-based control — if
anything, the back half of the run is worse on average than the front half. Even the single best
round of all 40 (-679.4) remains roughly 300-1500x worse than the corrected multi-seed 7-city
baselines (§25: `fixed_time` -2.250±0.505, `max_pressure` -0.044±0.016) — this is not a
"needs a longer run to get there" gap, it's an order-of-magnitude gap that 2x the round budget did
not meaningfully narrow.

**Caveat:** this is one seed, one extended run, not a multi-seed extension study — doesn't rule out
a different seed behaving differently, and doesn't by itself distinguish *which* structural cause
(aggregation dilution vs. some other instability source) is responsible, only that "just train
longer" is not it. Reallocates priority: further compute is better spent diagnosing *why* federated
aggregation destabilizes a 7-city shared policy (e.g. per-round weight-divergence / gradient-
conflict measurements across cities) than on running this same 40-round extension on more seeds.

## 29. Multi-seed `fixed_time`/`max_pressure` baselines on the 2-city and 3-city rosters — and the
    3-city roster turns out to eval on the same fallback city as 2-city

> **⚠ CORRECTED BY §43 (2026-08-18):** these baseline numbers are themselves fine (rule-based
> controllers, not affected by which city they're measured on the same way a trained policy's
> generalization claim is) — what's corrected is the *comparison drawn from them*. The "DQN
> best-round beats both baseline means" reading below was an in-distribution comparison (`city_1`
> fallback on both sides). Re-run on the true `city_5_holdout` in §43: `max_pressure` -0.34,
> `fixed_time` -2.73, trained DQN best-round -2855.95 — the DQN loses badly instead.

**2026-08-15.** Filled the two gaps §25 flagged as not yet done: proper 5-seed (`--eval_sumo_seed`
1-5) `fixed_time`/`max_pressure` baseline numbers on `environments_c1_4` (2-city) and
`environments_c1_4_6` (3-city), via `--baseline_controller` (no training, ~1min/job, 20 jobs run
2-at-a-time in ~7 minutes total).

| roster | controller | n | mean reward | std |
|---|---|---:|---:|---:|
| 2-city | `fixed_time` | 5 | -472.400 | 70.311 |
| 2-city | `max_pressure` | 5 | -240.698 | 25.784 |
| 3-city | `fixed_time` | 5 | -472.400 | 70.311 |
| 3-city | `max_pressure` | 5 | -240.698 | 25.784 |

**The 2-city and 3-city numbers are identical** — not a coincidence or a bug in this measurement.
Checked all 20/20 runs' logs: every single one, both rosters, logged `Using 'city_1' as evaluation
city for this run (compatibility fallback)`. **The 3-city roster (`environments_c1_4_6`) falls
back to the exact same `city_1` holdout-compatibility substitute as the 2-city roster** — same root
cause as §25's finding for 2-city (`city_5_holdout`'s action_dim=8 doesn't fit either roster's
narrower global `action_dim`). This means "3-city" was never a distinct evaluation endpoint from
"2-city" for any baseline-controller or trained-model run on this roster — resolves §25's open item
(2).

**Consistency check against §25's single-seed numbers:** §25 reported single-seed `fixed_time`
-563.77 / `max_pressure` -256.93 on `city_1`; this 5-seed run gives means of -472.4 / -240.7 with
modest spread (std 70.3 / 25.8) — same ballpark, confirms §25's single-seed read wasn't a fluke,
now with proper multi-seed rigor.

**Reading against the trained DQN:** the 2-city trained DQN's own 5-seed numbers (§21, same
`city_1` fallback city) were mean -2030.4 (std 515.0), best-round -43.9. So on 2-city/3-city: DQN
mean is still worse than both baselines (same direction as 7-city, §25), but DQN best-round (-43.9)
now clearly beats both baseline means (-472.4, -240.7) with proper multi-seed grounding on the
baseline side too — the "peak-competitive, average-not" pattern from §25 holds up, not an artifact
of the baseline side being single-seed.

**Not yet done:** a 3-city-roster trained-DQN multi-seed number to compare directly against these
baselines (the 3-city `fedavg` runs already exist in `results/run_2026_08_11-13_21_10_*` etc. but
haven't been pulled into a table here) — since 3-city evaluates identically to 2-city, this
comparison would likely just reproduce §21's 2-city read rather than add new information, so may
not be worth the effort. Item (4) from §25 (audit which past results used the true `city_5_holdout`
vs. the `city_1` fallback) remains open and is now more clearly scoped: only 7-city results are on
a true holdout; both 2-city and 3-city results in this entire document are `city_1`-fallback,
in-distribution evaluations.

## 30. First real test of the newly-decoupled `--disable_neighbor_attention` flag: mean-pooling is
    actively worse than using no neighbor info at all, on one seed

**2026-08-15.** Until today, `--disable_head_fix` controlled two unrelated things at once: the
aggregation-time choice (masked-head weighted average vs. naive uniform averaging) *and* the
network-forward-time choice (`NeighborAttentionQNetwork`'s masked attention over `neighbor_obs`
vs. simple masked mean-pooling) — a naming collision, not a deliberate coupling, confounding every
past masked-head ablation (§9-§12, §20, §23) with an attention-vs-pooling comparison nobody
intended to run. Split into two independent flags in the working tree this session (not yet
committed as of this writeup): `--disable_head_fix` (aggregation-only) and
`--disable_neighbor_attention` (network-forward-only, threaded through
`ParallelFederatedServer.neighbor_attention` / `_client_worker`). This section is the first run
that actually exercises `--disable_neighbor_attention` as an independent knob.

**Setup:** `environments_c1_4` (2-city: `city_1`+`city_4`), `--dueling --n_step 3`, `fedavg`,
single seed (3), 20 rounds each, via `analyse/run_concurrent_batch.sh` (`MAX_CONCURRENT=2`) at
`results/neighbor_ablation_2city.log`. Three trained variants plus two rule-based baselines added
for reference (`--baseline_controller`, default eval seed 12345, single deterministic episode
each — a quick consistency check against §29's proper 5-seed numbers, not a new baseline
measurement):

| variant | comm condition | network | final-round reward | best-round reward | across-round mean |
|---|---|---|---:|---:|---:|
| A1 `max_pressure` | n/a (rule-based) | n/a | -236.27 (single eval) | — | — |
| A2 `fixed_time` | n/a (rule-based) | n/a | -469.03 (single eval) | — | — |
| B `no_neighbor` | fully isolated (`p_isolate=1.0`) | attention (unused — no neighbor_obs ever arrives) | -3030.75 (r20) | **-30.69 (r6)** | -2834.90 |
| C `clean_attention` | clean (`p_link=p_isolate=p_hop_cutoff=0`) | attention | **-83.32 (r20)** | **-5.60 (r12)** | **-1182.42** |
| D `clean_pooling` | clean (same as C) | `--disable_neighbor_attention` (mean-pool) | -3911.75 (r20) | -836.87 (r7) | -3364.11 |

A1/A2 line up closely with §29's proper 5-seed baselines on this same `city_1`-fallback holdout
(`max_pressure` -240.698±25.784, `fixed_time` -472.400±70.311) — both single-eval numbers land
well within 1 std of the 5-seed means, a good sanity check that this batch's eval setup is
consistent with §29's.

**C (clean comm, attention on) is the only trained variant that beats both rule-based baselines
outright**, on final-round, best-round, and across-round mean — the first time in this project a
trained-DQN configuration has cleared that bar on *every* one of those measures rather than just
best-round (cf. §21/§29's "peak-competitive, average-not" pattern for the standard-comm-dropout
config). Clean communication plus intact attention looks like it removes a real source of the
DQN's baseline-losing problem, at least at 2-city scale, one seed.

**D (mean-pooling, otherwise identical to C) is not just worse than C — it's worse than B
(no neighbor info at all).** D's best round (-836.87) is the single worst best-round of any
trained variant here, and worse than both rule-based baselines; B's best round (-30.69), despite
having *zero* neighbor information reaching the network all run, beats both baselines and is
within 6x of C's best round. This is not "neighbor info doesn't matter" — it's a specific claim
that this codebase's mean-pooling fallback path actively hurts relative to ignoring neighbors
entirely, at least on this seed. If this holds up, it reframes the whole masked-head-ablation
history (§9-§12, §20, §23): those results were measuring attention-vs-pooling noise on top of
whatever real aggregation-fix signal existed, not aggregation alone.

**B's best round beating both baselines despite zero neighbor info is itself the well-documented
round-to-round volatility (§3, §12, §28), not evidence that neighbor info is unnecessary** — B's
own final round (-3030.75) is close to its worst, a ~100x swing from its round-6 peak within the
same single run. Read B's result as "this run happened to hit a great round," not as a stable
own-obs-only capability.

**Caveats — this is one seed, not a validated result:**
1. All three trained variants are a single seed (3). Given the volatility documented throughout
   this file, none of the final/best/mean numbers above should be treated as more than a
   directional lead until repeated across seeds — especially D's "worse than no-neighbor-info"
   claim, which is the most novel and most consequential-if-true finding here.
2. This depends on uncommitted code (`experiments/federated_training.py`,
   `federated/parallel_server.py` — the `--disable_head_fix`/`--disable_neighbor_attention` split).
   The fact that C and D produce visibly different training trajectories from the same starting
   config is itself a functional confirmation the split isn't a no-op, but the code should be
   committed and this ablation re-run post-commit before citing it as settled.
3. A1/A2 are single-episode/single-seed reference points, included only as a sanity check against
   §29's real baseline numbers, not as a replacement for them.

**Next step, not yet run:** repeat B/C/D across 5 seeds before drawing conclusions about
attention-vs-pooling as a structural effect — this is exactly the kind of one-seed pattern (cf.
§11 "clean win" → §12 "2 more seeds break the story") that has previously reversed on more data in
this project.

## 31. §30's 5-seed follow-up: the single-seed story doesn't replicate — no clean win for C, no
    clean loss for D, B/C/D are statistically indistinguishable from each other

> **⚠ ADDITIONAL CAVEAT FROM §43 (2026-08-18):** independent of this section's own multi-seed
> correction to §30, every B/C/D number here was also measured in-distribution on `city_1`, not
> the true `city_5_holdout` (same issue as §21/§29). §43 showed that gap is large enough to flip
> conclusions on its own. Nothing here has been re-checked with `--pad_to_true_holdout` yet.

**2026-08-16.** Repeated B/C/D (2-city, `environments_c1_4`, `--dueling --n_step 3`, `fedavg`)
across seeds 1, 2, 4, 5 (seed 3 already had from §30), via `analyse/run_concurrent_batch.sh`
(`MAX_CONCURRENT=3`, 12 jobs, `results/neighbor_ablation_2city_multiseed.log`). One real
complication mid-run: an ~8.5-hour wall-clock stall on the first 3 concurrent jobs (round 12 to
round 13 of `C_clean_attention_s1`, and correspondingly `D_clean_pooling_s1`/`B_no_neighbor_s2`
running alongside it) — no CPU/IO activity during the gap, clean resume afterward, exit=0 on every
job. Signature matches the Windows host sleeping and freezing the WSL2 VM along with it, not a
training bug; all 12 jobs completed without errors once the host stayed awake.

**5-seed results (round-number-keyed parsing, robust to the log-interleaving line corruption noted
in §30):**

| variant | final-round (mean±std, 5 seeds) | best-round (mean±std, 5 seeds) | across-round mean (mean±std) |
|---|---:|---:|---:|
| B `no_neighbor` | -1856.88 ± 1581.51 | -257.25 ± 317.60 | -2783.49 ± 755.30 |
| C `clean_attention` | -1740.88 ± 2334.53 | **-124.40 ± 252.51** | **-1952.84 ± 682.68** |
| D `clean_pooling` | **-826.69 ± 1724.78** | -194.00 ± 359.62 | -2784.58 ± 486.94 |

Reference, §29's 5-seed baselines on the same `city_1`-fallback holdout: `fixed_time` -472.400 ±
70.311, `max_pressure` -240.698 ± 25.784.

**Per-seed breakdown (best-round, the metric §30 leaned on most):**

| seed | B best | C best | D best |
|---|---:|---:|---:|
| 1 | -677.89 (r8) | -17.14 (r20) | -19.58 (r20) |
| 2 | -31.30 (r20) | -8.70 (r10) | -49.09 (r20) |
| 3 | -30.69 (r6) | -5.60 (r12) | -836.87 (r7) |
| 4 | -24.55 (r6) | -14.54 (r11) | -43.87 (r7) |
| 5 | -521.83 (r15) | -576.03 (r4) | -20.57 (r19) |

**|diff|/SE against §29's baselines** (this project's bar for a real, non-noise signal is ≥2):

| comparison | vs `fixed_time` | vs `max_pressure` |
|---|---:|---:|
| B final-round mean | 1.96 | **2.28 (B significantly *worse*)** |
| C final-round mean | 1.21 | 1.44 |
| D final-round mean | 0.46 | 0.76 |
| B best-round mean | 1.48 | 0.12 |
| C best-round mean | **2.97 (C significantly *better*)** | 1.02 |
| D best-round mean | 1.70 | 0.29 |

**Pairwise B/C/D on best-round: all |diff|/SE ≤ 0.73** (B-vs-D 0.29, C-vs-D 0.35, C-vs-B 0.73) —
no statistically distinguishable difference between any two of the three trained variants.

**§30's two headline claims do not replicate:**
1. *"C beats both rule-based baselines on every measure"* — false at 5 seeds. C only clears the
   significance bar against `fixed_time`, and only on best-round (2.97). Against `max_pressure` —
   the stronger baseline — C is not significantly different on any of final-round, best-round, or
   across-round mean. Seeds 2 and 5 crashed hard on C's final round (-4089.59, -4496.80), pulling
   the final-round mean and std to roughly the same bad territory as B and D; seed 3 (§30's only
   data point) was the best of the five seeds on every measure, not a representative one.
2. *"D (mean-pooling) is worse than B (no neighbor info), the most novel/consequential finding"* —
   false at 5 seeds. D-vs-B on best-round is |diff|/SE=0.29, indistinguishable from noise. D's
   seed-3 result (best round -836.87, the worst of any cell in this whole table) was itself the
   outlier among D's 5 seeds — every other D seed's best round is between -19.58 and -49.87,
   competitive with or better than C and B on the same seeds. One bad seed, not a structural
   pooling-vs-attention effect.

**What does hold up:** C's across-round mean (-1952.84) and best-round mean (-124.40) are still the
best of the three trained variants numerically, and the one clean significant result in this batch
(C beats `fixed_time` on best-round, 2.97) is a real, if partial, signal that clean communication
plus intact attention has some peak-performance edge — just not the sweeping "beats everything"
result §30 reported from one seed. **B also produced one clean significant result, in the opposite
direction**: B's final-round mean is significantly worse than `max_pressure` (2.28) — full comm
isolation reliably fails to recover a good policy by round 20, even though (per the per-seed table)
it can still hit a strong best-round in 3 of 5 seeds.

**Reframes item 8 in "Open questions" below and the reuse-caution this doc has flagged before
(§11→§12 is the standing precedent): a single seed's ablation result — however clean-looking, and
however good the mechanistic story sounds — is not evidence on its own in this project.** The
`--disable_head_fix`/`--disable_neighbor_attention` code split itself is still confirmed working
(C and D produce visibly different, seed-varying trajectories from the same starting config), that
part of §30 stands; it's the *directional conclusion* about attention vs. pooling that doesn't.

**Not yet done:** the same check at 3-city/7-city scale — deprioritized given how much the 2-city
signal weakened here; probably not worth running before deciding whether this line of investigation
is worth further compute at all.

**Where this data lives (§30 and §31 combined, 17 runs total):** batch-runner logs at
`results/neighbor_ablation_2city.log` (seed 3 + the two rule-based baselines, all four run via
`--baseline_controller`/single seed) and `results/neighbor_ablation_2city_multiseed.log` (seeds
1/2/4/5, 12 runs). Each run's own `run_dir/federated_history.json` + `run_dir/training.log` is the
underlying source for every number in both sections' tables — the batch log is a tag-prefixed
merge of all concurrent runs' stdout, `run_dir` is where each individual run's own untangled
checkpoints and history live:

| tag | seed | run_dir |
|---|---:|---|
| `A1_max_pressure` | n/a (rule-based) | `results/run_2026_08_15-22_44_28_160971` |
| `A2_fixed_time` | n/a (rule-based) | `results/run_2026_08_15-22_44_30_161054` |
| `B_no_neighbor` | 3 | `results/run_2026_08_15-20_09_49_128043` |
| `C_clean_attention` | 3 | `results/run_2026_08_15-20_09_49_128045` |
| `D_clean_pooling` | 3 | `results/run_2026_08_15-21_10_10_138403` |
| `B_no_neighbor_s1` | 1 | `results/run_2026_08_16-00_00_56_180856` |
| `C_clean_attention_s1` | 1 | `results/run_2026_08_16-00_00_56_180857` |
| `D_clean_pooling_s1` | 1 | `results/run_2026_08_16-00_00_56_180862` |
| `B_no_neighbor_s2` | 2 | `results/run_2026_08_16-01_23_34_201543` |
| `C_clean_attention_s2` | 2 | `results/run_2026_08_16-10_28_19_212398` |
| `D_clean_pooling_s2` | 2 | `results/run_2026_08_16-11_00_32_224372` |
| `B_no_neighbor_s4` | 4 | `results/run_2026_08_16-12_13_56_245104` |
| `C_clean_attention_s4` | 4 | `results/run_2026_08_16-12_46_21_256622` |
| `D_clean_pooling_s4` | 4 | `results/run_2026_08_16-12_47_15_257777` |
| `B_no_neighbor_s5` | 5 | `results/run_2026_08_16-14_32_20_287009` |
| `C_clean_attention_s5` | 5 | `results/run_2026_08_16-14_34_51_287841` |
| `D_clean_pooling_s5` | 5 | `results/run_2026_08_16-15_01_38_293963` |

All 17 `run_dir`s are currently untracked local output (`results/` is not committed) — they exist
only on this machine as of this writeup; the tables in §30/§31 are the durable record if the
directories are ever cleaned up.

## 32. §28's weight-divergence/gradient-conflict diagnostic, finally run: negative result — simple
    weight-space metrics don't predict the round-to-round reward swings

**2026-08-16.** §28 flagged "diagnose *why* federated aggregation destabilizes a shared policy
(e.g. per-round weight-divergence/gradient-conflict measurements across cities)" as the
recommended next step, ahead of guessing at more hyperparameters. Built `diagnostics/
weight_divergence.py` (new, reusable — takes any `run_dir` and two city names) and ran it against
checkpoints already on disk from §30/§31's runs (no new training needed — every round's
per-client and global `.pth` are saved by `parallel_server.py` regardless of this ablation).

**Method, extending §11's precedent (which measured per-row Q-head delta magnitude between
fix_on/fix_off) to whole-network and to cross-city *direction*, not just magnitude:** for each
round r, `delta_city = flatten(client_round_r) - flatten(global_round_(r-1))` for both cities,
then `||delta_city_1||`, `||delta_city_4||`, and `cos_sim(delta_city_1, delta_city_4)` (negative =
the two cities' updates pull the shared model in different directions that round — direct evidence
of the aggregation "dilution/conflict" §26/§28 hypothesized but never measured directly). Correlated
against that round's actual reward change (`eval_reward[r] - eval_reward[r-1]`), on 3 runs (19
rounds each) spanning both a crash-prone seed and a comparatively stable one:

| run | mean cos_sim (whole-net) | corr(cos_sim, d_reward) | corr(max_client_norm, \|d_reward\|) |
|---|---:|---:|---:|
| `C_clean_attention_s2` (late crash, round 20: -2732) | -0.077 (18/19 rounds negative) | -0.224 | -0.424 |
| `C_clean_attention_s1` (comparatively stable) | -0.082 (18/19 rounds negative) | -0.138 | -0.455 |
| `C_clean_attention` seed3 (§30's original) | -0.062 (16/19 rounds negative) | -0.051 | +0.108 |

**No reproducible predictive signal.** Correlations are weak (|r|<0.45) and, critically, **flip
sign across runs** for `max_client_norm` (-0.424, -0.455, **+0.108**) — the opposite of a
consistent "bigger/more-conflicting update this round → worse reward this round" story. Restricting
to just the dueling output heads (`advantage_head`+`value_head`, the layer §11 found the real
effect in for the masked-head question) doesn't recover a signal either — correlations stay weak
and sign-flip the same way (`cos_sim` vs `d_reward`: +0.140, -0.223, +0.031 across the same three
runs).

**One real, if secondary, structural finding survives:** whole-network cross-city cosine similarity
is consistently mildly negative (mean -0.06 to -0.08, negative in 16-18 of 19 rounds in every run
tested) — `city_1` and `city_4`'s updates are persistently, mildly opposed, not aligned, as a
constant background feature of this 2-city federation. Restricted to just the output heads, that
conflict mostly disappears (mean -0.02 to +0.05) — **the cross-city tension concentrates in the
shared backbone (attention + encoder layers used identically by both cities' different traffic
patterns), not in the task-specific output head.** This is real and consistent across all three
runs, but it's a constant, not a crash predictor: it doesn't spike before a bad round or relax
before a good one, so it explains *that* the cities are in tension, not *when* that tension turns
into a reward collapse.

**Reading:** this rules out the most obvious next hypothesis rather than confirming it. The
instability documented since §3 is not visible as an unusual weight-space event in the round that
precedes it — whatever drives a good round into a catastrophic one isn't "the clients disagreed
more than usual" or "the aggregated model moved further than usual" at the level of raw parameter
deltas. That pushes the likely mechanism toward something downstream of the weights themselves:
e.g. a small, unremarkable weight change flipping the greedy action at one or two pivotal
intersections/states (where SUMO's traffic dynamics could amplify a tiny policy change into a large
queue/waiting-time cascade), or eval-episode noise itself (few episodes, single SUMO seed per
round) rather than a genuine policy regression at all. Neither is tested here. **Caveat:** 19
autocorrelated samples per run is a small, non-independent sample for a correlation claim — the
sign-flipping across just 3 runs is itself the main evidence (a real effect should show *some*
directional consistency), not a formal power calculation. `diagnostics/weight_divergence.py` is
reusable if a future session wants to extend this to more runs/seeds or a different key filter.

## 33. §32's hypothesis (b) tested and rejected: the crashes are real, reproducible policy
    failures, not eval noise — they survive 6x more episodes almost unchanged

**2026-08-16.** §32 flagged eval-episode/SUMO-seed noise as the cheaper of two remaining
hypotheses to check. Built `diagnostics/reeval_checkpoint.py` (new, reusable — loads any
`global_round_NNN.pth` into a fresh `DQNAgent` and re-runs the same `HoldoutEvaluator` pipeline a
real training run uses, just with more episodes) and re-evaluated three checkpoints from
`C_clean_attention_s2` (§30/§31's run with the clearest late-run crash) at 30 episodes instead of
training's default 5 — 30 distinct SUMO seeds (`eval_seed_base + episode_index`) instead of 5:

| round | original (5 ep) | re-eval mean (30 ep) | re-eval std | episode pattern | verdict |
|---|---:|---:|---:|---|---|
| 10 ("good") | -8.70 | -42.16 | 188.17 | 29/30 excellent (-51 to +5), 1/30 catastrophic (-1036.76) | real good policy, one rare tail-risk episode |
| 16 ("crashed") | -3470.15 | -3389.13 | 672.84 | 1/30 good (-101.02), 29/30 bad (-2900 to -4189) | **real regression, survives averaging almost unchanged** |
| 20 ("crashed", the worst) | -4089.59 | -4094.58 | 301.45 | 0/30 good, all 30/30 uniformly bad (-3350 to -4677, tight) | **real, robustly consistent policy collapse** |

**Hypothesis (b) is rejected.** If the training-time crashes were 5-episode sampling flukes, more
episodes/seeds should have pulled rounds 16 and 20 back toward round-10-like numbers. Instead both
reproduce their original bad reward almost exactly (-3470→-3389, -4089→-4095) with 6x the sample
size — round 20 especially, where every single one of 30 different SUMO seeds lands in a narrow
bad band (std only 301, the tightest of the three). These are genuine policy failures, not
measurement artifacts.

**Secondary finding, not previously visible from 5-episode evals:** even round 10's *good* policy
has a real tail-risk failure mode — 1 in 30 seeds still produces a -1036 catastrophe despite 29
excellent episodes. The training-time 5-episode eval was likely just lucky not to sample it (or
similar rare bad seeds) in most rounds; this is a real, if rare, robustness gap in even the
best-performing checkpoints, and 5-episode training-time evals systematically under-sample it.

**Narrows the open mechanism question from §32 to hypothesis (a) alone** (a small weight change
flipping the greedy action at a handful of pivotal intersections/states, amplified by SUMO's
traffic dynamics into a large, durable queue/waiting-time regime) — or some other still-untested
weight/policy-level mechanism, but specifically *not* eval measurement noise. A natural next check
(not done here): compare round 16/20's action distributions (already logged per-episode by the
evaluator) against round 10's to see whether the bad rounds show a qualitatively different
(narrower, more degenerate) policy at specific intersections, extending §26's "not a collapsed
policy" mechanism dig — done there for a different run/roster (7-city) and not yet checked on this
2-city case. `diagnostics/reeval_checkpoint.py` is reusable for that or any other checkpoint
re-evaluation.

## 34. §33's action-flip hypothesis (a), tested via Q-gap: crashed rounds show a genuinely
    degenerate, near seed-independent policy — and it's confidence, not uncertainty, that's the
    signature of the failure

**2026-08-16.** User's question prompted this: is pure-argmax ("greedy") action selection fragile
when the top-2 Q-values are close, and would a softmax-style tie-break help? This maps directly
onto the Q-gap diagnostic the evaluator already computes (`|Q(top1)-Q(top2)|` per intersection per
episode, the same metric `diagnostics/q_gap_trend.py` was originally built around) — extended
`diagnostics/reeval_checkpoint.py` to surface it per-episode alongside reward, and re-ran the same
three checkpoints from §33 (round 10 "good", round 16 "crashed, one escape", round 20 "worst
crash", 30 episodes each = 30 different SUMO seeds per checkpoint).

**Round 16 is a smoking gun for a genuinely degenerate policy, not an eval-noise or tie-break
artifact:** 18 of its 30 episodes produced the *identical* reward, -3421.65 to two decimal places,
across 30 *different* SUMO seeds. A policy actually responding to different randomized traffic
cannot produce byte-identical outcomes across different seeds — this is direct evidence the
network locked into a fixed action sequence that ignores its own (seed-varying) observations most
of the time. Round 20 shows the same signature, weaker: 13/30 episodes identical (-4079.94).

**Q-gap correlates strongly with this — but in the opposite direction from the hypothesis under
test:**

| checkpoint | mean_gap range | corr(mean_gap, reward) | corr(min_gap, reward) | identical-reward episodes |
|---|---|---:|---:|---|
| round 10 (good) | 0.196-0.265 | -0.511 | -0.389 | 0/30 (all 30 distinct) |
| round 16 (crashed, 1 escape) | 0.145-0.990 | -0.565 | **-0.884** | 18/30 at -3421.65 |
| round 20 (worst crash) | 0.859-1.393 (uniformly high) | +0.360 | -0.060 | 13/30 at -4079.94 |

Within round 16 — the one checkpoint whose episodes actually span a wide gap range (0.14 to 0.99)
— **higher confidence (bigger gap) predicts worse reward, and the rare low-gap episodes are the
ones that escape the bad loop** (episode 11: gap 0.145, reward -101.02, the best result in the
whole run). Round 20 shows no informative variation to test against: every one of its 30 episodes
sits in the high-gap band (no escape valve sampled at all), consistent with a more completely
locked-in failure. Round 10's healthy policy has no degenerate repeats and stays in a narrow,
comparatively low-gap band throughout (0.20-0.27) — a different, more responsive regime.

**Reading:** not "near-ties cause fragile flip-flopping" (the hypothesis under test) but close to
the inverse — **the network gets *confidently* locked into a bad, repeating action loop, and
moments of relative uncertainty are what let it escape.** This is a genuinely new mechanism finding
for §32/§33's open question, more specific than "action-flip amplified by SUMO dynamics": it's not
that any small weight perturbation flips a coin-toss decision, it's that the aggregated policy can
collapse into ignoring its own inputs and repeating one confident, wrong action regardless of
traffic state. This doesn't contradict §26 (which found "not a collapsed/degenerate policy" on a
7-city run) — different roster, different run; the two findings just haven't been reconciled and
may reflect a roster-size-dependent difference in how the instability manifests.

**Practical implication for beating the rule-based baselines (ties back to the earlier "how do we
raise reward" question):** the fix implied here is close to what was originally proposed (softmax
tie-breaking) but for a different reason — not to arbitrate close calls more carefully, but to give
a confidently-locked-in-bad policy an escape hatch. Softmax or light epsilon-greedy action
selection at eval/deployment time, instead of pure argmax, is a concrete, cheap, untested next
experiment: re-run `diagnostics/reeval_checkpoint.py`-style evaluation on round 16's checkpoint
with stochastic instead of greedy action selection and see whether it reliably avoids the -3421.65
attractor. Not yet implemented — `HoldoutEvaluator._policy_action`'s `"trained"` branch currently
hardcodes `model.act(obs, explore=False)` (federated/evaluator.py:132), so this would need either a
temporary explore=True/temperature-based variant or a new policy_name branch.

## 35. Literature/tooling check: how does this project's reward, loss, and action-selection compare
    to RESCO (the benchmark its own city configs are drawn from) and the standard SOTA baselines?

**2026-08-16.** Not an experimental result from this repo — a sourced comparison against external
code/papers, prompted by §34's finding and the standing "how do we beat the rule-based baselines"
question. This project's `environments/*` city configs are literally RESCO's benchmark maps
(cologne3, ingolstadt7, grid4x4, arterial4x4 — see CLAUDE.md), so RESCO
(`github.com/Pi-Star-Lab/RESCO`, NeurIPS 2021 Datasets & Benchmarks) is the most direct comparison
available, fetched and read directly (not from memory) for this write-up. CoLight and PressLight
are well-established published baselines cited from prior knowledge, not re-verified against
source here.

**Reward function — this project trains against a reward RESCO doesn't even offer, and never tries
the one the literature argues is best-motivated.** This project uses `sumo_rl`'s default
`diff-waiting-time` (change in accumulated per-lane waiting time between consecutive steps, /100 —
inherited unchanged from the upstream `sumo-rl` package; no city config overrides `reward_fn`).
RESCO's `mdp_options/rewards.py` offers a different menu: `wait` (raw negative total wait, not
differenced), `wait_norm` (same, clipped to ±4 after /224 — RESCO's own defensive scaling, in the
same spirit as this project's `reward_clip=10.0`), `pressure` (entering-queued minus
exiting-queued — the reward MPLight actually trains against), `phase_queue`, `coslight` (a weighted
combination), and `oracle_delay` (privileged full-network vehicle time-loss, not realistic for
deployment). **PressLight (Wei et al., KDD'19) and MPLight both train against the `pressure`
reward specifically because it's provably connected to max-pressure control theory** — pressure-
based reward is argued in the literature to correlate with throughput maximization in a way
delta-waiting-time isn't. This project already has `pressure`/`max_pressure` wired in as a
*rule-based eval baseline controller* (`--baseline_controller max_pressure`) but has never used it
as the *training reward* for the DQN. That's a concrete, cheap, untested experiment: swap
`reward_fn` to a pressure-style signal (or add `sumo_rl`'s built-in `pressure`/`_pressure_reward`,
already implemented in `sumo_rl/environment/traffic_signal.py`) and see whether it changes the
crash dynamics documented in §3-§34, not just final performance.

**Action selection — this project's pure-argmax-at-eval convention is the field standard, not a
project-specific choice.** RESCO's PFRL-based `DQNAgent` (used by both IDQN and MPLight) uses
`LinearDecayEpsilonGreedy` during training and falls straight to `batch_argmax` (pure greedy) once
`self.training` is `False` (`resco_benchmark/agents/action_value/pfrl_dqn.py`) — exactly this
project's train-with-epsilon / eval-with-argmax split. **This means §34's "confidently locked into
a bad repeating action" failure mode is a real vulnerability of the standard convention shared
across this whole line of work, not a quirk of this codebase** — and a softmax/stochastic
eval-time policy (§34's proposed next step) would be a genuine departure from the field standard,
not "catching up" to something RESCO/MPLight/IDQN already do differently.

**Loss function — same family (Huber/TD), different optimizer for a documented reason.** RESCO's
`DQNAgent` uses PFRL's standard `DQN` class (Huber/smooth-L1 TD loss, PFRL's default) with plain
`torch.optim.Adam`. This project also uses Huber loss (already noted in CLAUDE.md's Phase 0 audit)
but uses `AdamW` instead of Adam — not an oversight, a deliberate fix from earlier in this project
(commit `dfadab5`) for a specific bug where plain Adam's weight-decay term slowly decayed masked-out
(never-gradient-touched) Q-head rows toward zero. Same loss family as the standard tooling; the
optimizer choice is a project-specific, tested correction, not a divergence from convention.

**Coordination architecture — this project is a genuine hybrid, not a reproduction of any one
baseline, and its central premise (cross-*city* weight sharing) has no direct RESCO/CoLight
analog.** RESCO's `IDQN` is fully independent per-intersection agents with zero sharing (closest
analog in this project: `--no_federation` combined with `--disable_neighbor_attention`). RESCO's
`MPLight` is one shared network *within a city* using the FRAP architecture (explicit phase-pair
competition embeddings, heterogeneous per-signal action spaces handled via `pair_to_act_map`/
`reverse_valid` index remapping). CoLight (Wei et al., CIKM'19 — not in RESCO's default agent set,
but the standard neighbor-attention baseline in the field) uses an index-free graph attention
network shared across intersections *within a city*, architecturally the closest published relative
to this project's own `NeighborAttentionQNetwork`. **None of RESCO's algorithms, MPLight, or
CoLight share one policy's weights *across different cities/maps*** — RESCO evaluates each
algorithm separately per scenario (Cologne, Ingolstadt, Grid4x4, ...), it does not do cross-map
federation. This project's actual central premise — one shared foundation-model policy federated
across topologically-different cities, with `action_mask`/`neighbor_mask` (not FRAP's phase-pair
structure) as the topology-generalization mechanism — has no direct equivalent in any of the three
comparisons above. The instability this whole document has been chasing since §3 is, in that sense,
a cost of doing something none of these established baselines attempt.

**Sources:** [RESCO GitHub](https://github.com/Pi-Star-Lab/RESCO), specifically
[`mdp_options/rewards.py`](https://github.com/Pi-Star-Lab/RESCO/blob/main/resco_benchmark/mdp_options/rewards.py)
and [`agents/action_value/pfrl_dqn.py`](https://github.com/Pi-Star-Lab/RESCO/blob/main/resco_benchmark/agents/action_value/pfrl_dqn.py)
(fetched and read directly, 2026-08-16); RESCO paper (Ault & Sharon, NeurIPS 2021 Datasets &
Benchmarks); PressLight (Wei et al., KDD 2019); CoLight (Wei et al., CIKM 2019, arXiv:1905.05717).

## 36. §35's experiment (b), softmax eval on crashed checkpoints: a good policy is reachable from
    the exact same weights — pure argmax just never finds it

**2026-08-16.** Tested §34/§35's proposed fix directly: added `--temperature T` to
`diagnostics/reeval_checkpoint.py` (a `SoftmaxPolicy` wrapper around the loaded agent, sampling
`softmax(Q/T)` over valid actions instead of pure argmax at eval time, non-invasive — production
`federated/evaluator.py` untouched). Ran at `T=0.2` on the same two crashed checkpoints from
§33/§34 (round 16, round 20), 30 episodes each, same seeds as the pure-argmax baseline.

| checkpoint | pure argmax (§33/§34) | softmax T=0.2 | escaped episodes (reward > -1000) |
|---|---:|---:|---|
| round 16 | mean -3389.13, std 672.84, 0/30 near-optimal | mean -3191.23, std 532.44 | 0/30 — modest, uniform improvement, no true escapes |
| round 20 | mean -4094.58, std 301.45, **0/30** near-optimal (fully locked) | mean -4018.42, std **2004.51** | **6/30 (20%)**, rewards -36 to -857 — matching C's best-known results |

**Round 20 is the headline result: a genuinely good policy is reachable from the exact same
weights that produce a uniformly catastrophic outcome under pure argmax.** Under greedy action
selection this checkpoint never once escaped the bad regime in 30 different SUMO seeds (§33).
Under softmax at the same temperature, 6 of 30 episodes land in the -36 to -104 range — on par
with this project's best-known trained results anywhere in this document (cf. §30's C best-round
-5.60). This is direct, positive confirmation of §34's diagnosis: the "crash" is the policy
confidently walking a bad deterministic path, not an inability to do well from these weights.

**But softmax is not a clean fix — it trades a lower floor for occasional escapes, roughly a wash
on the mean.** Round 20's *non-escaped* episodes got worse under softmax than under pure argmax
(-4974.04 mean vs -4094.58), because perturbing away from the locked bad trajectory without any
guidance mostly still lands in a different bad outcome, occasionally a much worse one — net effect
on the overall mean is small (-4018 vs -4095, not a reliable win). Round 16 shows a smaller,
more uniform gain (better mean, similar floor, no true escapes) — the same intervention doesn't
generalize identically to both checkpoints, plausibly because round 16's Q-gaps were already more
varied under pure argmax (§34) while round 20's were uniformly high (less "give" for a fixed
temperature to work with).

**Reading and next steps:** this is genuine evidence that the failure is a policy/inference-time
issue superimposed on otherwise-servicable weights, not (only) a training-data/weight-quality
problem — reinforcing §32's finding that weight-space metrics don't explain the crashes, and
sharpening §34's mechanism. Practical directions this opens, none tried yet: (1) multi-sample
eval/deployment — draw N stochastic rollouts and keep the best one via a cheap simulator check,
which would capture round 20's 20% escape rate without paying for the worse floor on the other 80%;
(2) temperature tuning/annealing rather than one fixed T; (3) most structurally interesting — since
epsilon has already decayed to ~0.05 by round 16-20 in this training schedule, the fact that a good
branch exists but the deterministic policy walks past it raises the question of whether training
itself has enough exploration noise late in the schedule to find and consolidate onto that branch,
which would tie this finding back into §28's still-unresolved "why does aggregation regress past
round 20" question rather than treating it as purely an eval-time fix.

## 37. §35's experiment (a) pilot result: pressure reward looks worse than `diff-waiting-time` —
    but `reward_clip=10.0` is hardcoded and almost certainly destroys most of pressure's signal

**2026-08-16/17.** `environments_c1_4_pressure/` pilot finished (2-city, seed 3, `--dueling
--n_step 3`, `fedavg`, 20 rounds, `run_2026_08_16-23_27_18_433878`), same everything as §30's
`C_clean_attention` seed 3 except `reward_fn: pressure` instead of the project default
`diff-waiting-time`.

| metric (`waiting_time`, reward-fn-agnostic — comparable across runs) | diff-waiting-time (§30 seed3) | pressure (this pilot) |
|---|---:|---:|
| best round | 3.20s (round 17) | 656.60s (round 15) |
| worst round | 2303.80s (round 1) | 2713.23s (round 1) |
| mean across 20 rounds | 571.1s (std 798.4) | 1836.6s (std 585.9) |
| final round (20) | 12.57s | 1089.41s |

**The pressure run never approaches a good policy at all — not even once in 20 rounds** (best
waiting_time 656.60s vs. diff-waiting-time's 3.20s, a >200x gap at each run's own best). It's also
*more* stable round-to-round (std 585.9 vs 798.4) — but stably bad, not stably good; no crash
dynamic, no escape either, just flat mediocrity the whole run (own reward trajectory: mean
-39405.6, std only 1715.1, never gets close to whatever "good" looks like on pressure's own scale).

**This is very likely explained by the confound flagged when the pilot was launched, not a genuine
verdict on pressure reward.** `DQNAgent.reward_clip` defaults to 10.0 (`agents/dqn.py:116`) and is
**not exposed as a CLI flag anywhere in `experiments/federated_training.py`** — confirmed by
grepping the whole training entry point, zero references outside `agents/dqn.py` itself. Every
training-time reward, regardless of `reward_fn`, gets hard-clipped to ±10 before it ever reaches
the replay buffer or a TD target (`agents/dqn.py:353-359`). `diff-waiting-time` is already
scaled (divided by 100 inside `_diff_waiting_time_reward`) to roughly fit this range by design.
Raw `pressure` (`entering_queued - exiting_queued`, unscaled vehicle counts) is not — this pilot's
round-1 unclipped *episode total* was -41482, meaning individual-tick pressure values are very
plausibly saturating the ±10 clip on nearly every tick, throughout training. If so, the network was
trained on an almost-binary "clipped high/low" signal that carries far less information than real
pressure differences do, which would fully explain uniformly-bad-but-stable performance without
saying anything about whether pressure itself is a worse reward design.

**Not a fair test as run. Two ways to fix it, neither done yet:** (1) add a `--reward_clip` CLIflag
so it can be set proportional to pressure's actual scale for this experiment (quick, but changes a
currently-hardcoded value everywhere it's threaded, worth checking nothing else assumes 10.0), or
(2) follow RESCO's own pattern (§35 — their `wait_norm` divides raw wait by 224 and clips to ±4
specifically so a differently-scaled reward still fits the same range) and add a `pressure_norm`-
style reward function that pre-scales pressure into roughly `diff-waiting-time`'s natural range
before it ever hits the existing clip, no CLI/agent changes needed. (2) is probably the smaller,
safer change. **Don't treat this pilot as a verdict on the pressure-reward hypothesis from §35 —
rerun with one of these fixes before concluding anything either way.**

## 38. §37's fix applied: `pressure_norm` pilot rerun properly-scaled — still doesn't beat
    `diff-waiting-time`, and shows the same degenerate-lock-in signature from §34

**2026-08-17.** Reran the §37 pilot with the fix: `pressure_norm` (new reward function,
`sumo_rl/environment/traffic_signal.py`, `clip(get_pressure()/10, -5, 5)`) instead of raw
`pressure`. Confirmed empirically before launching that this keeps the signal well inside both the
new internal clip (hit 0.1% of ticks) and `DQNAgent`'s existing `reward_clip=10.0` (hit 0% of
ticks) — the saturation problem from §37 is fixed. Same setup otherwise: 2-city, seed 3, `--dueling
--n_step 3`, `fedavg`, 20 rounds (`run_2026_08_17-09_51_35_580341`).

| metric (`waiting_time`, comparable across reward functions) | diff-waiting-time (§30 seed3) | pressure (§37, unscaled) | pressure_norm (this pilot) |
|---|---:|---:|---:|
| best round | 3.20s | 656.60s | 472.61s |
| worst round | 2303.80s | 2713.23s | 2960.11s |
| mean across 20 rounds | 571.1s (std 798.4) | 1836.6s (std 585.9) | 1933.0s (std 701.8) |

**Fixing the clip saturation didn't fix the underlying result — pressure_norm still never gets
anywhere near `diff-waiting-time`'s best rounds** (472.61s vs 3.20s, still two orders of magnitude
apart), and is actually marginally *worse* on mean/worst than the broken §37 pilot was, well within
noise for a single seed. This rules out §37's specific confound as *the* explanation — the clip
fix changed the signal quality substantially (confirmed above) but not the outcome, which points
toward a real difference between the reward designs on this setup rather than an artifact of the
earlier scaling bug.

**A second, unplanned finding: pressure_norm's training shows the same degenerate-lock-in signature
§34 found in `diff-waiting-time`'s crashed rounds.** 6 of 20 rounds (2, 6, 7, 13, 15, 16) have
eval-episode reward std of ~0.0000 across the default 5 eval episodes (different SUMO seeds each,
same mechanism as §34) — the same byte-identical-across-seeds fingerprint of a policy that's
stopped responding to its own (seed-varying) observations. **This is evidence the lock-in mechanism
isn't specific to `diff-waiting-time` as a reward design** — it reproduces under a differently-
scaled, differently-shaped reward too, which weakens the case that reward redesign alone would fix
it and strengthens §28's original suspicion that the cause is in the federated
aggregation/training dynamics rather than the reward function.

**Caveats, same as everywhere else in this document: single seed, one run.** Seed 3 might just be
an unlucky seed for `pressure_norm` specifically — `diff-waiting-time`'s own single-seed history in
this document (§21) shows real seed-to-seed spread too. This pilot is a real, useful negative data
point (rules out §37's confound as *the* explanation), not a final verdict on the pressure-reward
hypothesis from §35; would need multi-seed validation before treating "pressure doesn't help" as
settled, same standard applied to every other intervention here.

## 39. Item 11(a)'s recovery-finetune test: a short burst of reset exploration reliably walks
    "locked" checkpoints out of the bad regime — the strongest positive result in this document

**2026-08-17.** Built `diagnostics/recovery_finetune.py` (new, reusable): loads a checkpoint's
*weights* into a fresh `DQNAgent` and continues training via a real `ParallelFederatedServer` for
a short burst, deliberately **not** using `--resume` — that path computes `init_steps_done =
completed_round × local_episodes × steps_per_ep` specifically so epsilon keeps decaying from where
it left off (already ~0.05 by round 16-20, the exact regime §34 found the lock-in in). This script
instead resets `init_steps_done=0`, so epsilon restarts at 1.0 and decays over a fresh schedule
sized to the short recovery run (`compute_eps_decay`, `federated/utils.py`), while the network
starts from the checkpoint's actual weights, not random init. Ran 5-round recovery bursts on the
same two crashed checkpoints from §33/§34/§36 (round 16, round 20 of `C_clean_attention_s2`), same
roster/config otherwise.

| checkpoint | recovery round 1 | round 2 | round 3 | round 4 | round 5 (end) |
|---|---:|---:|---:|---:|---:|
| round 16 (partially locked) | **-22.67** | -1007.08 | -3364.83 | -721.25 | **-124.44** |
| round 20 (fully locked, worst crash) | -4426.64 | -4437.18 | -4114.01 | -4078.38 | **-243.96** |

**Both recoveries end in a good state — round 20's especially, since under pure argmax (§33) that
checkpoint never escaped the bad regime in 30 different SUMO seeds, and under eval-time softmax
(§36) it only escaped in 6/30 episodes with the non-escaped episodes getting worse on average.**
Here, actual training (not just eval-time sampling) on top of the same starting weights needed 4
rounds stuck in the bad regime before breaking out, but broke out decisively by round 5
(-243.96, waiting_time 77.14s) — and because this is *training*, that's now the network's actual
weights, not one lucky rollout: the good behavior should persist rather than needing to be
re-sampled every episode the way §36's softmax fix did. Round 16 (the less severely locked
checkpoint) recovered almost immediately (round 1: -22.67, near this document's best results
anywhere) before dipping and recovering again, ending at -124.44.

**This is the strongest positive result in this document for actually fixing (not just explaining)
the crash dynamics** — stronger than §36's eval-time softmax patch, because it changes the
underlying weights rather than requiring stochastic action selection at every deployment step.
Reframes the practical recommendation from §36: rather than (or in addition to) a
multi-sample-and-select deployment strategy, a cheap post-hoc recovery pass — detect a
crashed/locked round via eval reward or the eval-episode-std~0 signature (§34/§38), then burn a
handful of extra rounds with epsilon reset — looks like a genuinely deployable fix for this
specific failure mode.

**Caveats, consistent with the standard applied throughout this document:** single seed, single
starting checkpoint each, 5-round bursts only — not yet multi-seed, not yet tested on other crashed
rounds/checkpoints, and the two recoveries' paths look different enough (round 16 escaping almost
immediately vs. round 20 needing 4 rounds first) that the "how long does recovery take" question
is not yet answered with any confidence. Also not yet tested: whether a *shorter* burst (e.g. 2-3
rounds) is enough on average, whether this generalizes to 3-city/7-city rosters, and whether
repeatedly relapsing into a locked state during a full run (not just once, near the end) would make
this an expensive whack-a-mole rather than a one-time fix. Item 11(b) (full training-time softmax
exploration swap, replacing epsilon-greedy for the whole schedule) is now lower-priority relative
to this — the cheap option worked well enough on both tested checkpoints that the more invasive
change may not be needed.

## 40. Relapse-risk check on §39's recovery: durable for the moderately-locked checkpoint, not
    durable for the fully-locked one — the fix isn't uniform

**2026-08-17.** Extended §39's 5-round recovery bursts to 15 rounds on the same two checkpoints
(round 16, round 20 of `C_clean_attention_s2`), same `diagnostics/recovery_finetune.py`, to test
whether a recovered good state holds once exploration decays back down, or relapses the way normal
training does throughout this document (§3, §12, §28).

| round | round 16 (moderately locked) | round 20 (fully locked, worst crash) |
|---|---:|---:|
| 1-5 | -211, -3021, -2265, -1734, -2172 | -3513, -998, -41, -817, -3755 |
| 6-10 | -219, **-16**, **-49**, **-11**, -497 | -159, -3690, -84, -3852, -253 |
| 11-15 | **-18**, **-17**, -188, -725, **-17** | -760, **-19**, -1482, **-4114**, -949 |
| rounds 6-15: good/bad split (>-500 vs ≤-500) | **9/10 good**, mean -175.7 | **4/10 good**, mean -1536.1 |

**Round 16 stabilizes durably — this is the clean positive case.** After the rocky first 5 rounds,
rounds 6-15 stay in a good regime almost the whole way (9 of 10 rounds better than -500, several
at or near this document's best results anywhere), with only one shallow dip (round 14, -724.50) —
nowhere near the -1700 to -3000 range of the initial crash. This looks like genuine convergence to
a good policy, not luck holding out.

**Round 20 never stabilizes — it keeps relapsing through round 15**, including a fresh crash to
-4114.05 at round 14, essentially as bad as anything seen anywhere in this checkpoint's history.
Good and bad rounds keep alternating with no visible trend toward settling (4/10 good in rounds
6-10 vs. rounds 11-15, no improvement). Extending the recovery burst from 5 to 15 rounds did not
fix round 20's checkpoint the way it appeared to in §39's shorter test — §39's round-5 endpoint
(-243.96) was a good ROUND, not evidence the run had actually stabilized; round 20's own subsequent
rounds 7, 9, 11, 13, 14, 15 in this extended run are all back in the bad range.

**Reading: §39's fix is real but not uniform, and depends on how deeply locked the starting
checkpoint was.** Round 16 (partially locked — pure argmax still escaped 1/30 episodes, §33) fully
recovers. Round 20 (fully locked — pure argmax escaped 0/30 episodes, §33; the same checkpoint
softmax eval only rescued 20% of episodes in §36) keeps relapsing under continued training just as
it did under continued eval sampling. This tracks §34's "confidently locked" severity gradient
directly: the more completely a checkpoint had collapsed, the less a single recovery
intervention — eval-time softmax (§36) or training-time exploration reset (§39) alike — durably
fixes it, and the more it looks like the underlying aggregation dynamics (§28's still-standing
suspicion) keep pulling it back rather than the fix itself being wrong.

**Practical implication:** a recovery burst is not a one-shot cure-all — it may need to be applied
repeatedly (detect-and-recover as a running policy throughout training, not a single post-hoc pass)
for severely-locked cases, while a lighter touch might suffice for moderately-locked ones. Not yet
tested: whether *starting* training with this kind of periodic exploration-reset (rather than the
standard monotonic epsilon decay) prevents reaching the fully-locked state in the first place,
which would be a training-*design* fix rather than a post-hoc *repair* — closer to item 11(b)'s
full training-time exploration-policy change, now worth re-examining given round 20's result here.

## 41. Item 11(b) built and tested from scratch on the worst 2-city seed: periodic epsilon reset
    doesn't durably fix a bad seed either — consistent with §40, not a contradiction of it

**2026-08-17.** Built `--epsilon_reset_every N` (new CLI flag, `experiments/federated_training.py`
+ `federated/parallel_server.py`): every N rounds, each client's epsilon schedule restarts at
`eps_start=1.0` (`agent.steps_done = 0`) instead of continuing its monotonic decay — a periodic,
built-into-training version of §39/§40's one-shot post-hoc recovery burst. Mechanics: the round
number is now threaded into the existing per-round `("train", state)` message the server sends each
worker (`parallel_server.py::run()`), and each worker resets its own agent's clock when
`round_num % epsilon_reset_every == 0`. Reuses the run's existing `eps_decay` — no separate schedule
needed, confirmed fast enough to reach the floor again well before the next reset at this interval.

Tested on the worst-performing seed from §21's 5-seed validation of this project's actual
recommended config (`--dueling --n_step 3`, standard `DEFAULT_COMM_DROPOUT`, not the "clean comm"
override used in the B/C/D ablation thread) — **seed 5** (§21: mean -2747.4, best -23.5, worst
-5003.3) — with `--epsilon_reset_every 5` added, otherwise identical
(`run_2026_08_17-18_27_49_683258`).

| metric | seed 5 baseline (§21, no reset) | seed 5 + `--epsilon_reset_every 5` |
|---|---:|---:|
| mean reward | -2747.4 | -2579.8 (std 1486.8) |
| best round | -23.5 | -4.54 |
| worst round | -5003.3 | -5066.01 |
| rounds >-500 ("good") | not tabulated in §21 | **2/20 (10%)** |

**No meaningful improvement.** Mean is ~6% better, within noise for a single seed given the
round-to-round std (1486.8) is more than half the mean's own magnitude. Best round improved in
absolute terms but both were already near-optimal-scale. Worst round is marginally *worse*, not
better. Confirmed via the training log that resets fired exactly on schedule (rounds 5, 10, 15,
20) — the mechanism works as designed — but the wild alternation between near-optimal and
catastrophic rounds continued essentially unchanged through the whole 20-round run (round 4:
-23.49 → round 6: -5066.01 just one reset later; round 11: -4.54 → round 13: -3770.92 two rounds
later).

**This doesn't contradict §39's positive result — it extends §40's finding to a fresh full-schedule
run instead of post-hoc recovery from one specific already-crashed checkpoint.** §40 already showed
the same mechanism (exploration reset) durably fixed a *moderately*-locked checkpoint (round 16)
but not a *severely*-locked one (round 20, kept relapsing through 15 rounds). Seed 5 is this
project's worst-performing 2-city seed by a wide margin — the periodic-reset-from-scratch result
here (no durable fix) is the severity-dependent pattern §40 already predicted, not a new surprise.
**The standing hypothesis is now reasonably well-supported across two different test designs (§40's
post-hoc recovery and this section's from-scratch training): periodic/one-off exploration resets
help specifically when a run is moderately, not severely, destabilized — they are not a general
cure for this project's underlying round-to-round instability (§3, §12, §28's still-unresolved
"why does federated aggregation regress past round 20").**

**Not yet tried:** other reset intervals (5 may be too frequent — every reset costs several rounds
of re-exploration before the network can exploit again, which could itself be contributing to the
instability rather than fixing it; a longer interval, or resetting to a partial epsilon rather than
full 1.0, are both untested), other seeds (seed 5 alone doesn't establish whether this generalizes
even negatively — a less-bad seed might respond differently), and whether combining periodic reset
with the best-round-checkpoint-selection idea from this document's very first "how do we beat the
baselines" discussion (deploy the best round reached, whichever mechanism produced it) is more
promising than expecting the *mean* trajectory to improve.

## 42. §41 brought to all 5 seeds: `--epsilon_reset_every 5` is a clean null across the full 2-city
    validation, not just the worst seed

**2026-08-17/18.** Completed the other 4 seeds §41 didn't cover, same config as §21's baseline
(`--dueling --n_step 3`, standard `DEFAULT_COMM_DROPOUT`, `environments_c1_4`) plus
`--epsilon_reset_every 5`, via `analyse/run_concurrent_batch.sh` (`MAX_CONCURRENT=3`,
`results/epsilon_reset_5seed_batch.log`). One real-world interruption mid-batch: the host went to
sleep during seed 4's run (round 8 completed 23:04, round 9 didn't complete until 15:55 the next
day, ~17-hour gap) — same freeze-and-cleanly-resume signature as §30's incident, not a bug; the run
finished correctly from where it left off, confirmed by sequential round numbers with no restart.
Run dirs: seed 1 `run_2026_08_17-20_17_59_710286`, seed 2 `..._710290`, seed 3 `..._710289`, seed 4
`run_2026_08_17-22_22_42_744398`, seed 5 reused from §41 (`run_2026_08_17-18_27_49_683258`).

| seed | baseline mean (§21) | baseline best | reset mean | reset best | Δ mean |
|---|---:|---:|---:|---:|---:|
| 1 | -2455.2 | -13.5 | -2209.8 | -17.67 | +245.4 |
| 2 | -1667.1 | -18.9 | -2148.0 | -114.55 | -480.9 |
| 3 | -1954.8 | -142.8 | -1723.3 | -100.60 | +231.5 |
| 4 | -1327.5 | -20.8 | -1368.4 | -12.98 | -40.9 |
| 5 | -2747.4 | -23.5 | -2579.8 | -4.54 | +167.6 |
| **mean of 5** | **-2030.4** (std 515.0) | **-43.9** | **-2005.9** (std 468.3) | **-50.1** | — |

**|diff|/SE = 0.07 (mean reward), 0.18 (best round) — both far below this project's ≥2 bar for a
real signal.** Per-seed deltas are mixed in sign (3 of 5 positive, 2 negative) with no consistent
direction, exactly the pattern noise around a true null effect produces. This is now a clean,
properly-powered (5 seeds both sides) null result, not just a single-seed impression from §41:
**`--epsilon_reset_every 5` has no detectable effect, positive or negative, on this project's
standard 2-city config.**

**Reading, tying together §39-§42:** the periodic-reset idea helps specifically when a checkpoint
is *already* badly locked (§40's round-16 case: durable recovery) and does nothing reliable
otherwise — it's neither a broad improvement across typical seeds (this section) nor a reliable
fix for the worst case on its own (§41's seed 5, and §40's round-20 case both kept relapsing). The
practical takeaway for this project's actual training config: `--epsilon_reset_every` is not worth
turning on by default. It remains a plausible *targeted* tool — apply it only after detecting a
locked/degenerate round (the eval-episode-std~0 signature from §34/§38), not as a standing part of
the training schedule. Item 11(b)'s more invasive full softmax-exploration-policy swap is now even
less clearly worth building — the milder, cheaper exploration intervention already tested null in
aggregate, and the standing "why does aggregation regress" question (§28) still looks like the
higher-leverage target than further exploration-schedule tweaks.

## 43. First-ever true-holdout evaluation of a 2-city trained policy: the "best-round beats
    baselines" claim was entirely an artifact of in-distribution evaluation — it reverses
    completely once measured on a genuinely unseen city

**2026-08-18.** Merged in `debugging_andreea`'s `--pad_to_true_holdout` flag (widens a reduced
roster's shared Q-head to `city_5_holdout`'s action width up front, so `make_holdout_evaluator`
can select the real holdout instead of always falling back to an in-distribution training city —
see the merge commit and CLAUDE.md). Ran the first true-holdout 2-city training pilot in this
project's history: `environments_c1_4`, seed 3, `--dueling --n_step 3`, `fedavg`, 20 rounds,
`--pad_to_true_holdout` (`action_dim` widened 5→8). Confirmed in the log: `"Holdout city not found
under base_dir='environments_c1_4'; using 'environments' for evaluation"` — no
`"!!! NOT A TRUE HOLDOUT"` warning this time, meaning this run genuinely evaluated on
`city_5_holdout`, not `city_1`.

| condition | reward |
|---|---:|
| Trained DQN — best round (1 of 20) | **-2855.95** |
| Trained DQN — mean across 20 rounds | -6624.9 (std 1587.1) |
| Trained DQN — worst round | -9895.42 |
| `max_pressure` (also re-run with `--pad_to_true_holdout`, same true holdout) | **-0.34** |
| `fixed_time` (same) | **-2.73** |

**The trained policy's *best* round of the entire run is ~1000x worse than `fixed_time` and
~8400x worse than `max_pressure`.** This is not a subtle, statistical-noise-level gap the way
most comparisons in this document are — it's categorical. Both baselines were re-run with
`--pad_to_true_holdout` too (cheap, eval-only, no training needed) specifically so this is a fair
apples-to-apples comparison: same evaluation city on both sides, not the trained side being
penalized by an unfair target.

**This directly reverses §21/§29's headline positive claim.** Every "best-round reward -43.9 beats
both rule-based baselines" statement in this document (§21, §29, and CLAUDE.md's own "RESUME HERE"
section as of this writeup) was measured with the model evaluating on `city_1` — one of its own
*training* cities, confirmed in-distribution by §25/§29's own caveat, which was **already known and
already flagged as a caveat** but had never actually been tested against the real alternative until
now. It turns out that caveat wasn't a minor asterisk — it was load-bearing for the entire claim.
Once evaluated on a city the model never trained on, "best-round beats baselines" **does not
survive contact with a genuine holdout at all.**

**This brings the 2-city picture into alignment with the 7-city picture (§24-§29): at every
roster size tested so far, once evaluated on a true holdout, the trained DQN loses badly to both
`fixed_time` and `max_pressure`.** The 2-city roster's "it might still generalize, unlike 7-city"
framing implicit throughout §21-§42 (differentiating 2-city's best-round win from 7-city's
across-the-board loss) no longer has a basis — the only genuine difference was never re-run, until
now, on a fair target.

**Caveats, though the effect size makes them matter less than usual:** single seed (seed 3) —
worth confirming the direction holds on other seeds, though a ~1000x gap is not the kind of thing
that plausibly flips sign on a different seed the way a 2x or |diff|/SE≈1 result might. Also
single training run at 20 rounds — §28 already showed more rounds regresses rather than converges
for this general instability pattern, so there's no strong reason to expect a longer run closes a
gap this large. Not yet checked on 3-city (`environments_c1_4_6`, same in-distribution-fallback
problem per §25/§29) or repeated with the neighbor-attention-ablation conditions (B/C/D from
§30/§31) — those results likely need the same re-evaluation, though given the size of this effect,
revisiting every past 2-/3-city claim's validity is now the more urgent item than running new
ablations.

**This resolves (in the negative direction) one of the two open decisions flagged in CLAUDE.md's
"RESUME HERE" section since 2026-08-13: "does the trained DQN actually beat simple rule-based
control?"** The answer, once measured correctly at every roster size now tested, is no.

**Where this data lives:** trained-DQN run `results/run_2026_08_18-19_46_23_818099`; true-holdout
`max_pressure` `results/run_2026_08_18-22_01_39_854474`; true-holdout `fixed_time`
`results/run_2026_08_18-22_01_48_854557`. All three untracked local output, same caveat as §31's
index — this table is the durable record if the directories are ever cleaned up.

## 44. First reward-shaping pilot (7-city, `wait_weight=0.001`): no help, looks worse on this one
    seed — inconclusive given the weight choice, not a verdict on the idea

**2026-08-18.** Alongside §43, piloted `debugging_andreea`'s other new capability:
`--reward_shaping_wait_weight`, aimed at §26/§28's diagnosed 7-city gap (policy avoids gridlock but
doesn't drain queues to zero by episode end). `environments` (7-city), seed 1, `--dueling --n_step
3`, `fedavg`, 20 rounds, `--reward_shaping_wait_weight 0.001` (`run_2026_08_18-19_46_49_818365`).
Weight chosen empirically (measured `{ts}_accumulated_waiting_time`'s actual per-tick distribution
first — mean 1172, p90 3996 — same lesson as §37's clip-saturation trap, picked small enough that
the shaping term mostly stays within `reward_clip`'s ±10 range rather than swamping the base
reward).

| metric | unshaped `fedavg` baseline (§23/§25/§27, 5 seeds) | this pilot (1 seed, `wait_weight=0.001`) |
|---|---:|---:|
| mean reward | -6918.4 | **-9898.4** (std 901.7) |
| best round | -2182.0 | **-6447.99** |

**Looks worse on both metrics, on this one seed.** Not the direction the hypothesis predicted.
Round-to-round pattern is also different from the usual chaotic alternation seen everywhere else in
this document — mostly flat-bad with one round-7 partial escape (-6447.99), then back to uniformly
bad for the rest of the run; several rounds show near-zero std (e.g. round 11: std 0.0738), the
same degenerate-lock-in signature as §34/§38.

**Read this as inconclusive on the reward-shaping idea itself, not a rejection of it — the weight
chosen may simply have been too conservative to matter, or (less likely given the calibration) still
wrong in a way §37 didn't anticipate.** Single seed, single weight value, no comparison at a
stronger `wait_weight` or with `stopped_weight` added. Given how much the diagnosed 7-city queue-
draining problem matters to this project's central open question (§43 just showed the trained
policy loses badly to rule-based control at every roster size), this is worth another pass with a
larger weight before concluding reward shaping doesn't help — but not before checking whether the
degenerate-lock-in pattern showing up here again is the more fundamental blocker regardless of
reward design, consistent with §38's finding that the same signature appears under a completely
different reward function too.

## 45. §43 brought to full 5-seed rigor (2-city) plus a 3-city true-holdout check: the cleanest,
    most statistically overwhelming result in this entire document

**2026-08-18/19.** Completed item 12(a)/(b): seeds 1, 2, 4, 5 on `environments_c1_4` with
`--pad_to_true_holdout` (seed 3 already had from §43), plus a 3-city (`environments_c1_4_6`) pilot
on seed 3. Same host-sleep-during-a-long-batch interruption hit twice more (2-city seeds 2/4, and
the 3-city run, all around round 17-18 → resumed 10 hours later) — same clean-resume signature as
§30/§42, confirmed no data lost via sequential round numbers and `exit=0` on every job.

| seed | best round | mean across 20 rounds |
|---|---:|---:|
| 1 | -4676.79 | -8538.9 |
| 2 | -8262.38 | -10184.3 |
| 3 (§43) | -2855.95 | -6624.9 |
| 4 | -3470.40 | -7488.6 |
| 5 | -7124.94 | -8751.2 |
| **mean of 5** | **-5278.1** (std 2335.2) | **-8317.6** (std 1348.5) |
| 3-city (seed 3 pilot) | -3545.41 | -6111.3 |

Baselines (re-run on the same true holdout, §43): `max_pressure` -0.34, `fixed_time` -2.73.

**|diff|/SE = 5.05 (best-round) and 13.79 (mean reward) against `max_pressure`** — both far past
this project's ≥2 bar, by a wider margin than almost any other result in this document. This isn't
a borderline call requiring careful interpretation the way most §-numbered findings here are; it's
about as clean and decisive as a 5-seed comparison gets. **Every seed's best round is worse than
either baseline by three to four orders of magnitude; there is no seed where the direction is even
ambiguous.**

**3-city doesn't change the picture.** One seed on `environments_c1_4_6` lands right in the same
range as the 2-city seeds (best -3545.41, mean -6111.3) — adding a third training city doesn't
narrow the generalization gap at this scale. Not multi-seeded (lower priority given how consistent
the 2-city result already is and how expensive 3-city runs are), but nothing here suggests 3-city
would tell a different story.

**This closes out item 12 from "Open questions."** §43's finding was already large enough that a
seed flip seemed unlikely to matter; this confirms it formally. **The standing conclusion for this
entire document, updated: at every roster size and every seed tested with a genuine holdout
(2-city here, 3-city pilot here, 7-city since §24), the trained federated DQN loses to both simple
rule-based controllers, decisively.** The in-distribution "best-round beats baselines" framing that
shaped how §21/§29/§30/§31 were written should be considered fully superseded, not just caveated —
see the correction notes added to those sections. The open mechanistic question from §28 ("why does
federated aggregation produce this instability") remains exactly as open as before; this section
answers a different question (does the resulting policy generalize/compete with trivial baselines)
definitively in the negative, at every scale tested.

**Where this data lives:** run dirs for seeds 1/2/4/5 are logged in
`results/pad_to_true_holdout_2city_multiseed.log`'s `finished ... run_dir=` lines; 3-city pilot is
`results/pad_to_true_holdout_3city_pilot.log`. All untracked local output, same caveat as every
other reproducibility index in this document.

## 46. Does the architecture recommendation itself survive true-holdout evaluation? Single-seed
    check: yes for dueling+n_step over the plain baseline, but the gap to rule-based control is
    unmoved either way

**2026-08-19.** §43/§45 corrected the *evaluation* methodology but every architecture/head-fix
comparison that produced this document's standing recommendation (`--dueling --n_step 3`, §15/§19)
was itself run and ranked under the old, leaky in-distribution eval — never re-checked under
`--pad_to_true_holdout`. Ran three more seed-3, 2-city (`environments_c1_4`), true-holdout pilots
alongside the existing `--dueling --n_step 3` (head-fix on) data point from §43: plain `fedavg` (no
dueling, no n_step), dueling-only (no n_step), and `--dueling --n_step 3 --disable_head_fix`.

| config | best round | mean (20 rounds) |
|---|---:|---:|
| plain FedAvg (no dueling, no n_step) | -5705.06 | -8729.18 |
| dueling only (no n_step) | -6067.35 | -7937.38 |
| dueling+n_step, head-fix **off** | -4037.32 | -6159.30 |
| dueling+n_step, head-fix **on** (§43) | **-2855.95** | -6624.90 |
| `fixed_time` baseline | -2.73 | — |
| `max_pressure` baseline | -0.34 | — |

**The architecture ranking holds up under honest evaluation.** `--dueling --n_step 3` (head-fix on)
is still the best of the four on both metrics — not an artifact of the old leaky eval. The
masked-head fix's own contribution is smaller and mixed here (better best-round, slightly worse
mean than head-fix-off), consistent with Phase 1's own "ambiguous on mean, real peak benefit"
characterization (§11/§12) rather than a new finding.

**It doesn't matter for the standing question.** Even the best config's best round (-2855.95) is
still ~3 orders of magnitude worse than `fixed_time` and ~4 than `max_pressure` — architecture
choice moves the trained-DQN numbers around by a factor of ~2, not by the ~1000x needed to approach
either rule-based baseline. **Practical read: don't scale Phase 2 (aggregation-strategy comparison)
on the assumption a better architecture closes this gap** — every variant tested so far is still
catastrophically behind trivial control, so ranking aggregation strategies against each other right
now would only be ranking different flavors of "still loses badly."

**Caveats: single seed (seed 3) for all three new configs.** Not yet known whether the
dueling+n_step-over-baseline ranking holds on other seeds — given this document's standing pattern
of single-seed stories not replicating (§11→§12, §30→§31), this specific ranking should be treated
as provisional. **Follow-up launched same day:** extending the plain-FedAvg baseline (currently the
only one of these four configs with just one seed and no multi-seed reference point) to seeds
1/2/4/5 under the same true-holdout setup, via `analyse/run_concurrent_batch.sh`
(`results/true_holdout_baseline_5seed.log`) — so the architecture-recommendation claim above can
eventually get the same 5-seed rigor §45 already gave the "does DQN beat baselines" question.

**Where this data lives:** `results/run_2026_08_19-15_51_32_969415` (plain fedavg),
`results/run_2026_08_19-15_52_06_969729` (dueling-only), `results/run_2026_08_19-15_52_09_969822`
(dueling+n_step, head-fix off). All untracked local output, same caveat as every other
reproducibility index in this document.

## 47. §46's 5-seed follow-up: the architecture-recommendation gap over plain FedAvg does NOT reach
    significance — another single-seed story that doesn't replicate

**2026-08-19.** Completed §46's flagged follow-up: extended the plain-FedAvg (no dueling, no
n_step) true-holdout baseline from seed 3 alone to the full 5-seed set (1/2/4/5 added), same
`environments_c1_4`/`--pad_to_true_holdout` setup, via `analyse/run_concurrent_batch.sh`
(`results/true_holdout_baseline_5seed.log`). Compared against `--dueling --n_step 3`'s existing
5-seed numbers (§45).

| | best round | mean (20 rounds) |
|---|---:|---:|
| plain FedAvg, mean of 5 seeds | -6227.66 (std 2433.75) | -8798.57 (std 1345.73) |
| `--dueling --n_step 3`, mean of 5 seeds (§45) | -5278.1 (std 2335.2) | -8317.6 (std 1348.5) |
| **\|diff\|/SE** | **0.63** | **0.56** |

**Both well below this project's own ≥2 significance bar — the §46 single-seed finding does not
replicate at 5 seeds.** Per-seed spread explains why: plain-FedAvg seed 4 was catastrophic (best
-10124.05, its worst seed by far) but seed 5 was actually *better* than three of the five
dueling+n_step seeds (best -3396.76, beating dueling+n_step's own seeds 1/2/5). The two
distributions overlap enough that "dueling+n_step beats plain FedAvg" is not a supportable claim at
this rigor, even though it looked clean on seed 3 alone. **This is the same standing pattern as
§11→§12 and §30→§31 — yet another single-seed architecture story that doesn't survive multi-seed
scrutiny once true-holdout evaluation is the yardstick.** Note this is a different (and weaker)
finding than §15/§19's original dueling/n-step wins, which were multi-seed themselves but measured
under the old in-distribution eval — this section doesn't re-litigate whether dueling+n_step helps
*in-distribution* (still true, per §15/§19), only whether it helps *on a true holdout*, which is
now genuinely unresolved rather than confirmed.

**Practical read, updating §46's:** there is currently no config — architecture or aggregation
strategy — with a statistically supportable claim to distance itself from a trivial `fedavg`
baseline once evaluated on a true holdout, and none come remotely close to `fixed_time`/
`max_pressure` regardless. The standing recommendation `--dueling --n_step 3` should be treated as
"still our best guess, not a confirmed win" rather than settled. **This reinforces, more strongly
than §46 did, that scaling to Phase 2 (comparing aggregation strategies against each other) is
premature** — there isn't yet a validated non-trivial baseline to build that comparison on top of.

**Where this data lives:** run dirs logged in `results/true_holdout_baseline_5seed.log`'s
`finished ... run_dir=` lines (seeds 1/2/4/5); seed 3 is `run_2026_08_19-15_51_32_969415` (§46).
All untracked local output, same caveat as every other reproducibility index in this document.

## 48. First direct test of §28's open question: is the confidently-locked degenerate policy an
    aggregation-specific effect, or does independent (no-federation) training show it too?

**2026-08-24.** §34 characterized crashed federated rounds as a genuinely degenerate, confidently-
locked policy (near-zero eval-episode reward variance, e.g. std ~0.07-2 in §33/§44) — but never
established whether federated weight-averaging *causes* this lock-in or whether it's just generic
DQN/SUMO training instability that federation happens to inherit. `--no_federation` (each client
trains fully independently, aggregation skipped entirely, `federated/parallel_server.py`'s
`eval_named_states` evaluates each city's own local model on the holdout separately) is the natural
control: same cities, same true-holdout eval, same architecture, only the aggregation step removed.

Ran the isolated variable directly against the existing federated data point: `environments_c1_4`,
seed 3, `--dueling --n_step 3`, `--pad_to_true_holdout`, 20 rounds, `--no_federation`
(`results/run_2026_08_24-12_50_13_4599`, `results/no_federation_c1_4_s3_pilot.log`) — the same
config as §43/§46's federated seed-3 run (`run_2026_08_19-15_51_32_969415`: best -2855.95, mean
-6624.90) except aggregation is skipped, so `city_1` and `city_4` each keep training on their own
frozen-apart local models the whole run, with the true holdout evaluated against both separately
every round.

| | best round | mean (across 20 rounds, both models) | min eval-episode std seen |
|---|---:|---:|---:|
| `city_1` alone (no federation) | -3710.34 | -6535.11 | 187.72 |
| `city_4` alone (no federation) | -2524.03 | -7108.22 | 24.63 |
| combined (both models, both cities) | **-2524.03** | -6821.67 | 24.63 |
| federated (aggregated global model, §43/§46 seed 3) | -2855.95 | -6624.90 | (not re-checked here) |

**Two findings, one clean and one still open.** (1) **Raw reward is not dramatically different** —
no-federation's best round is nominally slightly better than federated's, its mean slightly worse;
both are firmly in the same catastrophically-bad-vs-baselines range (§45/§46), so removing
aggregation doesn't rescue absolute performance, unsurprising given §45's baselines gap is
3-4 orders of magnitude. (2) **The near-zero-std lock-in signature never appears anywhere in this
run.** Across all 40 model-round evaluations (2 models x 20 rounds, 5 episodes each), the *lowest*
std observed is 24.63 — two orders of magnitude above the ~0.07-2 range that characterizes crashed
federated rounds in §33/§34/§38/§44. Every no-federation round instead looks like ordinary noisy
mediocrity (episode-to-episode variance in the hundreds), never the "byte-identical reward across
different SUMO seeds" pattern that defines the degenerate-lock-in failure mode. **This is evidence,
not proof, that the confident-lock-in specifically implicates aggregation** — a policy trained with
no aggregation at all can still be bad, but on this one seed it never collapses into the same kind
of confidently-repeating-bad-action trap that federated training regularly produces.

**Caveats, several stacking:** single seed (seed 3) only, one roster (`environments_c1_4`),
default `eval_episodes=5` per round (not the 30 §33 used to be fully sure a low std wasn't a small-
sample fluke — though 24.63 is far enough above the ~0-2 range that 5 episodes is probably enough
to rule it out here). `city_4` (3 intersections) and `city_1` (16, same scale as the holdout) are
structurally different training cities, so their no-federation numbers aren't a matched pair with
each other, only each against its own federated-aggregate counterpart. Given this project's
standing pattern of single-seed stories not replicating (§11→§12, §30→§31, §46→§47), **this result
should be treated as a promising first data point, not a settled answer to §28** — the natural
follow-up is the same one every other finding in this document has needed: more seeds before
trusting the direction, ideally alongside a `--temperature`/`diagnostics/reeval_checkpoint.py` check
on whichever no-federation round is worst, to positively confirm "not locked" rather than just
"no round happened to hit a low std at 5 episodes."

**Where this data lives:** `results/run_2026_08_24-12_50_13_4599/federated_history.json`
(`eval_per_model` has the per-city breakdown), `results/no_federation_c1_4_s3_pilot.log`. All
untracked local output, same caveat as every other reproducibility index in this document.

## 49. §48's 5-seed follow-up, and a correction: no-federation training DOES show the same
    confident-lock-in as federated training once checked properly — the single-seed read was an
    artifact of trusting a 5-episode std as sufficient

**2026-08-24/25.** Extended §48 to the full 5-seed set (seeds 1/2/4/5 added to seed 3, same
`environments_c1_4`/`--pad_to_true_holdout`/`--dueling --n_step 3`/`--no_federation` config, via
`analyse/run_concurrent_batch.sh`, `results/no_federation_c1_4_5seed.log`, all 4 exit=0).

**Raw reward, apples-to-apples (one model per seed, matching federated's sample size — the naive
combined-both-models number is a confound, see below):**

| | best round (mean of 5 seeds) | mean 20 rounds (mean of 5 seeds) | \|diff\|/SE vs. federated (§45) |
|---|---:|---:|---:|
| `city_1` alone, no federation | -2857.84 (std 2100.32) | -7629.43 (std 730.74) | 1.72 / 1.00 |
| `city_4` alone, no federation | -3826.02 (std 2369.41) | -7030.75 (std 1802.61) | 0.98 / 1.28 |
| federated (§45, aggregated global model) | -5278.1 (std 2335.2) | -8317.6 (std 1348.5) | — |

Both below this project's ≥2 significance bar — **no-federation is not a statistically supportable
win or loss on raw reward**, consistent with the single-seed read. (A naive "best of both models
combined" comparison gives a misleadingly significant-looking 2.33 — that's a sample-size confound,
not a real effect: pooling both cities' rounds gives no-federation 40 "shots" per seed against
federated's 20, so its max is expected to be higher purely from order statistics. Always compare
per-model, not pooled, against a single-model baseline.)

**The lock-in question is where this correction matters.** §48 used training-time 5-episode
`std_reward` as a quick screen and found nothing near the ~0.07-2 range federated crashed rounds
show, concluding (tentatively, single-seed) that aggregation might specifically cause the lock-in.
At 5 seeds, the same 5-episode screen still finds nothing below std=5.64 across all 200 model-round
evaluations — **but §33 already established that 5-episode std is not trustworthy enough on its own
to rule lock-in out**, and this dataset proves the point directly. Reran the single lowest-std
candidate (seed 5, `city_4`, round 2: 5-episode training-time std=5.64) through
`diagnostics/reeval_checkpoint.py --episodes 30 --pad_to_true_holdout` (needed adding
`--pad_to_true_holdout` support to that script first — it never got the flag when
`--pad_to_true_holdout` became the standard way to run anything after §43, so it couldn't load a
Q-head from any post-§43 checkpoint until now; fixed, `diagnostics/reeval_checkpoint.py`):

```
mean_reward=-9586.24  std_reward=1.5775
per_episode_reward: only two distinct values across 30 different SUMO seeds,
  -9584.47 (x14) and -9587.60 (x16) -- a spread of 3.13 out of ~9586
mean_gap=9.07  min_gap=1.21 (consistent across every episode)
```

**This is the exact confident-lock-in signature §33/§34 defined for federated crashed rounds** —
near-identical reward regardless of SUMO seed, moderate-not-huge Q-gap, no rare escape in 30 tries.
And it isn't isolated: `city_4` seed 5's rounds 1-5 are all tightly clustered (-9594 to -10190,
progressively falling 5-episode std) before a real escape at round 6 (-3472) — the same
multi-round-lock-then-escape shape §39/§40 studied in federated checkpoints.

**Corrected conclusion: the confident lock-in is not aggregation-specific.** A completely
independent, never-aggregated single city can get just as confidently locked into a bad repeating
policy as a federated one. This is the opposite lean from §48's tentative read, and **reframes
§28's open question** — "why does federated aggregation produce this lock-in" was itself probably
the wrong framing; the lock-in looks like a more fundamental property of DQN training against this
SUMO reward/action-space setup (consistent with §37/§38's finding that the same signature
reproduces under a completely different reward function too), which federated aggregation inherits
rather than causes. Doesn't rule out aggregation making it *worse* or *more frequent* — that
comparison needs a matched lock-in-rate count across the 5x2x20=200 no-federation model-rounds vs.
the federated runs' rounds, not yet done — but "aggregation is the root cause" is no longer a
supportable hypothesis on this evidence.

**Caveats:** one 30-episode confirmation on one candidate round — didn't repeat this on the other
low-std candidates (seed 2 `city_1` rounds 2/3, std 21.18/21.33; seed 3 `city_4` round 1, std 24.63)
or on a genuinely mid-range-std round as a negative control, so "5.64 was real, everything above it
is fine" isn't itself confirmed, just plausible. Same standing caveats as §48: 2-city roster only,
`city_1`/`city_4` aren't a matched pair with each other. Given how much this correction changes the
practical read of §28, this thread (which specific mechanism triggers the lock, and whether
aggregation changes its frequency/severity even if it isn't the root cause) is now a stronger
candidate for further compute than extending to another roster size blind.

**Where this data lives:** `results/no_federation_c1_4_5seed.log`, run dirs
`results/run_2026_08_24-18_18_09_{65147,65150,65151}` (seeds 1/2/4) and
`results/run_2026_08_24-22_45_28_110824` (seed 5); seed 3 is `run_2026_08_24-12_50_13_4599` (§48).
30-episode reeval: `results/reeval_no_fed_worst_round.log`. All untracked local output, same caveat
as every other reproducibility index in this document.

## 50. §49's open follow-up answered: aggregation does NOT measurably change the confident-lock-in's
    frequency — federated and no-federation show statistically indistinguishable lock-in rates

**2026-08-25/26.** §49 left one thing unmeasured: even though the confident-lock-in signature isn't
aggregation-*caused* (a completely independent no-federation city locks in just as readily as a
federated one), aggregation could still change how *often* it happens. Built the matched count §49
asked for: same screen (5-episode training-time `std_reward`), same absolute threshold (<50),
applied identically to both populations, then confirmed every candidate with a real
`diagnostics/reeval_checkpoint.py --episodes 30 --pad_to_true_holdout` run (not just trusting the
cheap screen, per §33/§49's own lesson that 5-episode std alone isn't reliable enough to call it).

**Populations compared:** the same 5-seed, 2-city (`environments_c1_4`), `--dueling --n_step 3`,
`--pad_to_true_holdout` runs already used in §45 (federated, 5 seeds × 20 rounds × 1 aggregated
model = 100 model-rounds) and §49 (no-federation, 5 seeds × 20 rounds × 2 independent per-city
models = 200 model-rounds) — no new training, all checkpoints already existed. (One correction made
during setup: the federated seed-3 run cited in §45's table is `run_2026_08_18-19_46_23_818099`
[§43's original run, `dueling: True`] — not `run_2026_08_19-15_51_32_969415`, which the fidings
prose elsewhere cites in a different context [§46's "plain fedavg" architecture-comparison arm,
confirmed `dueling: False` from its `training.log`]. Verified all 10 run dirs' actual CLI args
directly from `training.log` before using any of them, not just trusted prose cross-references.)

**Step 1 — screening rate (free, no new compute, both from existing `federated_history.json`):**
applying the identical `std_reward < 50` threshold to both populations gave 7/100 federated
model-rounds (7%) and 13/200 no-federation model-rounds (6.5%) as lock-in candidates — already
close before any confirmation step.

**Step 2 — 30-episode confirmation of all 20 candidates** (7 federated + 13 no-federation; ~15-20
min/checkpoint at 3-way concurrency, `analyse/run_lockin_reeval_batch.sh`, new, reusable for any
future matched-candidate reeval batch): classified each as **LOCKED** if the 30 episodes collapsed
onto a small number of near-identical reward values (≤12 distinct values, matching or tighter than
§33/§34's own confirmed round-20 case) or **not locked** if rewards were genuinely spread across the
full range (many distinct values, no repeats).

| | candidates screened (std<50) | confirmed LOCKED | confirmed not locked | rate over all model-rounds |
|---|---:|---:|---:|---:|
| federated (100 model-rounds) | 7 | 7 | 0 | **7/100 = 7.0%** |
| no-federation (200 model-rounds) | 13 | 12 | 1 | **12/200 = 6.0%** |

The one non-lock-in case (no-federation, seed 5, `city_1`, round 13) is a clean negative control:
21 distinct values across 30 episodes, spanning -321 to -4809 (no repeats), i.e. a policy that
actually responds to its (seed-varying) observations — exactly what a non-degenerate round should
look like, confirming the classification method isn't just rubber-stamping everything as locked.
Every other candidate showed 1-12 distinct values, most (10/19) showing 1-3 — e.g. `fed_s4_2`
(1 distinct value across all 30 episodes), `nofed_s2_3c1`/`nofed_s2_13c4`/`nofed_s5_4c4`/
`nofed_s5_5c4`/`nofed_s4_1c4` (also 1 distinct value each).

**Statistical comparison (two-proportion, pooled SE, same convention as every other comparison in
this document):** p_fed=0.070, p_nofed=0.060, pooled SE=0.0298, **|diff|/SE = 0.34** — nowhere near
this project's ≥2 significance bar. **Aggregation does not measurably change the confident
lock-in's frequency**, at least not at a magnitude this sample size could detect. Combined with
§49, this closes out §28's original question about as fully as it's going to get on this roster
size: the lock-in is a property of DQN training against this SUMO reward/action-space setup,
federated aggregation neither causes it nor makes it appreciably more or less frequent.

**Caveats:** this is a floor, not a census — only the 20 candidates below the std<50 screen were
confirmed; any lock-in hiding above that threshold in either population (possible in principle,
though every confirmed case here had 30-episode std under 65, well inside the screen, and §33's own
highest-CV confirmed case (round 20, ~7% CV) is matched or beaten by every case here) wouldn't be
counted. The screen itself was applied identically to both populations though, so even the
uncorrected *candidate* rate (7% vs 6.5%, before any 30-episode confirmation) already told the same
story — the confirmation step mainly ruled out the classification being an artifact of the cheap
screen, not the direction of the result. Same standing 2-city-only, `city_1`/`city_4` not a matched
pair caveats as §48/§49. A natural (not yet done) extension: repeat at a different roster size
(3-city or 7-city) to see if the null holds there too, though given how close 7%/6% already are and
how expensive each 30-episode reeval is (~15-20 min), this is a lower-priority spend than most other
open items in this document right now.

**Where this data lives:** `results/lockin_rate_reeval_2026_08_26/` (20 reeval logs +
`candidates.txt` recording exactly which checkpoint each candidate came from), driver script
`analyse/run_lockin_reeval_batch.sh` (new, reusable). All reeval logs are copied out of
`/tmp` into this results dir specifically so they survive past this session, unlike most of this
document's other "untracked local output" citations.

## 51. Does escaping the confident lock-in actually close the gap to baselines? Mostly no — one
    striking exception found, zero new compute

**2026-08-26.** Direct follow-up to §50, and to §26's older open question about what actually
drives the 1000-8400x baseline gap. §50 established that confident lock-in happens at roughly the
same ~6-7% rate whether or not aggregation is involved — but does *not* by itself establish how much
of the overall reward gap that lock-in accounts for. Tested directly, reusing existing
`federated_history.json` data from all 5 federated + 5 no-federation seed runs (§45/§49) — zero new
training or eval compute.

**Bucketed all 300 model-rounds (100 federated + 200 no-federation) by 5-episode `std_reward`** (a
noisy but free proxy for locked-vs-not, per §49's own caveat) and compared mean reward/waiting-time/
stopped/arrived against the `max_pressure`/`fixed_time` true-holdout baselines (§43):

| bucket | n | mean reward | mean waiting_time | mean stopped | mean arrived |
|---|---:|---:|---:|---:|---:|
| low-std (<50, mostly confirmed LOCKED per §50) | 20 | -9364.0 | 1779.3 | 1000.8 | 263.8 |
| mid-std (50-200) | 104 | -8977.3 | 1642.8 | 1009.0 | 304.5 |
| high-std (>200, NOT locked) | 176 | -6659.5 | 1226.9 | 849.5 | 553.4 |
| `max_pressure` baseline | — | -0.34 | 2.91 | 1.0 | 1462.0 |
| `fixed_time` baseline | — | -2.73 | 6.97 | 10.0 | 1439.0 |

**Escaping the lock-in helps, but only modestly — locked (-9364) to not-locked (-6660) is a ~29%
improvement, nowhere close to bridging the gap to either baseline (still 2400-3500x worse on
reward).** This directly answers the question against the optimistic reading: **confident lock-in
is not the main driver of the overall baseline gap** — it's a real, well-characterized failure mode
(§33/§34/§48-50) layered on top of a much larger, still-unexplained deficiency that afflicts locked
and unlocked rounds close to equally. This is consistent with, and sharpens, §26's older finding
("not a collapsed/degenerate policy... points at residual end-of-episode congestion") — the
policy's ceiling problem is broader than the lock-in phenomenon that's absorbed most of this
document's mechanism-hunting attention since §32.

**One striking exception, found by sorting all 300 model-rounds by reward:** exactly one —
`nofed_seed5_city_1_round_013` (`results/run_2026_08_24-22_45_28_110824/clients/city_1_round_013.pth`,
the same checkpoint §50 used as its confirmed-not-locked negative control) — lands anywhere near
baseline territory: reward -126.1 (5-ep training-time eval), waiting_time 71.96, stopped 69.2,
arrived 1382.6 (94.6% of `max_pressure`'s throughput). Still meaningfully worse than either baseline
(25x on waiting_time, not 2500x), but in a completely different regime from every other checkpoint
ever evaluated in this project — the next-best of all 300 rounds is -1215.8, nearly 10x worse. Its
30-episode reeval (§50) is less dramatic but still consistent with a real, if fragile, competent
policy: mean -2688.24 over 30 episodes (min -321.64, max -4808.77, 21/30 distinct values, no lock-in
signature) — worse than the 5-episode number suggested (likely partly a lucky 5-seed draw, per
§33's standing lesson that few-episode evals overstate good rounds) but still by far the best 30-
episode mean recorded anywhere in this document.

**Not yet investigated — the natural next step:** what's actually different about this one
checkpoint? Candidates worth checking directly, no new training needed: (a) diff its weights against
the immediately adjacent rounds (`city_1_round_012.pth`, `city_1_round_014.pth`) from the same run to
see if it's a sharp, isolated spike or part of a real trend; (b) inspect its action distribution and
per-intersection Q-gaps (same tooling as §26/§34) to see whether it looks qualitatively different
from typical checkpoints or just a less-unlucky sample of the same policy family; (c) check whether
`city_1`'s neighboring rounds in the *federated* (aggregated) run ever come close, or whether this is
specific to the no-federation condition — `city_1` never aggregates with `city_4` in this run, so if
this is a real, findable "good" region of weight space, it's worth knowing whether federation's
averaging step would have destroyed it.

**Caveats:** single checkpoint, single seed, single city — could be a fragile fluke rather than a
reproducible "good" region, and n=1 is a weak basis for any strong claim. The bucketed comparison
above is a global pattern across 300 rounds so is on firmer ground than the single-checkpoint
observation, but the std-based locked/not-locked split is still the same imperfect proxy §49 already
flagged as unreliable in absolute terms (only directly confirmed for the 20 candidates checked in
§50, not for all 300 rows bucketed here) — read the bucket comparison as indicative, not as a
replacement for a properly confirmed census.

**Where this data lives:** no new files — reuses `federated_history.json` from the run dirs already
cited in §45 (federated) and §49 (no-federation).

## 52. §51's outlier checkpoint inspected: a genuine isolated escape reached by an ordinary-sized
    gradient step, not a stable basin — and its own no-federation model's best-of-100 beats every
    federated seed's best-of-100

**2026-08-26.** Direct follow-up to §51's single striking exception
(`nofed_seed5_city_1_round13`). Two zero-new-training-compute checks, reusing existing checkpoints
and `federated_history.json` data.

**(1) Weight-space diff against immediate neighbors** (`torch.load` + L2 distance,
`city_1_round_011.pth` through `_015.pth`, same run):

| round transition | L2 weight distance | 5-ep reward at destination round |
|---|---:|---:|
| r11 → r12 | 2.97 | -7486.25 |
| r12 → r13 | 2.76 | **-126.10** |
| r13 → r14 | 1.95 | -4071.14 |
| r14 → r15 | 1.80 | -8501.01 |

**Round 13 is a genuine, isolated escape, not a stable basin the training process settled into.**
The full round 9-17 trajectory: -8600 → -9204 → -9406 → -7486 → **-126.10** → -4071 → -8501 → -8873
→ -9452 — a sharp one-round spike immediately relapsing back toward the same catastrophic range it
came from. Critically, **the weight movement producing this spike (L2=2.76) is unremarkable** —
essentially the same magnitude as every neighboring step (1.80-2.97), not a discontinuous jump.
**This means a genuinely good, near-competent policy is reachable by perfectly ordinary gradient
steps in this setup — it just isn't retained.** This is the same "confidently locked, rare
low-confidence moments let it escape" mechanism §34 characterized via Q-gap at the single-checkpoint
level, now visible at the training-trajectory level: escapes happen, but the very next update
(similarly ordinary in size) typically walks straight back into the bad regime rather than
consolidating the improvement. Directly explains why "just train longer" doesn't help (§28) — more
rounds means more chances to pass through a good region, not more chances to stay there.

**(2) Fair (matched-n, per-model, not pooled — same convention §49 established) best-of-100
comparison across the three populations already used throughout §45/§49/§50/§51:**

| population | n | best-of-100 (5 seeds × 20 rounds each) |
|---|---:|---:|
| `city_1` alone, no-federation | 100 | **-126.10** |
| `city_4` alone, no-federation | 100 | -1698.66 |
| federated (aggregated global model) | 100 | -2855.95 |

Unlike §49's pooled-both-models comparison (correctly flagged there as a sample-size confound),
this is apples-to-apples — 100 model-rounds on every side. **Both independent no-federation models'
best-ever round beat the federated model's best-ever round, on identical sample sizes.** The best
federated round anywhere in this document's 5-seed 2-city true-holdout data ranks only 14th out of
all 300 model-rounds evaluated across every run in §45/§49 combined; every round ranked 1-13 is
no-federation.

**Read with real caution, more than most findings in this document:** a best-of-N comparison is an
extreme-value statistic, not a mean — it doesn't admit the same |diff|/SE significance convention
used everywhere else here (max is far noisier/more outlier-driven than a mean under repeated
sampling), and this entire ranking is dominated by the single round-13 spike from part (1) above.
This is one run's worth of evidence, not independently replicated, and sits squarely in this
project's standing "single-seed/single-run story doesn't replicate" pattern (§11→§12, §30→§31,
§46→§47, all cautionary examples of exactly this kind of promising-looking result reversing on more
data). **Do not read this as "no-federation has a higher ceiling than federated" as a settled
claim** — it's a lead worth a multi-seed matched-pair replication (same seed, same city, federated
vs. no-federation, compare each seed's best-of-20 head to head) before trusting the direction, not
a result to build on yet. Notably, this would be a *different* claim from §49/§50 (mean reward and
lock-in *rate* were both statistically indistinguishable between federated and no-federation) — this
is specifically about the tail/ceiling, which those comparisons weren't designed to detect.

**Not yet done:** action-distribution and per-intersection Q-gap inspection of the round13
checkpoint itself (same tooling as §26/§34) to characterize *what* it's doing differently from its
neighbors, beyond the aggregate reward/waiting-time numbers already in §51.

**Where this data lives:** no new files — reuses checkpoints already cited in §50/§51
(`results/run_2026_08_24-22_45_28_110824/clients/city_1_round_0{11,12,13,14,15}.pth`) and
`federated_history.json` from the run dirs cited in §45/§49.

## 53. §52's checkpoint inspected further: round13's escape shows the exact §34 signature (low
    Q-gap, diverse actions) at the whole-round level, not just within a single checkpoint's episodes

**2026-08-26.** Direct follow-up, zero new compute — the 5-episode training-time eval already
recorded per-episode, per-intersection Q-gaps and action counts for every round in
`federated_history.json` (`eval_per_model[...]['q_gaps']` / `['action_counts']`), so round13 and its
neighbors could be compared without any new SUMO runs.

| round | reward | mean Q-gap | min Q-gap | max Q-gap | avg distinct actions/intersection | avg dominant-action fraction |
|---|---:|---:|---:|---:|---:|---:|
| 11 | -9406.42 | 3.8403 | 0.7023 | 10.4731 | 7.50 | 0.633 |
| 12 | -7486.25 | 5.0825 | 0.2316 | 18.2924 | 7.75 | 0.713 |
| **13** | **-126.10** | **0.1379** | **0.0610** | **0.2427** | 8.00 | **0.378** |
| 14 | -4071.14 | 0.9180 | 0.1646 | 3.6800 | 7.88 | 0.348 |
| 15 | -8501.01 | 7.1902 | 0.1745 | 21.7129 | 7.56 | 0.701 |

**Round 13's Q-gap is 30-50x lower than every neighboring round** (0.14 vs. 3.8-7.2), and its
dominant-action fraction is the lowest of the five (0.378 vs. 0.63-0.71 for the fully bad rounds) —
i.e., the network is markedly *less* confident and spreads its action choices more evenly across
each intersection's available actions during the escape round than during the locked ones. Round 14
(the partial-recovery round immediately after) shows the same pattern at intermediate strength
(Q-gap 0.92, dominant-fraction 0.348) before round 15 fully relapses to high-confidence,
high-dominant-fraction, catastrophic-reward territory.

**This is exactly §34's mechanism** — "the network gets confidently locked into a bad, repeating
action loop, and moments of relative uncertainty are what let it escape" — **replicated at a new
level of analysis.** §34 established this within a single fixed checkpoint's 30 evaluation episodes
(different SUMO seeds, same weights); this section shows the identical low-confidence-enables-escape
signature *across training rounds* (different weights, same city, same evaluation protocol) on a
completely different run (no-federation, `city_1`, this document's first inspection of this failure
mode outside a federated context). Strengthens §49/§50's conclusion that this is a fundamental
property of DQN training against this SUMO setup, not an artifact specific to how or when it was
checked: the same confidence/lock-in relationship shows up whether you hold weights fixed and vary
the SUMO seed (§34) or hold the SUMO seed protocol fixed and vary the weights round-to-round (here).

**Practical implication, sharpened from §34's original one:** since a low-Q-gap state is both **(a)
reachable by an ordinary gradient step** (§52) and **(b) the direct correlate of the only
near-competent round found anywhere in this document's 300 evaluated model-rounds** (§51/§52), the
open question shifts from "does uncertainty help escape" (now confirmed twice, independently) to
"why doesn't training preferentially move toward and stay in low-Q-gap regions" — i.e. why does an
ordinary DQN/SUMO training trajectory keep drifting back into high-confidence bad regions rather
than consolidating a low-confidence good one once found. §34 already flagged softmax/stochastic
eval-time action selection as an untested lever for this; a new, more direct one this section
suggests: some form of confidence-regularization or entropy bonus *during training* (not just at
eval time) that discourages the network from collapsing to a high-Q-gap, high-dominant-action state
in the first place — untested, would need a new training-time flag (e.g. a Q-value entropy penalty
term in the loss), not yet implemented anywhere in this codebase.

**Caveats:** single round, single city, single seed, no-federation only — the exact same "n=1,
needs replication" caveat as §51/§52. The Q-gap/dominant-fraction numbers here come from the
5-episode training-time eval, not the 30-episode gold-standard reeval §33 established as necessary
before trusting a single checkpoint's numbers fully — though the *contrast* between round13 and its
neighbors is large enough (30-50x on Q-gap) that this is unlikely to be a small-sample artifact the
way a marginal call would be.

**Where this data lives:** no new files — reuses `results/run_2026_08_24-22_45_28_110824/federated_history.json`
(`eval_per_model[...]['city_1']['q_gaps']` and `['action_counts']` for rounds 11-15), same run
already cited throughout §50-§52.

## 54. `--q_entropy_weight` implemented and piloted: first training-time intervention targeting §34's
    confident-lock-in mechanism directly — promising single-seed signal, not yet validated

**2026-08-26.** Direct action on §53's suggested lever. Implemented a new training-time
regularization term in `agents/dqn.py::DQNAgent.optimize()`: `loss -= q_entropy_weight *
mean_batch_entropy(softmax(Q_masked))`, computed on the online network's Q-values for the current
training batch (masked to each sample's valid actions, matching `_mask_q`'s existing convention).
0.0 (default) is an exact no-op — the entropy term is skipped entirely, not just multiplied by
zero. Wired through both the parallel (`--parallel`, primary) and sequential training paths;
`--q_entropy_weight` is the new CLI flag. Unlike §34/§36's softmax-eval idea (which only helps at
*deployment*, after training is already done), this acts *during* training, directly rewarding the
network for not collapsing into the high-Q-gap, high-dominant-action state §53 characterized.
Smoke-tested with a 1-round run before committing to real training compute (clean exit, ordinary
loss magnitude, no NaN/explosion) and confirmed the existing test suite's 3 pre-existing failures
(`gym_test.py`/`pz_test.py`, unrelated to `agents/dqn.py`) are unaffected — 19/22 tests pass either
way.

**Pilot (single seed, matching this project's standing cheap-screen-before-scaling convention —
§37/§44 used the same pattern): 3 weight values, seed 3, `environments_c1_4`, `--dueling --n_step 3
--pad_to_true_holdout`, 20 rounds each, `analyse/run_concurrent_batch.sh`,
`results/q_entropy_pilot_s3.log`, all exit=0.** Compared against the existing seed-3 baseline
(§43/§46, same config, `q_entropy_weight=0` implicitly, never re-run — reused):

| condition | best-round | mean (20 rounds) | min 5-ep std | rounds with std<50 |
|---|---:|---:|---:|---:|
| baseline (qew=0, §43/§46) | -2855.95 | -6624.90 | 45.2 | 1/20 |
| `qew=0.001` | **-2183.01** | **-5462.07** | 140.6 | **0/20** |
| `qew=0.01` | -4164.60 | -7177.24 | 33.7 | 1/20 |
| `qew=0.05` | **-1591.34** | **-5206.92** | 124.4 | **0/20** |

**Two of three weight values (0.001 and 0.05) beat baseline on both best-round and mean reward, and
neither hit a single round with 5-episode std below 50 anywhere in 20 rounds** — the baseline's one
low-std round (round 4, std=45.2) was confirmed as a genuine lock-in via 30-episode reeval in §50.
This is the first training-time intervention tested anywhere in this document that shows both a
reward improvement AND a reduction in the raw incidence of the low-std screening signal
simultaneously — §41/§42's `--epsilon_reset_every` was a clean null on reward, and no other training-
time lever (pressure reward §37/§38, FedProx §14, server momentum §18) has targeted the lock-in
mechanism this directly.

**The middle value (`qew=0.01`) was worse on both reward measures and did NOT avoid low-std rounds**
— a non-monotonic result across the three weights tested, which could mean either a narrow
"sweet spot" not centered on 0.01, or (more likely, given this project's standing pattern) that
n=1-seed comparisons here are simply noisy enough that 0.01's apparent badness isn't meaningful
either. No way to distinguish these from this pilot alone.

**Read with the same standing caution as every other single-seed result in this document
(§11→§12, §30→§31, §46→§47): promising, not proven.** A 3-value, 1-seed screen is exactly the kind
of result that has reversed on more seeds every previous time it's been tried here. Before treating
`--q_entropy_weight` as a real fix (or even a real lead) rather than a lucky seed-3 draw, the
natural next step is a 5-seed validation of the two promising values (0.001 and 0.05) against the
same seed-3/§45 baseline convention, ideally with true-holdout `--pad_to_true_holdout` eval
throughout (already used here) and, for whichever value survives, a 30-episode
`diagnostics/reeval_checkpoint.py` confirmation on its lowest-std round the way §50 did, to verify
the absence of low training-time std actually reflects an absence of lock-in rather than just a
shift in what the cheap screen catches.

**Where this data lives:** `results/q_entropy_pilot_s3.log`, run dirs
`results/run_2026_08_26-13_29_07_437380` (qew=0.001), `..._437383` (qew=0.01), `..._437384`
(qew=0.05). Baseline reused from `results/run_2026_08_18-19_46_23_818099` (§43/§46, no new run).
All untracked local output, same caveat as every other reproducibility index in this document.

~~**IN PROGRESS as of this writeup:** 5-seed validation launched for both promising values~~ **Done
— see [§55](#55-54s-5-seed-q_entropy_weight-validation-complete-reward-gain-doesnt-reach-significance-but-the-lock-in-rate-reduction-does--a-split-result-not-a-clean-win).**
Neither weight's reward improvement reaches significance at 5 seeds, but `qew=0.05`'s lock-in-rate
reduction does (z=2.71) — a split result, not the clean win this pilot's single-seed numbers
suggested.

## 55. §54's 5-seed `--q_entropy_weight` validation complete: reward gain doesn't reach
    significance, but the lock-in-rate reduction does — a split result, not a clean win

**2026-08-27.** Completed the 5-seed validation §54 launched (seeds 1/2/4/5 added to the seed-3
pilot, both promising weight values, same `environments_c1_4`/`--dueling --n_step 3
--pad_to_true_holdout`/20-round config, `results/q_entropy_5seed.log`, all 10 jobs exit=0). Baseline
5-seed numbers reused directly from §45 (no re-run needed — same config, `q_entropy_weight=0`).

| condition | best-round mean (5 seeds) | mean-reward mean (5 seeds) | \|diff\|/SE vs baseline |
|---|---:|---:|---:|
| baseline (§45) | -5278.1 (std 2335.2) | -8317.6 (std 1348.5) | — |
| `qew=0.001` | -4653.1 (std 1501.5) | **-7071.0** (std 1305.1) | 0.50 (best), **1.49 (mean)** |
| `qew=0.05` | -4932.3 (std 2140.3) | -7636.0 (std 1411.6) | 0.24 (best), 0.78 (mean) |

**Neither weight clears this project's ≥2 significance bar on reward, on either metric.** `qew=0.001`
gets closest (1.49 on mean) — a real lead, numerically better than baseline in 3 of 5 seeds
(seed3, seed5 clearly, seed2 modestly), worse in 2 (seed1, seed4) — but not statistically
supportable at 5 seeds. This is the same pattern this document has hit repeatedly (§11→§12,
§30→§31, §46→§47): a clean-looking single-seed pilot (§54: both weights beat baseline on both
measures, seed 3 only) doesn't survive multi-seed scrutiny.

**The lock-in-frequency claim tells a different, more interesting story.** Counting rounds with
5-episode std < 50 (the same cheap screen §49/§50 established and confirmed via 30-episode reeval)
across all 5 seeds:

| condition | low-std rounds | rate | z vs baseline |
|---|---:|---:|---:|
| baseline | 7/99 | 7.1% | — |
| `qew=0.001` | 3/100 | 3.0% | 1.31 (not significant) |
| `qew=0.05` | 0/100 | **0.0%** | **2.71 (significant)** |

**`qew=0.05` significantly reduces the raw incidence of the confident-lock-in signature (z=2.71),
even though that reduction doesn't translate into a statistically supportable reward improvement.**
This isn't a contradiction — it's exactly consistent with §51's earlier finding that confident
lock-in is a *secondary* factor in the baseline gap, not the main driver (locked vs. not-locked
rounds differed by only ~29% mean reward in §51, both still 2400-3500x worse than baselines).
Suppressing the lock-in signature is real and measurable, but the lock-in was never the dominant
reason this project's trained policies lose so badly to `fixed_time`/`max_pressure` — so fixing it,
even successfully, doesn't move the headline number much. `--q_entropy_weight` is a genuine,
verified, working intervention on the *specific mechanism* it targets; it just isn't a fix for the
larger baseline gap, because that gap isn't primarily caused by this mechanism.

**Reading:** don't adopt `--q_entropy_weight` as a standing default based on this — the reward
case isn't there. It remains legitimately interesting as confirmation that the confident-lock-in
mechanism (§34, §53) is real, training-time-controllable, and separable from whatever else is
producing the much larger baseline-comparison gap. Not yet done: a 30-episode
`diagnostics/reeval_checkpoint.py` confirmation on `qew=0.05`'s runs specifically (§54 flagged this
as the natural follow-up for whichever value survived — arguably still worth doing to confirm the
z=2.71 lock-in-rate reduction holds up the way §50 confirmed the original cheap-screen counts, even
though the headline reward result is now null).

**Where this data lives:** `results/q_entropy_5seed.log` (all round-by-round data for both
weights × seeds 1/2/4/5), `results/q_entropy_pilot_s3.log` (seed 3, from §54), baseline reused from
§45's run dirs. All untracked local output, same caveat as every other reproducibility index here.

**Session note:** this validation was launched by a separate session that was subsequently lost
(deleted/disconnected) before it could write up the result — the training processes themselves
kept running independently on the same machine and completed normally, and all prior work through
§54 was already safely committed to git, so nothing was actually lost. This section completes that
session's queued next step.

## 56. §55's follow-up done, but it overturned the thing it was supposed to confirm: the
    std<50 screen has substantial false negatives on BOTH arms, undermining the reported
    z=2.71 lock-in-rate reduction

**2026-08-27.** Set out to do the cheap confirmation §55 flagged as its natural next step (a
30-episode `diagnostics/reeval_checkpoint.py --pad_to_true_holdout` check on `qew=0.05`'s
screened-zero lock-in claim). Since `qew=0.05` had literally 0/100 rounds below the std<50
threshold, there was nothing to confirm directly — so, following §49's precedent (checking the
single lowest-std round in a batch even when nothing crossed the threshold, which is exactly how
§49 caught a real lock-in the screen had missed), checked the 4 rounds closest to the threshold
from above (5-episode std 56.9-107.0, all screened as "not locked") instead. **3 of 4 turned out to
be genuine confident lock-ins at 30-episode rigor** — reward collapsing onto 7-10 distinct values
with <1% relative spread across 30 different SUMO seeds, matching §50's own confirmation criterion
exactly. Only one (std=107.0, 30-ep spread 4.09%, the widest of the four) was a clean negative,
matching the escape signature instead.

**This meant the qew=0.05 side of §55's comparison was undercounting, so for a fair test, ran the
identical check on baseline's own near-threshold rounds (5-episode std 51.0-63.7, the closest four
above baseline's own 7 already-screened rounds).** Result: **4 of 4 also confirmed genuine lock-in**
(1-12 distinct values, spread 0.00-2.0% of magnitude).

| checkpoint | 5-ep std (screen) | 30-ep std | distinct values (of 30) | rel. spread | verdict |
|---|---:|---:|---:|---:|---|
| qew05 s2 round7 | 56.94 | 22.32 | 10 | 0.52% | **locked** |
| qew05 s1 round1 | 93.81 | 34.69 | 7 | 0.97% | **locked** |
| qew05 s2 round6 | 106.44 | 3.83 | 7 | 0.09% | **locked** |
| qew05 s5 round9 | 106.96 | 136.37 | 10 | 4.09% | not locked (escape) |
| baseline round2 | 51.04 | 0.09 | 2 | 0.002% | **locked** |
| baseline round18 | 56.23 | 2.23 | 2 | 0.04% | **locked** |
| baseline round9 | 56.46 | 59.51 | 12 | 1.97% | **locked** |
| baseline round15 | 63.65 | 0.00 | 1 | 0.00% | **locked** |

**Corrected minimum-bound counts** (originally-screened-and-confirmed, §50/§55, plus these newly
confirmed near-threshold rounds — a lower bound, since only 4 of each side's ~95-96 remaining
non-flagged rounds were checked, not all of them): baseline 11/99 (11.1%), qew=0.05 3/100 (3.0%),
**z = 2.24** (pooled-proportion two-proportion test) — still above this project's ≥2 bar, but a
much weaker and less clean result than §55's reported z=2.71, which rested on an unexamined 0%
floor for qew=0.05 that this check now shows was an artifact of the screen, not a real zero.

**Reading:** §55's headline "qew=0.05 eliminates the lock-in signature entirely (0/100, z=2.71)" is
**not reliable as stated** — don't cite that specific number going forward. The qualitative
direction (qew=0.05 reduces lock-in rate) still holds at the weaker corrected bound (z=2.24), but
the true rate on both sides is unknown until a full 30-episode recount is done on every non-flagged
round (baseline: ~95 remaining; qew=0.05: ~96 remaining) — expensive (each 30-episode reeval takes
several minutes; ~190 checkpoints is a multi-hour batch), not yet done, and not started without
being asked given the cost. This is the same std<50-screen-has-false-negatives lesson §49 already
established once (there, on a batch where the screen found *zero* candidates at all); this section
shows the same failure mode recurs even in a batch where the screen found *some* candidates — the
threshold is not a reliable dividing line anywhere near it, on either side of the comparison.

**Unrelated, cheap side-finding from the same session (queued together with the above): phase-
switching frequency is not the primary driver of the baseline gap, on the one non-locked checkpoint
tested.** Built `diagnostics/behavior_compare.py` (reuses `HoldoutEvaluator`'s existing per-tick
`action_log`, no new training) to compare the trained DQN's phase-switch rate and dominant-action
fraction against `max_pressure` on identical holdout episodes, using qew=0.05's best-ever checkpoint
(seed3 round9, reward -1591.34 — not a locked round). Result: switch rate 0.366/tick (DQN) vs.
0.368/tick (`max_pressure`) — essentially identical — and dominant-action fraction 0.445 vs. 0.312 —
same order of magnitude, not the qualitative difference a "thrashing" or "degenerate concentration"
explanation would predict. Yet mean waiting time is 593s (DQN) vs. 2.91s (`max_pressure`), a 204x
gap. **Rules out switching frequency/action-concentration as the primary driver on this checkpoint**
— the trained policy switches about as often as `max_pressure` does, it's simply picking worse
actions when it does, which narrows (but doesn't yet answer) the open §51/§55 question of what
*does* primarily drive the gap now that both confident lock-in (§51) and switching behavior (this
section) are ruled out as the main cause.

## 57. Reward-clip saturation ruled out for the default reward; the actual failure is a small,
    persistent, whole-episode per-tick deficit at every intersection that compounds toward the
    end of the episode — the first real quantitative handle on the primary-driver question

**2026-08-27.** §37's writeup asserted `diff-waiting-time` was "already scaled... to roughly fit"
`DQNAgent.reward_clip`'s hardcoded ±10 range (`agents/dqn.py:116`), by reasoning about the /100
scaling in `_diff_waiting_time_reward`'s source — but that scaling applies to the *accumulated*
waiting time before differencing, not to the diff itself, and was never actually measured the way
raw `pressure`'s saturation was (§37 found 26% of ticks exceeding the clip for unscaled pressure).
Built `diagnostics/measure_reward_clip_saturation.py` to check this directly, since if the default
reward is *also* getting clip-saturated during congested episodes, that would be a training-signal-
destruction mechanism affecting every single non-pressure experiment in this document, with nothing
to do with confident lock-in (§51) or switching behavior (§56).

**Two checks, both on the true holdout city (`city_5_holdout`, `--pad_to_true_holdout`), replaying
the exact trajectory that produces this project's catastrophic reported rewards:**

| policy | per-tick reward mean | std | first-half mean | second-half mean | episode-total reward | % ticks \|r\|≥10 |
|---|---:|---:|---:|---:|---:|---:|
| trained DQN (baseline round15, confirmed fully locked, §56) | -0.899 | 0.920 | -0.548 | -1.250 | -10356.62 | **0.00%** |
| `max_pressure` | -0.0000 | 0.093 | -0.0007 | 0.0006 | -0.34 | **0.00%** |

(`max_pressure`'s -0.34 total exactly matches the known baseline number from §43/§45, confirming
the diagnostic replicates the real eval faithfully.)

**Reward-clip saturation is definitively ruled out as a contributor to the baseline gap — 0.00% of
ticks exceed the clip on either policy, even for a maximally-locked, catastrophically-bad
checkpoint.** The assumption in §37 was correct, it just had never actually been checked; now it
has, on the actual holdout trajectory rather than an assumption from reading the source. Combined
with §56 (switching behavior ruled out) and §51 (lock-in ruled out as the *primary* driver), this
closes off three plausible mechanisms.

**What the data does show, precisely for the first time: the trained policy's per-tick reward is
never catastrophic (max magnitude 3.67, nowhere near the ±10 clip) — it's just persistently,
ubiquitously slightly negative, at every one of the 16 intersections, every tick of the whole
episode, while `max_pressure` stays essentially at zero throughout.** And the deficit compounds:
second-half-of-episode mean (-1.250) is 2.3x worse than first-half (-0.548), while `max_pressure`
shows no such trend (flat at ~0 in both halves). **This is the first quantitative confirmation of
the qualitative "not a collapsed policy, residual end-of-episode congestion" reading from way back
in §26** — congestion genuinely does build up disproportionately later in the episode under the
trained policy, consistent with small per-tick suboptimality compounding over time in a way
`max_pressure`'s locally-reactive control never lets happen (SUMO traffic doesn't self-clear;
un-drained queues only grow).

**Reading:** the primary-driver search has now ruled out three candidate mechanisms (lock-in §51,
switching behavior §56, reward-clip saturation this section) and converged on a more precise
picture of the failure itself: a chronic, small, compounding per-tick control deficiency, not an
intermittent catastrophic event. This points toward a genuinely different class of hypothesis than
anything tested so far — something about long-horizon credit assignment or value-function quality
(does the Q-function actually predict the compounding cost of a slightly-wrong action correctly?),
not exploration, reward scale, or degenerate action selection.

**Follow-up in the same session: the deficit is present essentially from round 1, not something
training grows in.** Same checkpoint run's `global_round_001.pth` (same seed, same holdout episode)
gives mean=-0.714, first-half=-0.393, second-half=-1.035 (2.6x compounding) — qualitatively the
same signature as round 15's fully-locked checkpoint (mean=-0.899, 2.3x compounding), just somewhat
less severe in total (-8223.57 vs -10356.62). **Training 15 more rounds barely changes the
qualitative picture and only modestly worsens the total** — this rules out "training progressively
learns/reinforces the bad behavior" and instead points to "the network never learns effective
moment-to-moment reactive control the way `max_pressure` gets for free from its hand-computed
pressure signal, from essentially the start of training onward." That reframes the open question
again: not "what does training do wrong" but **"why can't this architecture/observation design
learn `max_pressure`-level local reactivity at all, at any point in training"** — worth checking
next whether the *observation* actually contains what `max_pressure` uses (approach/exit queue
counts) with enough fidelity, since `max_pressure` computes its action directly and exactly from
that signal while the DQN has to learn the same mapping indirectly through reward alone.

**Where this data lives:** all output from `diagnostics/measure_reward_clip_saturation.py`, run
directly (not batched — single-episode, ad hoc), not persisted beyond this write-up's tables.

## 58. Paper-readiness check: pulled RESCO's actual published numbers on the exact same scenario
    this project trains on, and found two real, previously-unaccounted-for confounds — a training
    budget ~2.5-35x smaller than what published DQN methods need to converge, and an evaluation
    protocol RESCO's own numbers never attempt

**2026-08-27.** Before pushing further on the primary-driver mechanism hunt, checked whether this
document's headline finding ("trained DQN loses to rule-based baselines by 3-4 orders of magnitude")
is consistent with the actual published literature its own city configs are drawn from — §35 already
compared reward/loss/architecture/eval-convention choices against RESCO, but never pulled RESCO's
*numbers*. Fetched and read the RESCO paper directly (Ault & Sharon, NeurIPS 2021 Datasets &
Benchmarks — `datasets-benchmarks-proceedings.neurips.cc`, Table 1 and Figure 4).

**Scenario match confirmed exactly:** this project's `city_4` (`environments/city_4`,
`cologne3.net.xml`) has 3 traffic-light-controlled intersections (`grep -c "<tlLogic"`), matching
RESCO's "Cologne Corridor" benchmark task precisely (RESCO's Cologne Single/Corridor/Region
correspond to this project's `cologne1`/`cologne3`/`cologne8` net files by intersection count: 1/3/8).

**RESCO's Table 1, Cologne Corridor column (best episode, averaged over 5 seeds):**

| algorithm | Avg. Delay (s) | Avg. Trip Time (s) | Avg. Wait (s) | Avg. Queue |
|---|---:|---:|---:|---:|
| IDQN (independent DQN, closest architecture to this project's non-federated case) | 23.99 | 59.0 | 8.5 | 0.87 |
| IPPO | 22.13 | 57.45 | 7.37 | 0.76 |
| MPLight (shared DQN across intersections, pressure reward) | 83.65 | 123.93 | 46.25 | 5.4 |
| FMA2C | 25.37 | 61.68 | 11.3 | 1.68 |

**RESCO's own DQN-family methods land within a factor of ~2-3 of each other and of the rule-based
dashed-line baselines shown in Figure 4 for this scenario — nothing like this project's 200x+ gap
on `waiting_time`** (trained DQN 593s vs. `max_pressure`'s 2.91s, §56's behavior-compare checkpoint;
or -0.34 vs. -9500ish reward on the true holdout, §43/§45). **This is strong evidence the
catastrophic gap documented throughout this project is not an inherent property of DQN-based
traffic-signal control on these networks** — published implementations on the identical network
topology get DQN into the same ballpark as rule-based control, sometimes better (MPLight beats
`max_pressure` by 11-19% per §35/RESCO's own comparison), never orders of magnitude worse.

**Two concrete, previously-unmeasured confounds found by reading the paper's methodology, not
just its results table:**

1. **Training budget.** RESCO's Figure 4 states IDQN and MPLight reach their best performance by
   roughly **episode 100**; FMA2C and IPPO need "many times" more (~1,400 episodes) to converge.
   This project's standard training budget is `--rounds 20 --local_episodes 2` = **40 total
   episodes per city** — well under a third of even IDQN's own ~100-episode convergence point, let
   alone what a coordinated/attention-based method might need. Every number in this document's
   50+ sections about the trained-DQN-vs-baseline gap was measured at this budget. **Nobody has
   checked whether the gap persists, narrows, or closes at a training budget matched to what
   published methods actually need.**
2. **Evaluation protocol mismatch.** RESCO's Table 1 is entirely **in-distribution** — every
   algorithm is trained and evaluated on the identical scenario (there is no cross-scenario
   holdout anywhere in RESCO, MPLight, or CoLight; §35 already established this for architecture,
   worth restating for evaluation protocol specifically). This document's headline "loses
   decisively" claims (§43/§45 onward) are specifically about **true cross-city holdout**
   evaluation (train on cities A+B, eval on unseen city C) — a harder, non-standard protocol none
   of the compared literature attempts. That doesn't invalidate the true-holdout finding (it's
   still a real, correctly-measured result, and generalization-to-unseen-topology is this
   project's actual research premise, unlike anything in RESCO) — but it means **this document has
   never actually produced a true apples-to-apples comparison against RESCO's own numbers**, only
   an evaluation of a fundamentally harder task RESCO doesn't test.

**Action taken, not yet complete as of this write-up:** built `environments_c4_only/` (single-city
roster, `city_4`/cologne3 only, matching `environments_c1_only`'s existing convention) and launched
a `--no_federation` training run on it alone, budgeted to substantially exceed RESCO's ~100-episode
IDQN convergence point, **evaluated in-distribution** (on `city_4` itself via the natural
single-city fallback, deliberately *not* `--pad_to_true_holdout` this time — for this specific
comparison the in-distribution fallback is the *correct* protocol, not the bug it was flagged as
for generalization claims in §25/§43) — the first genuinely matched comparison (same scenario, same
protocol, comparable training budget) against RESCO's Table 1 numbers anywhere in this document.
Result to follow in the next section once the run completes.

## 59. §58's comparison run finished: under a budget-matched, in-distribution protocol, the trained
    DQN is competitive with `max_pressure` and clearly beats `fixed_time` — the "loses to baselines
    by 3-4 orders of magnitude" framing does not hold once the two confounds §58 identified are
    controlled for. The single most important correction in this document.

**2026-08-27.** The `environments_c4_only`/`--no_federation`/120-round/240-episode run (cologne3
alone, `--dueling --n_step 3`, seed 42) finished cleanly. 24 training-time evals (every 5 rounds,
3 episodes each) across the run:

| | reward | waiting_time (s) |
|---|---:|---:|
| best round (10) | -1.32 | 36.9 |
| median across 24 evals | -28.6 | 164.6 |
| mean across 24 evals | -222.0 | 348.5 |
| worst round (80) | -1253.2 | 1500.4 |

**Still highly volatile round-to-round — this document's confident-lock-in/instability findings
(§32-34/§51-53) are not contradicted, they're independently reproduced here too** (best -1.32 to
worst -1253.2 is itself a >900x swing within one single-city, non-federated, budget-matched run).
What's different is the *ceiling*: this run repeatedly reaches near-optimal rounds (10, 30, 35, 70,
75, 85, 105 all land in the -1 to -34 reward range), something no run anywhere in §43 onward's
true-holdout numbers ever came close to.

**Confirmed the best round (85) with a robust re-evaluation, not just the noisy 3-episode
training-time screen:** `diagnostics/reeval_checkpoint.py` at 15 episodes gives mean_reward=-1.55
(std 1.27, range -0.44 to -3.31 — a *tight*, non-locked, genuinely-good distribution, not a fluke
draw). `diagnostics/behavior_compare.py` at 10 episodes, same checkpoint, run against the rule-based
controllers on the identical (in-distribution) city:

| policy | mean_reward | mean_waiting_time (s) |
|---|---:|---:|
| trained DQN (round 85) | **-2.01** | **37.4** |
| `fixed_time` | -0.97 | 230.6 |
| `max_pressure` | -9.98 | 27.3 |

**The trained DQN beats `fixed_time` by 6.2x on waiting time and is within 1.4x of `max_pressure`
— on *reward* it actually beats `max_pressure` (-2.01 vs -9.98, though the two aren't directly
optimizing the same signal, so this specific comparison is weaker evidence than the waiting-time
one).** Against RESCO's own published IDQN number for this exact scenario (Cologne Corridor,
Table 1, §58): Avg Wait 8.5s vs. this run's 37.4s — about 4.4x worse, a real remaining gap, but
categorically different from the 200x+ gap this document has reported at every true-holdout
comparison since §43.

**This is the single most important correction anywhere in this document.** The headline claim
repeated from §43 through §57 — "the trained DQN loses to rule-based baselines by 3-4 orders of
magnitude, at every roster size, full stop" — **does not hold once the two confounds §58 identified
are controlled for** (adequate training budget, in-distribution evaluation matching what RESCO and
the rest of the literature actually test). It was never a wrong measurement — every true-holdout
number in this document is still a correct, real result **for the specific, harder task it measured
(generalizing a 40-episode-trained policy to an unseen topology)** — but the framing that gap says
something fundamental about "DQN-based traffic signal control" or even about "this project's
implementation" was too strong. The gap is much more consistent with **undertraining plus a genuine
cross-topology generalization penalty**, both of which are normal, expected, addressable factors —
not evidence of a broken pipeline or an intractable problem.

**What this does and doesn't change:**
- Does NOT invalidate the confident-lock-in mechanism work (§32-34/§51-53) or the reward-clip/
  switching-behavior ruling-outs (§56/§57) — those are real properties of this training setup,
  independently reproduced again in this very run's own round-to-round volatility.
- DOES mean every "trained DQN loses catastrophically to baselines" sentence in this document
  (§43, §45, §47, and the RESUME HERE summary) needs to be read as **"...under 40-episode training
  evaluated on an unseen topology,"** not as a general verdict on this project's DQN or on
  federated/non-federated traffic-signal RL as a method.
- DOES reopen Phase 2 scaling as a live option — the original 2026-08-26 user decision to hold off
  on Phase 2 pending the baseline-gap investigation was made without knowing the gap was this
  sensitive to training budget; worth revisiting once more data is in.
- Reframes the paper-readiness question entirely: the strongest version of
  this project's paper is no longer "surprising negative result, DQN fundamentally fails," it's
  closer to **"federated DQN training under realistic small-budget/aggregation-drift conditions
  degrades badly, and we characterize exactly how (confident lock-in, instability, and now training-
  budget sensitivity) even though the same architecture is capable of RESCO-competitive control
  given enough budget and in-distribution evaluation."** That's a different, arguably more
  interesting and more publishable paper than the one this document was building toward through §57.

**Caveats, stated plainly:** single seed (42), single scenario (cologne3/Cologene Corridor only),
single training-budget point (120 rounds/240 episodes — not a sweep, so the *shape* of the
budget-vs-performance curve, and whether it's monotonic, is still unknown), and the round-to-round
volatility means "best round" is doing a lot of work in this comparison, same standing caveat as
`max_pressure`/`fixed_time` comparisons elsewhere in this document. This is exactly the kind of
clean single-run result this document has learned (§11→§12, §30→§31, §46→§47) not to trust without
multi-seed replication. **Next step, not yet done: repeat this same run (`environments_c4_only`,
`--no_federation`, 120 rounds, budget-matched) across 5 seeds**, and ideally also run the
*federated* 2-city version at the same extended budget to check whether federation-with-adequate-
budget also closes most of the gap, before rewriting this document's standing conclusions any
further.

**Where this data lives:** `results/run_2026_08_27-21_58_40_747849/` (training run + history),
`environments_c4_only/` (new single-city roster, symlink-based per this project's existing
`environments_c1_only`/`environments_phase0` convention).

## 60. Launched: extending §45's federated 2-city 5-seed runs to a 1.25x-RESCO-budget training
    length via `--resume`, to test whether the *true-holdout* gap (not just the in-distribution
    one §59 already closed) shrinks with adequate training

**2026-08-27.** §59 answered the in-distribution question (single city, no federation). This
launches the more important remaining one: does the **true-holdout** gap this document has reported
since §43 — train on 2 cities, evaluate on the genuinely unseen `city_5_holdout` — also shrink with
more training, when federated? Rather than retraining from round 0, resumed §45's existing 5-seed
`environments_c1_4`/`--dueling --n_step 3 --pad_to_true_holdout` runs (all already at round 20,
checkpoints on disk) via `experiments/federated_training.py --resume`, extending each to round 63
(126 total episodes/city, ≈1.25x RESCO's ~100-episode IDQN convergence point, matching §58's
budget target on the federated/true-holdout side instead of the single-city/in-distribution side).

**Exact resume mapping** (seed → original run_dir, all confirmed at round 20 before launch):
seed1→`run_2026_08_18-22_31_10_863898`, seed2→`run_2026_08_18-22_31_10_863894`,
seed3→`run_2026_08_18-19_46_23_818099` (§43's original), seed4→`run_2026_08_18-22_31_10_863897`,
seed5→`run_2026_08_19-00_14_20_889188`. Launched via `analyse/run_concurrent_batch.sh` at
`MAX_CONCURRENT=3` (this project's standard batch runner), log at
`results/pad_to_true_holdout_extended_5seed.log`. All 5 confirmed via the "Resuming from ...
completed round 20 -- continuing to round 63" log line before moving on.

**Known caveat with `--resume` on this codebase, worth flagging explicitly (not a bug introduced
here, an existing property documented in `resolve_resume`'s docstring plus one more found while
checking it): only the global Q-network weights and an *approximated* epsilon step-counter survive
a resume** (`experiments/federated_training.py`'s docstring already covers replay
buffer/optimizer-momentum reset). **The learning-rate schedule also restarts from the base `--lr`
(3e-4) rather than continuing from wherever `lr_decay=0.97` had already brought it by round 20**
(~1.67e-4) — each resumed worker is a fresh process whose agent is constructed with the base LR and
decays from there, `agent.decay_lr()` has no resume-awareness. Net effect: rounds 21+ initially
train at a higher LR than a truly continuous 63-round run would have used at that point, before
decaying back down over the next several rounds. Worth keeping in mind when interpreting rounds
21-30 specifically; unlikely to matter much by round 63.

**Not yet complete as of this write-up — results to follow once the batch finishes.**

## 61. §60's batch finished: more training budget DOES significantly improve the true-holdout
    gap too — but nowhere near as much as it did in-distribution (§59). The cross-topology
    generalization gap is real, substantial, and NOT primarily a training-budget artifact.

**2026-08-28.** All 5 resumed seeds finished cleanly (exit=0) at round 63 (126 episodes/city,
≈1.25x RESCO). Compared against §45's original round-20 numbers on the identical true-holdout
protocol:

| | best-round (mean of 5 seeds) | mean-reward (mean of 5 seeds, matched-window\*) |
|---|---:|---:|
| round 20 (§45) | -5278.1 (std 2335.2) | -8317.6 (std 1348.5) |
| round 63 (§60/§61) | **-2285.2** (std 1558.1) | **-6013.2** (std 1640.9) |
| \|diff\|/SE vs round 20 | **2.38** | **2.43** |
| `max_pressure` baseline | -0.34 | -0.34 |
| ratio, round 63 vs `max_pressure` | ~6721x | ~17686x |

(\*mean-reward for round 63 computed over rounds 21-63 only, to compare like-for-like against
round 20's "mean across all 20 rounds" rather than diluting it with the already-known rounds 1-20.)

**Both improvements clear this project's ≥2 significance bar — more training budget does help the
true-holdout generalization gap, not just the in-distribution one (§59).** Per-seed, the pattern is
consistent: every seed's best round got better (seed3 most dramatically, -2855.95→-327.10, a
genuine sustained improvement plateau across rounds 45-63, not a single-round spike — round 59 hit
waiting_time=117.91s, the best true-holdout waiting time anywhere in this document outside the
single §51/§52 n=1 outlier). **But the remaining gap to `max_pressure` is still enormous — roughly
6700-17700x, not the ~1.4x §59 found in-distribution.** Going from 20 to 63 rounds roughly halved
the gap's magnitude; closing the rest would need an amount of further training this document has no
basis yet for estimating (the improvement rate itself isn't characterized — one budget point added
to one other budget point isn't a curve).

**This is the cleanest evidence yet that §58's two confounds are not equally responsible for the
two different comparisons this document has been conflating.** In-distribution (§59): the gap was
*almost entirely* a training-budget/protocol artifact — controlling for both nearly closed it.
True-holdout (§60/§61): training budget matters (statistically significant, |diff|/SE > 2 on both
measures) but is nowhere near sufficient on its own — **the cross-topology generalization penalty
is real, large, and not explained away by undertraining.** This validates rather than undermines
this document's and this project's original research premise (generalizing a shared policy across
topologically different cities is a genuinely hard, unsolved problem) — it just means the specific
"3-4 orders of magnitude, full stop" framing from §43-§57 was measuring a mix of two effects
(fixable undertraining + a real generalization gap) without distinguishing them, and this section is
the first to separate them.

**Reading for the paper-readiness question this whole thread started from:** the strongest version
of this project's contribution is now reasonably clear — not "federated DQN fails at traffic
control" (§59 disproves that framing) and not "the true-holdout gap is just a bug" (§61 disproves
that framing too) — but **"once training-budget and evaluation-protocol confounds are controlled
for, federated DQN traffic control is competitive in-distribution but still generalizes badly to
unseen topologies, and that gap persists (though shrinks) with more training."** That's a real,
publishable, well-evidenced finding, with the mechanism work (§32-34/§51-57) as a serious
complementary thread on top of it (why the training itself is so unstable/volatile even where it
eventually reaches good rounds).

**Not yet done:** (1) a robust (15+ episode) re-evaluation of seed3's round 50/59 checkpoints,
launched but not yet returned as of this write-up — the training-time numbers above use the
standard 5-episode screen this document has repeatedly found can understate/overstate a single
round's true performance (§33/§49/§56); (2) the symmetric no-federation comparison at this same
63-round budget (launched immediately after this section, see next section if present, or check
`results/no_federation_c1_4_extended_5seed.log`) — needed to know whether federation itself still
matters at this larger budget, extending §49/§50's 20-round-budget finding (no significant
difference) to the new budget point; (3) characterizing the actual budget-vs-performance curve
(more than 2 points) before trusting any extrapolation about how much more training would be needed
to fully close the true-holdout gap.

**Where this data lives:** same 5 run dirs as §45/§60 (`results/run_2026_08_18-*`,
`results/run_2026_08_19-00_14_20_889188`), extended in place via `--resume`; batch log
`results/pad_to_true_holdout_extended_5seed.log`.

## 62. First intervention actually targeted at the true-holdout generalization gap itself: added
    `max_pressure`'s exact input signal (outgoing-lane pressure/density) to the DQN's observation
    — it was structurally absent before, not just underused

**2026-08-28.** §61 separated two effects this document had conflated: training budget (mostly
fixes the in-distribution gap, §59) and cross-topology generalization (real, large, budget-
resistant on the true holdout, §60/§61). Every intervention tried in §32-§57 targeted symptoms of
training instability, not this. Before assuming this is a deep, unfixable architectural problem,
checked something concrete and falsifiable: **does the DQN's observation even contain the signal
`max_pressure` uses?**

**Confirmed via direct code reading, not assumption: no.** `TrafficSignal.get_pressure()`
(`sumo_rl/environment/traffic_signal.py:312`) computes `#veh on outgoing lanes − #veh on incoming
lanes`, using both `self.lanes` (incoming, controlled by the signal) AND `self.out_lanes`
(downstream, discovered from `getControlledLinks`). `SumoLaneExtractor.extract()`
(`environments/federated_env.py`, the only lane-feature source for the `--parallel` federated
training path — NOT `environments/common.py`, a separate/older code path used by
`local_training.py`/`centralized.py`/`evaluate.py`/etc., left untouched here, out of scope) only
ever read `ts.lanes` — incoming lanes. **Outgoing-lane information was never extracted anywhere in
the federated observation pipeline.** The DQN could not have learned to approximate `max_pressure`
even in principle; the required input didn't exist in its observation. Confirmed this isn't a
"redundant, would help a little" fix — it's a genuine hole, the same category of finding as this
project's real bugs (§24's `fixed_time`, §25's holdout fallback).

**Implementation** (`environments/federated_env.py`, all in the `--parallel`/`federated_env.py`
path only): extended `LaneExtractor.extract()`'s return signature from 4 values
(`lanes, phase, elapsed_green, yellow_time`) to 6 (`+ pressure, out_density`), computed in
`SumoLaneExtractor.extract()` via the existing (already-implemented, previously unused by this
pipeline) `TrafficSignal.get_pressure()`/`get_out_lanes_density()`. `pressure` reuses the exact
`clip(get_pressure()/10, -5, 5)` normalization §38's `pressure_norm` reward already established
empirically (std~12, range -57..+36 on this project's RESCO configs) — same scale, different
purpose (observation feature here, reward there). `out_density` is the mean of
`get_out_lanes_density()` (already `[0,1]`-clipped per lane). Both added as two new
`LaneEncoder.GLOBAL_FEATURES` entries, so `TopKEncoder.output_dim` (= `own_dim`) grows automatically
by 2 — confirmed empirically: **`own_dim` 115→117**. All three call sites that unpack
`extract()`'s tuple (`NeighborSummaryExtractor.summarize`, `_extract_cached`, `_build_obs`) updated
to match; the two call sites that don't need the new values use `*_` so future extensions of this
tuple won't require touching them again. Verified with a raw env probe (no training): fields are 0
at reset (no traffic yet), populate with real values after a few random-action steps (e.g.
pressure=0.8, out_density=0.15 after 30 ticks) — confirmed reading real SUMO state, not a stub.

**This is a genuine architecture change, not a resumable tweak — `own_dim` changing breaks
`load_state_dict` compatibility with every existing checkpoint in this document.** Every pilot from
here on trains fresh from round 0, no `--resume` from pre-existing runs.

**Pilot launched, matching §60/§61's exact protocol for a clean before/after comparison:**
`environments_c1_4`, `--pad_to_true_holdout`, `--dueling --n_step 3`, seed 3, 63 rounds (same seed
and budget as §60/§61's best-performing seed, `run_2026_08_18-19_46_23_818099`, best=-327.10) —
same everything except the two new observation features. Not yet complete as of this write-up.

**Correction, same day, 2026-08-28 17:11 — the first pilot attempt was confounded and its result
is invalid; killed and relaunched correctly.** The first launch command omitted `--lr_decay` and
`--min_lr`. Both default to values nobody in this document has ever actually used:
`--lr_decay` defaults to **1.0 (no decay at all)**, `--min_lr` defaults to `1e-6` — every other run
in this document, including the §60/§61 baseline this pilot exists to compare against, explicitly
passes `--lr_decay 0.97 --min_lr 1e-5`. So the first attempt trained for 54/63 rounds (before this
was caught) at a **constant, non-decaying 3e-4 learning rate**, an entirely different and
uncontrolled training dynamic, not an isolated test of the new observation features. Its
round-by-round numbers looked decisively worse than the §60/§61 baseline at the same round range
(rounds 49-54 averaged -6453 vs. the baseline's -651 at the same rounds, and the run's best round
anywhere in 54 rounds was -3866 vs. baseline's -327) — **but that comparison is invalid and was not
trusted as a finding about the pressure feature.** Killed the process (PID 900537) and relaunched
identically except with `--lr 3e-4 --lr_decay 0.97 --min_lr 1e-5` explicit, confirmed via the
new run's own logged arguments (`'lr_decay': 0.97, 'min_lr': 1e-05`) and `own_dim=117` (pressure
feature still present). New pilot run dir: `results/run_2026_08_28-17_11_26_1008135`. This is the
run whose result should actually be trusted; the killed one should not be cited.

**Where this data lives:** code change in `environments/federated_env.py` (committed); corrected
pilot run dir `results/run_2026_08_28-17_11_26_1008135`. The killed, invalid first attempt's
partial data (`results/run_2026_08_28-09_38_37_900537`, 54 rounds) is left on disk for the record
but should not be used for any comparison.

**Scope cut, same day: §60's no-federation-at-63-rounds batch (item 17b) descoped from 5 seeds to
3.** Seeds 1-3 were already ~65% done (round 41-42/63) and cheap to let finish; seeds 4-5 hadn't
started yet (still queued behind `MAX_CONCURRENT=3`). Given this batch answers a lower-priority,
largely-confirmatory question (§49/§50 already found no significant federation-vs-no-federation
difference at the original 20-round budget) compared to the pressure-feature pilot's more
decisive, still-open question, and seeds 4-5 would have cost another ~10+ hours from a standing
start, let seeds 1-3 run to completion but prevented 4-5 from ever starting (a watcher kills
either process the instant it would spawn, rather than touching the batch orchestrator's job
control and risking seeds 1-3). **This means item 17b lands as a 3-seed result, below this
document's own standing 5-seed rigor bar** — read accordingly, as directional not confirmatory,
consistent with how partial-seed results are already flagged elsewhere in this document.

## 63. Corrected pressure-feature pilot finished: worse than baseline on this seed, not better —
    single-seed, needs multi-seed replication before trusting the direction either way

**2026-08-28.** The corrected pilot (§62's fix, `results/run_2026_08_28-17_11_31_1008135`, proper
`--lr_decay 0.97 --min_lr 1e-5`, same seed 3/budget/protocol as §60/§61's baseline) finished all 63
rounds cleanly.

| metric | baseline (§60/§61, no pressure feature) | pressure-feature pilot |
|---|---:|---:|
| best round | -327.10 (round 50) | -3844.45 (round 50) |
| mean, rounds 21-63 | -2906.42 | -5115.78 |

**Worse on both measures, by a wide margin — not the hoped-for improvement.** Adding
`max_pressure`'s exact input signal (confirmed structurally absent, §62) did not help this seed;
it hurt.

**Read this with real caution, in both directions, before concluding anything:**
- This is a single seed. This document has hit the "promising/discouraging single-seed result
  doesn't replicate" pattern repeatedly in both directions (§11→§12, §30→§31, §46→§47) — the same
  standard that says "don't trust a good single-seed result" also says "don't trust a bad one."
- **A subtlety specific to this comparison, not present in this document's other single-seed
  redos: `own_dim` changed (115→117), so this isn't a perfectly matched pair even at the same
  nominal `--seed 3`.** A wider observation means more parameters in the input layer, which shifts
  every subsequent draw from the same seeded RNG stream (weight init beyond the input layer,
  replay sampling order, etc.) — so "same seed, different `own_dim`" is not a true ceteris-paribus
  A/B pair the way "same seed, same architecture, one flag changed" is elsewhere in this document.
  Some unknown fraction of this seed-3 comparison's gap could be an artifact of effectively landing
  on a different draw from the training-instability process this whole document has spent 50+
  sections characterizing, not the feature itself.
- Given how volatile this training setup already is at any fixed configuration (§60/§61's own
  extended run swung >900x within one seed), a single seed moving in the wrong direction is not
  strong evidence the hypothesis is wrong — but it is also not nothing, and should not be waved
  away just because it's inconvenient.

**Honest status: the pressure-feature hypothesis is not confirmed helpful, and this one data point
suggests it may not be — genuinely uncertain pending multi-seed replication, not a case of "just
needs more seeds to show the win."** Not yet decided/started: whether to spend further seeds on
this specific feature, or treat this as sufficient signal to deprioritize it in favor of the other
levers discussed (clustered federation, few-shot calibration, wider training roster) — a call for
the next session/decision point, not made unilaterally here.

**Where this data lives:** `results/run_2026_08_28-17_11_31_1008135/federated_history.json`.

## 64. No-federation-at-63-rounds batch finished (3 seeds, per the scope cut in §60's addendum):
    extends §49/§50's "federation doesn't matter" finding to the larger budget too — still no
    significant difference, if anything a (non-significant) lean the other way

**2026-08-29.** Seeds 1-3 (`environments_c1_4`, `--no_federation`, same 63-round/`--pad_to_true_
holdout` protocol as §60/§61, trained fresh not resumed per §60's `--resume` correctness note)
finished cleanly; seeds 4-5 were killed within seconds of starting, as planned, negligible compute
lost.

| | best-round (mean) | mean-reward, rounds 21-63 (mean) |
|---|---:|---:|
| federated, 5 seeds (§60/§61) | -2285.2 (std 1558.1) | -6013.2 (std 1640.9) |
| no-federation, 3 seeds (this section) | -3690.6 (std 650.8) | -7093.2 (std 304.8) |
| \|diff\|/SE | 1.78 | 1.43 |

**Neither clears this project's ≥2 significance bar — no statistically supportable difference
between federated and no-federation training at this extended budget, same conclusion as §49/§50
found at the original 20-round budget, now extended to 63 rounds.** If anything the raw numbers
lean toward no-federation being *worse* (not better) than federated here, opposite of any
"federation is the problem" story, though the gap isn't significant so this shouldn't be
over-read either. **Read with the caveat already flagged when this batch was scoped down: n=3, not
this document's standard n=5, so directional, not confirmatory** — but consistent directionally
with the now twice-replicated (§49/§50, this section) finding that federation itself is not a
meaningful driver of this project's core instability or generalization problems, at either budget
tested.

**Where this data lives:** `results/run_2026_08_28-09_17_39_894049` (seed1),
`results/run_2026_08_28-09_17_39_894045` (seed2), `results/run_2026_08_28-09_17_39_894048` (seed3);
batch log `results/no_federation_c1_4_extended_5seed.log`.

## 65. Real bug found and fixed before it could produce a meaningless result: `ClusteredFedAvgStrategy`
    has never actually clustered by genuine per-city differences — it silently degenerates to an
    arbitrary alphabetical split, because every client's live Q-head is always the same padded
    width by this project's own architecture design

**2026-08-29.** Preparing the "one more bounded shot" agreed with the user (clustered federation,
the cheapest remaining lever targeting the actual named cross-topology-generalization problem).
`ClusteredFedAvgStrategy.assign_clusters()` (`federated/aggregation_strategies.py`) is documented
as "cluster clients by `action_dim`" — reads `info.metadata.get("action_dim")`, falling back to
the live `head.4.bias` tensor shape if metadata doesn't have it. **`ClientRoundInfo.metadata` is
never populated anywhere in `federated/parallel_server.py` (the real `--parallel` training path)
— it's always the dataclass default, an empty dict** — so this always falls through to the
head-shape fallback. But **every client's live Q-head is always padded to the same shared global
`action_dim` before training even starts** (`ActionMaskPadder`, the core mechanism that lets one
Q-head serve every topology) — so the head-shape fallback sees identical widths for every city, in
every roster, unconditionally, by the fundamental design of this whole federated system, not
something specific to `--pad_to_true_holdout`. `cluster_cities`'s tie-break on equal values is
`(action_dim, name)` sorted — with all action_dims tied, this reduces to a pure alphabetical split.

**Caught empirically before trusting it:** manually verified `cluster_cities()` should split
`environments_c1_4_6` (arterial4x4/`city_1` action_dim 5, cologne3/`city_4` action_dim 4,
ingolstadt7/`city_6` action_dim 3) as `{city_6: 0, city_4: 0, city_1: 1}` (the two closer native
widths grouped, the outlier alone). A first smoke-test launch instead logged
`{'city_1': 0, 'city_4': 0, 'city_6': 1}` — alphabetical, not similarity-based. That mismatch is
what surfaced the bug, rather than assuming the smoke test's success meant the mechanism was
correct.

**This means the one prior `clustered_fedavg` result in this document (mentioned in passing
earlier, 7-city roster, 20-round budget, 5 seeds, mean -6494.5, best mean of every strategy tested
but |diff|/SE=0.85 vs. plain `fedavg`, not significant) was also measuring an arbitrary alphabetical
partition, not genuine similarity-based clustering** — the docstring's "cluster by action_dim"
claim was never actually true for any run in this document's history until the fix below. Worth
knowing if that number gets cited: it wasn't testing what its name implies.

**Fix** (`federated/parallel_server.py`): captured each city's own native `env.max_action_dim`
(from `build_federated_env(cfg)`, before `ActionMaskPadder` widens it) in `_client_worker`, threaded
it through the worker→server IPC queue (extended the result tuple from 8 to 9 fields) into
`ClientRoundInfo(metadata={"action_dim": native_action_dim}, ...)`. Verified with a second smoke
test: cluster assignment now correctly reads `{city_6: 0, city_4: 0, city_1: 1}`, matching the
manually-computed expectation exactly. **Only the `--parallel` path (`federated/parallel_server.py`)
was fixed — `federated/server.py` (the sequential path, not used for real training per this
project's own convention) has the identical gap and was left alone, out of scope, same reasoning
as leaving `environments/common.py` untouched in §62.**

**Pilots launched, now on genuinely-functional clustering:** `environments_c1_4_6` (3-city — the
2-city roster this session has otherwise used throughout cannot test this at all: with exactly 2
cities, `cluster_cities` always produces one city per cluster regardless of similarity, identical
to `--no_federation`), seed 3, 63 rounds, `--dueling --n_step 3 --lr_decay 0.97 --min_lr 1e-5
--pad_to_true_holdout`, matching this session's established protocol. Two runs: plain `fedavg`
(no extended-budget 3-city baseline exists yet — §45's only 3-city data point was a single-seed
20-round pilot) and `clustered_fedavg --n_clusters 2`. Both pending as of this write-up.

**Where this data lives:** code fix in `federated/parallel_server.py` (committed); pilot run dirs:
`results/run_2026_08_29-07_01_27_1193341` (plain `fedavg`),
`results/run_2026_08_29-07_01_27_1193340` (`clustered_fedavg`) — both launched 2026-08-29 07:01,
both targeting round 63, logs at `/home/deea/.claude/jobs/7f50f065/tmp/c146_fedavg_pilot.log` and
`.../c146_clustered_pilot.log` (job-local, not under version control — read the run dirs'
`federated_history.json` directly if this job's tmp dir is gone by the time this is read).

**Update, 2026-08-31: the plain `fedavg` baseline finished (2 days elapsed wall-clock, including
at least one host-sleep gap — training itself unaffected, same resilience as documented elsewhere
in this file).** Best round -4294.87 (round 39), mean(rounds 21-63) -6479.05, worst -10064.61.
This is the first extended-budget (63-round) 3-city true-holdout data point this document has —
for reference, §45's single-seed 20-round 3-city pilot got best -3545.41/mean -6111.3, so more
budget did **not** obviously help here the way it did on the 2-city roster (§60/§61) — a 3-city,
single-seed comparison, not yet something to read much into. `clustered_fedavg` still running as
of this update (round 34/63) — comparison pending.

**Final update, 2026-08-31: `clustered_fedavg` finished all 63 rounds — no real improvement over
plain `fedavg`, closing out this experiment per the decision rule agreed 2026-08-29.** Full-run
comparison (both runs, same 43-round eval cadence extended to 63):

| | best round | mean(rounds 21-63) | std(21-63) |
|---|---:|---:|---:|
| `fedavg` | -4294.87 (round 39) | -6479.05 | 1750.37 |
| `clustered_fedavg` | -4218.43 (round 30) | -6866.61 | 1229.18 |

Best-round: `clustered_fedavg` marginally ahead (+76.45, ~1.8%). Mean(21-63): `fedavg` marginally
ahead (clustered is -387.56 worse, ~6%). Both differences are small relative to either run's own
round-to-round std (1200-1750) — a wash, not a signal in either direction. **This is a single seed
on each side, so no `|diff|/SE` is even computable** (that statistic needs multiple seeds; a
single run's own across-round std is not a between-seed SE) — but the agreed rule was to treat
"no real improvement" as the stopping signal regardless, and a result this close, with no
consistent direction between the two headline metrics, does not clear that bar even informally.
**Per the 2026-08-29 STRATEGIC CONTEXT decision (CLAUDE.md RESUME HERE): this is the signal to
stop chasing reward-improving training/aggregation-time interventions for the cross-topology gap.**
Combined with the now-fully-exhausted list — federation strategy (§49/50/64, null), architecture
(§46/47, null under true holdout), extra pressure feature (§62/63, negative single-seed), reward
shaping (§44, inconclusive), and now genuine clustering (this section, null) — no training-time
lever tested anywhere in this document has closed the true-holdout generalization gap
characterized in §58-61. One further lever remains queued and is a different *category* of
intervention (test-time/few-shot adaptation rather than a training-time aggregation choice, so not
covered by the stopping rule above): fine-tuning a trained checkpoint on synthetic randomized
traffic on the holdout topology itself before evaluating on its real traffic (implemented,
smoke-tested, not yet run at real settings — see `diagnostics/finetune_on_holdout.py` and
`diagnostics/generate_random_routes.py`). If that also comes back null, the honest next step is
writing up the characterized-gap paper described in the STRATEGIC CONTEXT block rather than
testing further levers.

## 66. Fine-tune-on-holdout (the queued test-time-adaptation lever) shows a large improvement —
but "large improvement over zero-shot" and "closes the gap to baselines" are very different claims,
and only the first one is true here.

Ran `diagnostics/finetune_on_holdout.py` against the finished 3-city `fedavg` baseline's
`global_round_063.pth` checkpoint (§65/§66 above, `results/run_2026_08_29-07_01_27_1193341`):
5 rounds, `--local_episodes 2`, federated across 5 independent SUMO `randomTrips.py`-generated
route files on `city_5_holdout`'s own net (grid4x4) — never touching `grid4x4_1.rou.xml`, the real
route file the eval below is scored against — `--lr 5e-5` (deliberately gentler than the 3e-4 used
for training from scratch), `--seed 3`, `--eval_episodes 5`.

| round | eval_reward (real holdout traffic) |
|---:|---:|
| zero-shot (unmodified checkpoint) | -8668.31 |
| 1 | -1321.95 |
| 2 | -636.43 |
| 3 | -696.86 |
| 4 | -1691.45 |
| 5 (final) | **-413.96** |

Final round is a **~20.9x improvement over zero-shot** (-413.96 vs -8668.31), monotonic-ish with
one dip at round 4 (consistent with this document's standing round-to-round volatility, not a new
phenomenon). This is the single largest relative improvement any lever in this document has
produced from a fixed starting checkpoint.

**But scale matters, and this is where the caution comes in.** Pulled `fixed_time`/`max_pressure`
on the exact same evaluator (same holdout config, same `episodes=5`) for a genuine apples-to-apples
comparison, rather than trusting older numbers from a possibly-different eval configuration
elsewhere in this document: `fixed_time` mean_reward=-2.73, `max_pressure` mean_reward=-0.34. The
fine-tuned checkpoint's -413.96 is still **~152x worse than `fixed_time` and ~1218x worse than
`max_pressure`** — nowhere near baseline-competitive in absolute terms. What actually happened:
the gap to `max_pressure` shrank from ~25,500x (zero-shot) to ~1,200x (fine-tuned) — roughly a 21x
reduction in the size of the gap, which is substantial, but "closing 95% of an enormous log-scale
gap" still leaves three orders of magnitude on the table. Read this the same way §59-61 taught this
document to read training-budget improvements: real, worth having, not close to sufficient alone.

**Caveats, same standing pattern as every other single-run result in this document (§11→12,
§30→31, §46→47, §62→63):** single seed, single starting checkpoint, 5 eval episodes per round (not
this project's more rigorous 30-episode confirmatory check — worth re-running the winning round-5
weights through `diagnostics/reeval_checkpoint.py --pad_to_true_holdout --episodes 30` before
trusting this number further). Not yet tested: whether this generalizes to a DIFFERENT starting
checkpoint (e.g. the `clustered_fedavg` run's best round, or a completely different seed) or holds
up at multi-seed rigor.

**Where this leaves the queued plan:** this is not a null result, so the §65 contingency (train
from scratch on a wider/more diverse roster, queued in CLAUDE.md 2026-08-31) does not trigger yet.
Next concrete steps, in order: (1) a 30-episode confirmatory re-eval of the round-5 checkpoint
(`results/finetune_holdout_fedavg_c146_round063/global_round_005.pth`) to rule out the 5-episode
number being an optimistic screen (§33's standing lesson) — **done, see §67**; (2) if that holds
up, multi-seed replication of the fine-tune protocol itself before trusting the direction — **done,
see §68**; (3) only if this doesn't hold up at rigor does the wider-roster contingency become the
next thing to try — **did not trigger, see §67/§68**.

## 67. §66's step (1), the 30-episode confirmatory re-eval, is done — the fine-tuning benefit
holds up, more decisively than the 5-episode screen suggested, even though the raw magnitude
shrank (§33's "screens are optimistic" lesson, confirmed again).

Ran `diagnostics/reeval_checkpoint.py --base_dir environments_c1_4_6 --pad_to_true_holdout
--dueling --episodes 30` against both the zero-shot checkpoint
(`results/run_2026_08_29-07_01_27_1193341/global_round_063.pth`) and the fine-tuned round-5
checkpoint (`results/finetune_holdout_fedavg_c146_round063/global_round_005.pth`) from §66, same
real-holdout-traffic evaluator both times.

| | mean_reward | std_reward | min / max | mean Q-gap |
|---|---:|---:|---:|---:|
| zero-shot | -8664.73 | **0.00** | -8664.73 / -8664.73 | 1.59 |
| fine-tuned (round 5) | -1092.10 | 636.86 | -2624.82 / -3.49 | 0.03-0.15 |

Two things worth separating. **(a) The zero-shot checkpoint is a perfect, byte-identical
confident lock-in** — all 30 different SUMO seeds produced the exact same reward, the cleanest
instance of §34's mechanism anywhere in this document (previous confirmed cases still had a few
distinct values, e.g. §49's -9584.47/-9587.6 pair). **(b) The fine-tuned checkpoint is not
locked at all** — high variance, small Q-gaps an order of magnitude below zero-shot's — and even
its *worst* episode (-2624.82) beats zero-shot's constant value by 3.3x, so this isn't a
mean-driven result riding on a few lucky episodes; every single one of the 30 fine-tuned episodes
outperforms every one of zero-shot's. **(c) The magnitude did shrink under more rigorous eval**:
§66's 5-episode screen showed a 20.9x improvement (-413.96 vs -8668.31); the 30-episode number is
a 7.9x improvement (-1092.10 vs -8664.73) — real and large, just not as large as the optimistic
5-episode draw implied, exactly the pattern §33 first flagged and has recurred several times since
(§62→63, etc.).

Still far from baseline-competitive in absolute terms: `fixed_time`=-2.73, `max_pressure`=-0.34
(§66's numbers, same evaluator) — fine-tuned is ~400x off `fixed_time` and ~3200x off
`max_pressure`. Same reading as §59-61 and §66: real, worth having, not close to sufficient alone.

**Decision:** per the CLAUDE.md-recorded rule ("if the fine-tune-on-holdout test does NOT show a
real benefit, the next lever is training with wider roster"), this is a confirmed real benefit,
not a null — **the wider-roster-retrain contingency does NOT trigger.** Per §66's own next-steps
ordering, moving to step (2): multi-seed replication of the fine-tune protocol itself, launched
same session (seeds 3/7/11/17 against the same zero-shot starting checkpoint, `--rounds 5` each,
same settings as §66) — see the next section once that lands for whether the effect survives
multiple starting seeds or was itself a lucky single-seed draw (this document's standing
single-seed caution, §11→12/30→31/46→47/62→63, applies here too — this is real evidence of a
benefit, not yet proof it replicates).

## 68. Multi-seed replication (seeds 3/7/11/17) confirms the fine-tune-on-holdout benefit
decisively — this is now the strongest, most cleanly confirmed positive result anywhere in this
document, and does not fit the "single-seed story doesn't replicate" pattern seen everywhere else
(§11→12/30→31/46→47/62→63).

Same protocol as §66 (`--rounds 5 --local_episodes 2 --n_variants 5 --eval_episodes 5`, same
`fedavg` round_063 starting checkpoint, same 5 fixed randomized-traffic route files reused across
seeds), varying only `--seed`:

| seed | round 1 | round 2 | round 3 | round 4 | round 5 |
|---:|---:|---:|---:|---:|---:|
| 3 | -1321.95 | -636.43 | -696.86 | -1691.45 | -413.96 |
| 7 | -1674.97 | -1704.26 | -412.01 | -2203.05 | -707.31 |
| 11 | -1017.97 | -6.66 | -240.73 | -483.74 | -1458.29 |
| 17 | -2252.63 | -14.74 | -230.88 | -116.87 | -1240.25 |

Against zero-shot's §67-confirmed 30-episode value of -8664.73: best-of-5-rounds per seed gives
mean=-211.84, SE=116.14, **|diff|/SE=72.78**; the more conservative, selection-bias-free
round-5-only (final round, not cherry-picked) gives mean=-954.95, SE=239.57, **|diff|/SE=32.18**.
Both clear this project's ≥2 bar by more than an order of magnitude — nothing else in this document
comes close to this large a margin. Every single round of every one of the 4 seeds tested is at
least 3.9x better than zero-shot; several individual data points are 10-1000x better. This is not
a fragile, lucky-seed effect.

**Still true and worth restating: this closes a huge fraction of the gap without coming close to
being baseline-competitive.** Even the single best result across all 20 (4 seeds × 5 rounds) data
points (-6.66, seed 11 round 2) is still ~20x worse than `max_pressure` (-0.34, §66's number, same
evaluator) and ~2.4x worse than `fixed_time` (-2.73). The honest framing for a paper: a cheap,
robust, mechanistically-grounded (§67's lock-in-escape mechanism) test-time intervention that
reliably recovers most of the reward on offer without reaching rule-based-controller parity — a
real, well-replicated, citable finding on its own, not a "solved it" result.

**Decision, matching §67's: the wider-roster-retrain contingency (queued in CLAUDE.md 2026-08-31)
continues to NOT trigger** — this lever is confirmed working, not stalled.

**Next steps in progress (agreed 2026-09-01, prioritized as "dose-response on a fixed baseline
first, vary-the-starting-checkpoint second"):** characterizing the fine-tune-duration dose-response
curve — same `fedavg` round_063 checkpoint, same seed=3, `--rounds 7` in one run (the script evals
every round via `eval_every=1`, so one run captures rounds 1-7 directly, no need for 5 separate
short runs) — launched, in progress as of this writing
(`results/finetune_holdout_fedavg_c146_round063_dose7`). Deferred until that curve is in: (a)
whether the benefit holds starting from a *different* federated checkpoint (earlier/weaker, or the
`clustered_fedavg` run instead of `fedavg`); (b) whether regenerating the random traffic each round
(rather than the same 5 fixed variants held constant for the whole burst) helps further — only
worth the engineering cost (`ParallelFederatedServer` workers don't currently support swapping a
route file mid-run without a full env rebuild) if the dose-response curve shows signs of
overfitting to the fixed 5 patterns (a plateau or reversal at higher round counts would be that
signal).

## 69. Fine-tune-duration "dose-response" curve (4 seeds, 7 rounds each) — the benefit keeps
growing past round 5, two seeds get remarkably close to baseline at their peak, but the
underlying instability is not fixed by fine-tuning either: round 7 is not reliably better than
round 6.

**Important methodological caveat, discovered while checking this run — not a clean single-variable
dose-response curve.** `compute_eps_decay` sizes the exploration schedule from `--rounds`, so a
`--rounds 7` run spreads epsilon decay more slowly than a `--rounds 5` run — round 5 of a 7-round
run experienced a *different* (slower) exploration schedule than round 5 of the standalone 5-round
runs in §66/§68. Confirmed directly: this run's seed-3 rounds 1-5 (-2529.97, -164.57, -1176.60,
-411.08, -2185.30) do NOT match §66's original seed-3 5-round run (-1321.95, -636.43, -696.86,
-1691.45, -413.96) at all, despite identical checkpoint, seed, and route files. **So "round N of a
7-round run" and "round N of a 5-round run" are not directly comparable — this conflates fine-tune
duration with exploration pace.** The 4 seeds tested here (3/7/11/17) ARE directly comparable to
each other (same `--rounds 7`, same schedule), so within-this-batch comparisons are valid; a truly
clean duration-only dose-response curve would need separate standalone runs per duration, not yet
done.

Same protocol as §66/§68 otherwise (`--local_episodes 2 --n_variants 5 --eval_episodes 5`, same
`fedavg` round_063 checkpoint, same 5 fixed route files), `--rounds 7`, seeds 3/7/11/17:

| round | seed 3 | seed 7 | seed 11 | seed 17 | mean across seeds |
|---:|---:|---:|---:|---:|---:|
| 1 | -2529.97 | -1236.94 | -2243.80 | -2339.19 | -2087.47 |
| 2 | -164.57 | -283.56 | -212.29 | -494.91 | -288.83 |
| 3 | -1176.60 | -1495.40 | -705.43 | -292.28 | -917.43 |
| 4 | -411.08 | -940.85 | -53.58 | -1522.17 | -731.92 |
| 5 | -2185.30 | -5.61 | -244.66 | -149.08 | -646.16 |
| 6 | -223.08 | **-1.24** | -213.18 | -150.32 | **-146.96** |
| 7 | -235.82 | -1335.14 | **-4.31** | -534.76 | -527.51 |

**Round 6 has the best mean reward of any round in this batch (-146.96)** — noticeably better than
§66/§68's round-5 numbers, so the effect had clearly not plateaued by round 5. Against zero-shot's
30-episode-confirmed -8664.73: best-of-7-rounds per seed gives mean=-79.80, SE=44.59,
**|diff|/SE=192.54** (even more extreme than §68's 5-round version); the unbiased round-7-only
gives mean=-527.51, SE=290.28, **|diff|/SE=28.03**. Both far past this project's ≥2 bar.

**Two seeds reached genuinely near-baseline performance at their peak** — seed 7 round 6: -1.24
(vs. `max_pressure`'s -0.34, `fixed_time`'s -2.73 — within striking distance of both); seed 11
round 7: -4.31 (~12.7x off `max_pressure`, ~1.6x off `fixed_time`). These are by far the closest
any result anywhere in this document has come to rule-based-controller parity.

**But this is not stable, and that's the real finding here.** Seed 7 relapsed sharply the very
next round after its near-baseline peak: -1.24 (round 6) -> -1335.14 (round 7), a >1000x reversal
in one round. This is the exact confident-lock-in volatility signature documented throughout this
project (§32-34, §51-52's "a good policy is reachable by ordinary gradient steps, it just isn't
retained") — now observed *within* the fine-tune-on-holdout lever too, not just in from-scratch
federated training. Round-to-round mean also dips after round 6 (mean -527.51 at round 7 vs.
-146.96 at round 6), driven mostly by seed 7's relapse, not a clean monotonic curve.

**Framing for the paper, updated:** fine-tuning on synthetic random traffic is a real, robust,
multi-seed-confirmed way to substantially close (and at its peak, nearly eliminate) the
cross-topology gap — but it does not fix the underlying training instability characterized
elsewhere in this document. The honest story is "this lever can reach near-baseline performance,
demonstrating the gap is not fundamentally unclosable, but doesn't reliably stay there any better
than plain federated training does" — a nuance that strengthens rather than weakens the mechanism
narrative (§32-57), since it shows the instability is a property of the training dynamics, not
something fine-tuning-as-currently-implemented incidentally works around.

**Next steps, updated 2026-09-01:** (1) a clean single-duration-variable dose-response curve
(separate standalone runs at each duration, not one long run) if the exact shape still matters for
the paper; (2) checkpoint selection strategy — since round 7 isn't reliably better than round 6,
"best-round-so-far" (as this document already does for federated training) is probably the right
way to pick a fine-tuned checkpoint to report/deploy, not "final round"; (3) still deferred:
whether the benefit holds from a different starting checkpoint, and whether per-round traffic
regeneration helps (no new evidence of overfitting to the fixed 5 patterns from this batch, so
still not clearly worth the engineering cost yet).

## 70. THE CONTROL §66-69 WAS MISSING: a randomly-initialized network, fine-tuned identically,
matches or beats the federated-pretrained one. On this evidence the federated pre-training is
not what produces §66-69's headline improvement — read this section before citing §66-69 as
evidence for the foundation-model premise.

**Why this section exists.** §66-69 established a large, multi-seed-confirmed improvement from
fine-tuning a federated checkpoint on synthetic holdout-topology traffic. But none of those runs
could distinguish three explanations: **(a)** the federated pre-training provides a useful
starting point that fine-tuning adapts (the premise this whole project rests on); **(b)** the
starting point is irrelevant and "fine-tuning" is simply "training on the target city", which
would work as well from scratch; **(c)** a mix — pretraining helps reach a good result
faster/more reliably without being necessary. Added `--random_init` to
`diagnostics/finetune_on_holdout.py` to discriminate: it infers the architecture from the
checkpoint (so the control is *exactly* matched — same own_dim/neighbor_dim/action_dim/dueling/
head_fix) then discards the weights and starts from a fresh random init. Pretrained-vs-random
weights is the single variable.

**Protocol (both arms identical apart from starting weights):** starting checkpoint
`results/run_2026_08_29-07_01_27_1193341/global_round_039.pth` (the `fedavg` baseline's *best*
training round, not its final one — see the zero-shot table below for why that matters),
`--rounds 8 --phase1_rounds 4 --phase2_lr 1e-5 --local_episodes 2 --n_variants 5
--eval_episodes 5 --seed 3`, same 5 fixed randomized route files.

**Zero-shot, before any fine-tuning (5-episode eval, same evaluator):**

| starting point | zero-shot reward |
|---|---:|
| federated round 39 (best training round) | -4294.87 |
| **random init (untrained)** | **-7689.59** |
| federated round 63 (final training round) | -8668.31 |

**Two findings here, and the second is independently important.** First, pre-training *does* give
a real head start over random weights (-4294.87 vs -7689.59, ~1.8x) — so hypothesis (b) is not
true at the starting line. Second, and more damaging: **the federated model's FINAL checkpoint
(round 63) is worse on the true holdout than a randomly initialized network.** Training from round
39 to round 63 drove the model below random on the target city. That is a concrete, quantified
instance of the confident-lock-in thread (§32-34/§51-57), and it converges with §67's finding that
round 63's 30-episode eval had *exactly zero* variance across SUMO seeds — the model is not merely
suboptimal, it is confidently locked into behaviour worse than noise. It also independently
justifies the "best-round, not final-round" checkpoint-selection rule §69 arrived at.

**Fine-tuning curves (5-episode per-round screen):**

| arm | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | best | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained (r39) | -1016.68 | -862.91 | -426.58 | -2372.52 | -526.58 | **-275.38** | -1156.64 | -1338.06 | -275.38 (r6) | -1338.06 |
| random init | -876.39 | -282.84 | -581.94 | **-0.41** | -1060.33 | -1573.63 | -2661.61 | -1726.84 | -0.41 (r4) | -1726.84 |

**Matched 30-episode confirmatory re-eval of each arm's best checkpoint** (the 5-episode screen is
known to run optimistic — §33, re-confirmed here: the control's headline -0.41 became -406.85):

| arm | mean_reward | std | min / max |
|---|---:|---:|---:|
| pretrained r39 → fine-tune round 6 | -693.84 | 531.12 | -1853.50 / -1.25 |
| **random init → fine-tune round 4** | **-406.85** | 501.56 | -1375.79 / **-0.03** |

**The random-init control is 1.71x better than the federated-pretrained arm** at matched
30-episode rigor. Both arms have enormous within-checkpoint variance (std ~500 on means of
-400/-700; both have single episodes reaching essentially perfect play, -0.03 and -1.25, alongside
episodes near -1400/-1850) — these are not stable policies, they are policies that *sometimes*
find a near-optimal region.

**On statistical rigor, stated precisely because it matters for how far this can be pushed.**
Computing |diff|/SE from the 30 episodes gives 2.15, nominally past this project's ≥2 bar — **but
that is the wrong statistic for this claim and should not be quoted as if it settled the
question.** Those 30 episodes measure episode-to-episode (SUMO-seed) variance *within one fixed
checkpoint*; the claim "pre-training does/doesn't help" needs variance across *training* seeds,
which §69 showed is enormous (one seed swung -1.24 → -1335.14 in a single round). With one training
seed per arm, the dominant uncertainty is unmeasured. **Correct reading: directionally, random init
matches or beats federated pre-training at this budget; NOT established at this project's own
standard of rigor.** The decisive experiment — multi-seed replication of both arms — is not yet
run and is the single highest-value thing left in this thread.

**Also relevant: the two-phase LR schedule (this section's other change) did not work.** It was
added on the hypothesis that §69's large round-to-round swings meant the LR was too high to hold a
good region once found. Rounds 5-8 ran at 1e-5 (5x lower). Both arms nonetheless *degraded* through
phase 2 — the control monotonically (-1060 → -1573 → -2662 → -1727), the pretrained arm still
swinging ~4x — and both ended 5-6x worse than their own best round. **Two independent arms failing
to stabilize under a 5x LR cut is reasonable evidence the volatility is not a step-size problem**,
pushing the explanation back toward the confident-lock-in mechanism rather than optimizer
settings.

**What this does and does not overturn.** It does NOT retract §66-69: fine-tuning on
holdout-topology traffic really does produce a large improvement over the zero-shot federated
model, replicated across seeds. What it changes is the *interpretation* — that improvement is
evidence for "training on the target city's topology helps", not for "federated pre-training
transfers usefully." Given this project's premise is a shared foundation policy across topologies,
that distinction is central rather than incidental, and it is the honest headline: **at this
budget, on this holdout, the federated pre-training is not measurably the source of the benefit.**
For the paper this is a legitimate and publishable negative control — arguably a more interesting
contribution than the positive result it qualifies, and squarely in the same category as RESCO's
own "published methods underperform simple baselines" finding.

**Next steps, in priority order:** (1) **multi-seed replication of both arms** (3-5 seeds each,
same matched protocol) — the decisive experiment, converts this from directional to established
or refutes it; (2) if the control holds, re-examine whether *any* claim in this document depends
on federated pre-training contributing beyond a better starting point; (3) a fair-budget caveat
worth testing eventually — this compares arms at a *fixed* 8-round fine-tune budget, which is the
decision-relevant comparison given the lever's selling point is cheapness, but does not establish
what happens if the random-init arm is given substantially more training.

## 71. The roster-diversity hypothesis tested directly and comes back NULL: 14 topologically
diverse cities perform no better on the true holdout than 3. Together with §70 this closes off
"the training data isn't diverse enough" as the explanation.

**The hypothesis.** §70's random-init control found the federated pre-training wasn't transferring
anything useful. The most obvious benign explanation: every roster this project has trained on is
small and topologically narrow (2-7 cities, mostly uniform intersection shapes), so the model never
*practices* generalizing across genuinely different topologies during training — it only meets a
different one at holdout-eval time. If that were the constraint, a much wider and more varied
roster should help. This section tests exactly that.

**New tooling (`diagnostics/generate_grid_cities.py`, committed).** Generates synthetic cities by
`netgenerate`-ing a perfect NxM lattice, deleting a `--drop_fraction` of interior traffic-light
nodes plus every edge touching them, and rebuilding with `netconvert`
(`--keep-edges.components 1`, so a deletion that splits the lattice still yields one routable net).
Junctions around each hole become 3-way T-junctions and dead-ends, so one city contains a genuine
mix of 2-, 3- and 4-way intersections rather than netgenerate's uniform 4-way monoculture.

**Two measured facts encoded as defaults in that script, both of which would have silently
invalidated this experiment:** (1) SUMO's default `--tls.layout opposites` collapses *every*
junction to 2 actions regardless of its shape (measured: a 4x4 grid gives `{2: 16}`) — i.e. the
default would have produced an action-space monoculture even though the *geometry* was varied;
`incoming` makes phase count follow junction degree (3-way→3 actions, 4-way→4). (2) At
`--lanes 1`, netconvert declines to signalize the junctions at all (zero `tlLogic` elements).

**Roster built:** 8 generated cities (3x3/4x4/5x5/6x6, drop 0-30%), 116 intersections total,
action-count spread `{2-action: 8, 3-action: 42, 4-action: 66}`, with genuine *within-city* mixing
(e.g. `grid_4x4_drop20` = 1/5/7 intersections at 2/3/4 actions). `environments_wide/` combines
these with the 6 real RESCO training cities = **14 training cities**, roster action_dim spread
2/3/4/5 (holdout 8), vs. the 3-city comparison roster's 26 intersections. Verified before
launching, given §25's silent-fallback incident: `resolve_city_configs_and_dims` agrees on
(117, 3, 8) across all 14, and `make_holdout_evaluator` resolves to `city_5_holdout` with
`is_true_holdout=True` — a real holdout, not a compatibility fallback.

**Result.** Identical protocol/seed/budget to the 3-city `fedavg` baseline of §65
(`--dueling --n_step 3 --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5 --pad_to_true_holdout --seed 3
--eval_every 1 --eval_episodes 5`). Run stopped by user request at round 58/63; all numbers below
use the **matched** rounds-21-58 window on both sides, n=38 each:

| | mean(21-58) | std | best round ever |
|---|---:|---:|---:|
| wide (14 cities) | -6304.85 | 1311.43 | **-3451.49** (round 41) |
| narrow (3 cities) | -6110.73 | 1488.81 | -4294.87 |

**Mean: a clean null — |diff|/SE = 0.60**, far below this project's ≥2 bar, with the narrow roster
nominally *ahead*. **Best-round: the wide roster wins** (-3451.49 vs -4294.87, ~20% better) — the
best true-holdout round anywhere in this document.

**How to read the split.** The best-round edge is a single round out of 58 in a run whose rewards
swing from -3451 to -9588. §51/§52 established precisely this pattern (good policies are
*reachable* by ordinary gradient steps but not *retained*), and §69 saw the same thing inside the
fine-tune lever. A better single peak is therefore weak evidence of a better basin; the mean is the
more robust statistic and it says nothing changed. Worth noting the wide run's best came at round
41 and its second-best stretch was rounds 50-52, i.e. late — so "wide roster needs more rounds" is
not fully excluded by stopping at 58, but 38 matched rounds showing no mean separation makes a
large late effect unlikely.

**Conclusion, and it is a substantive one.** Going from 3 → 14 cities (26 → 116 intersections,
uniform → genuinely mixed 2/3/4-way, action_dim spread 2-5) produced **no mean improvement in
cross-topology generalization**. A 30-city roster behaving differently when 14 behaves like 3 is
not a good bet. **Combined with §70's random-init control, two independent lines of evidence now
point the same way: the binding constraint is not the training data — not its quantity, not its
topological diversity — but the algorithm's failure to retain and transfer what it learns.**
That is a much sharper statement of the project's central negative result than §43-§61's
"generalization gap" framing, and it is the right one to build the paper's argument around.

**Caveats:** single seed (§69 showed seed-to-seed variation here is enormous, so this is
directional, not confirmatory — though note it would take a *large* seed effect to move a
|diff|/SE of 0.60 past 2); stopped at 58/63 rounds by user request, with matched windows used
throughout so no comparison is biased by the early stop; the generated cities are synthetic grids,
so this tests *topological* diversity, not real-world demand-pattern diversity.

**Data:** `results/run_2026_09_01-22_46_01_1690128` (wide, 58 rounds + per-round checkpoints),
`results/run_2026_08_29-07_01_27_1193341` (narrow 3-city baseline, full 63). Rosters:
`environments_wide/`, `environments_grid/`; nets under `sumo_rl/nets/generated/`.

## 72. Three candidate mechanisms for the §70/§71 transfer/retention problem, launched as small
    pilots in parallel, pre-registered here before results are known

**2026-09-04.** §70/§71 converged on a sharper diagnosis than "generalization gap": the algorithm
isn't retaining/transferring what it learns (random-init beat federated-pretrained at fine-tune
time, §70; 3→14 training cities didn't help, §71). Per the user's request, three concrete
mechanisms that could plausibly fix *that specific* problem are being piloted small/cheap, one
seed each, before committing to a bigger run on whichever survives. **Pre-registering the designs
and decision rule here, before any run has finished**, per this project's own standing discipline
(see the |diff|/SE ≥ 2 bar used everywhere else in this doc) — results get added once in.

**A. Does training with `--q_entropy_weight` (already implemented, §54-56) produce a checkpoint
that transfers *better* under fine-tuning than plain FedAvg, even though §55/§56 already showed it
doesn't improve raw federated reward?** Not previously tested — §54-56 only ever measured the
q_entropy-trained checkpoint's own federated-eval reward, never ran it through the §66-70
fine-tune-on-holdout pipeline. New training run: `environments_c1_4_6`, seed 3,
`--dueling --n_step 3 --pad_to_true_holdout --q_entropy_weight 0.05` (the value that survived
§55/§56), `--rounds 20` (matches an existing checkpoint round number so the comparison arm below
needs no rerun) — otherwise identical to the §70/§71 baseline run's config
(`results/run_2026_08_29-07_01_27_1193341`). Then fine-tune its round-20 checkpoint with the exact
§66-70 protocol (`diagnostics/finetune_on_holdout.py --rounds 8 --phase1_rounds 4 --phase2_lr 1e-5
--local_episodes 2 --n_variants 5 --seed 3`). Three-way comparison at matched round 20: this new
qew-pretrained arm vs. plain-fedavg-round-20 fine-tuned (new fine-tune run, reusing the *existing*
`global_round_020.pth` from the baseline run — no retrain needed) vs. random-init fine-tuned
(reusing §70's existing result directly — architecture is round-independent, so that arm doesn't
need to be re-run at round 20 specifically).

**B. Shrink-and-perturb (Ash & Adams 2020) before fine-tuning.** New `--shrink_alpha`/
`--perturb_std` flags added to `diagnostics/finetune_on_holdout.py` this session: right before the
fine-tune burst starts (NOT before the zero-shot eval, which stays comparable to every past
number), each pretrained weight tensor `W` becomes `shrink_alpha*W + perturb_std*std(W)*noise`.
Tests §70's actual anomaly directly — if pretrained-then-finetune loses to random-init-then-
finetune because the pretrained checkpoint has lost plasticity (locked into an overconfident state,
§32-34), partially shrinking+perturbing it toward a fresher init before fine-tuning should recover
some of random-init's advantage while keeping the pretrained features, landing somewhere between
the plain-pretrained arm (-693.84) and the random-init arm (-406.85) rather than at either extreme
(or beyond either, which would itself be informative). Pilot: `global_round_039.pth`
(the exact checkpoint §70 used), `shrink_alpha=0.5 perturb_std=0.1`, otherwise identical protocol
to §70 (`--rounds 8 --phase1_rounds 4 --phase2_lr 1e-5 --local_episodes 2 --n_variants 5 --seed 3`).

**C. `--fedavg_blend` (Reptile-style damped server update) — implemented since at least
2026-08-15, cited in the CLI help, never once tested anywhere in this document.**
`federated/parallel_server.py`/`server.py`: `agg_state = blend*agg_state + (1-blend)*prev_global`
each round, `blend=1.0` (default) an exact no-op. This is mechanically the "meta-learning" lever
discussed with the user — a damped step toward each round's aggregate optimizes the global model to
be a good *starting point for local adaptation* rather than a fixed policy good on average, which
matters more now that fine-tuning (§66-70) is the thing that actually gets deployed downstream.
Pilot: `environments_c1_4`, seed 3, `--dueling --n_step 3 --pad_to_true_holdout --rounds 20
--fedavg_blend 0.5`, otherwise identical to the existing seed-3 baseline
(`results/run_2026_08_18-19_46_23_818099`, §43/§46: best=-2855.95, mean=-6624.90) — no rerun needed
for the baseline side of this comparison.

**Decision rule (agreed pattern, same bar used throughout this doc):** each pilot is single-seed,
read as a lead not a result. Whichever of A/B/C shows a real lead on this cheap pilot goes to
multi-seed validation before any larger training commitment; a mechanism that doesn't clear even a
single-seed lead here is deprioritized without further seeds, consistent with how this document has
handled every other candidate lever (§37, §41, §54, §65).

**Status: all three finished 2026-09-04/05 — B and C are clean single-seed misses, A's real
comparison (the fine-tune step, not the raw pretrain reward) came back close to a wash.**

- **B (shrink-perturb, `shrink_alpha=0.5 perturb_std=0.1`):** best round -1985.37, final round
  -4922.88 — both worse than the zero-shot baseline it started from (-4294.87), and far worse than
  either of §70's existing arms (pretrained -693.84, random-init -406.85). Clean miss, deprioritized
  per the decision rule above.
- **C (`--fedavg_blend 0.5`, environments_c1_4, 20 rounds):** best -5414.22, mean -7507.08, vs. the
  existing seed-3 baseline's best -2855.95 / mean -6624.90 — worse on both measures, single seed.
  Clean miss, deprioritized.
- **A (q_entropy=0.05 pretrain on environments_c1_4_6, round-20 checkpoint, then the standard
  §66-70 fine-tune protocol):** the three-way comparison at matched round 20 — plain-FedAvg
  pretrained→fine-tuned: best -0.41, final -0.49; q_entropy=0.05 pretrained→fine-tuned: best -2.88,
  final -114.44; random-init→fine-tuned (§70, reused): -406.85. **Plain FedAvg remains the best
  pretraining choice at this checkpoint round, and both real-pretraining arms clearly beat
  random-init here** — note this does NOT reproduce §70's finding at round 39, where random-init
  won; round 20 vs. round 39 is itself a candidate explanation (round 39 may already be past the
  point where more federated training starts hurting retention, consistent with §70's own
  round-63-worse-than-random finding), not yet disentangled.

**Net effect on the standing question:** none of A/B/C reversed the §70/§71 diagnosis. Per the
agreed decision rule, this closes out B and C without further seeds; A's own next step (why does
round 20 disagree with round 39 on pretrained-vs-random) is a new, narrower open question, not
pursued further as of this writeup in favor of §73 below.

## 73. Algorithm swap (DQN → PPO / Munchausen-DQN): three short pilots, none beat DQN yet

**2026-09-04/05.** Direct response to the user's question "can we keep FedAvg but replace DQN with
something that works better for this task" (prompted by §70's retention-failure diagnosis and the
observation that DQN's `argmax(Q)` policy can collapse into the confident-lock-in failure mode,
§32-34/§51-57, in a way a stochastic policy structurally can't). FedAvg itself is untouched in all
of this — only the local agent/network changes.

**Implemented:** `agents/ppo.py` (on-policy actor-critic, clipped surrogate + GAE + entropy bonus,
reuses the existing attention trunk via new `policy_head`/`ac_value_head` outputs on
`NeighborAttentionQNetwork`) and `agents/munchausen_dqn.py` (off-policy — same replay buffer/
sample-efficiency as DQN, unlike PPO — Boltzmann policy over Q with an entropy-regularized soft-
Bellman target, Vieillard et al. 2020). Both match `DQNAgent`'s external interface exactly, so
`federated/client.py`/`FederatedServer` needed zero changes; a new `--algo {dqn,ppo,munchausen}`
flag selects the agent in both the sequential and `--parallel` paths. Two real pre-existing bugs
found and fixed along the way: `federated/server.py` and `federated/parallel_server.py` both
hardcoded `global_model.q.state_dict()`/`.q.parameters()` (DQNAgent-specific attribute access) for
checkpoint-saving and weight-norm logging, which would crash for any non-DQN agent — switched to
the already-existing agent-agnostic `state_dict()` interface. Also added `--d_model`/`--n_heads`
(previously hardcoded at 128/4) and `--munchausen_temp`/`--munchausen_alpha` as CLI flags.

**Results so far, all `environments_c1_4_6`, seed 3, true holdout:**

| config | rounds | best | mean |
|---|---:|---:|---:|
| DQN + q_entropy=0.05 (baseline, §72 pilot A) | 5 (of 20) | **-5933.60** | **-8016.66** |
| DQN + q_entropy=0.05 (same run, full budget) | 20 | -3288.73 | -6093.66 |
| PPO (default hyperparams) | 20 | -7273.83 | -9382.94 |
| Munchausen-DQN (default: temp=0.03, alpha=0.9, d_model=128) | 5 | -7198.59 | -8539.36 |
| Munchausen-DQN (d_model=512, n_heads=8 -- "big model" check) | 5 | -8417.92 | -9611.91 (got *worse* every round after round 1) |

**Reading:** neither alternative has beaten plain DQN yet, at either matched budget tested.
PPO's shortfall has a known, uncontrolled confound — it's on-policy (discards each round's
rollout after one use) while DQN reuses a persistent replay buffer, so at this project's standard
tiny `--local_episodes 2`, DQN gets far more gradient signal per round for reasons that have
nothing to do with which algorithm suits the task better; not yet tested with a larger episode
budget that would remove this confound. Munchausen-DQN has no such confound (same off-policy
replay-buffer training as DQN) and still lost at matched 5-round budget with default
hyperparameters — a real, if preliminary, negative data point. The "bigger model" variant is the
more interesting finding: capacity did not help and the run actively degraded round-over-round,
suggesting more parameters without more data/rounds to fit them may hurt at this tiny budget
rather than being neutral.

**Not yet a verdict on either algorithm** — only default/one-off hyperparameters have been tried
for Munchausen, and PPO hasn't been tested with a fairer (larger) episode budget. Per the user's
explicit steer (2026-09-05, "short training with different model and types of models" — prefer
breadth of many short trials over one long multi-seed replication), the immediate next step is a
broad overnight sweep of short (5-round) trials across Munchausen hyperparameters (temp, alpha,
dueling, n_step, d_model in {64,256}), a plain-DQN capacity check (d_model=256), the untested
`--dueling --n_step 3 --q_entropy_weight 0.05` combo, and one PPO run with `--local_episodes 8` to
directly test the on-policy sample-efficiency confound — see the batch launched immediately after
this section for the exact job list and results as they land. The higher-value but slower §70
multi-seed replication (random-init vs. pretrained across more seeds) remains queued but
deprioritized for tonight per that same steer.

**Overnight sweep, batch 1 (Munchausen temperature/alpha, 5 rounds each, environments_c1_4_6,
seed 3) — first real lead found:**

| config | best | mean |
|---|---:|---:|
| DQN+qew baseline (rounds 1-5, reused from above) | **-5933.60** | -8016.66 |
| Munchausen temp=0.1, alpha=0.9 | -7932.27 | -9127.86 |
| **Munchausen temp=0.01, alpha=0.9** | -6382.16 | **-7490.44** |
| Munchausen temp=0.03, alpha=0.5 | -8016.12 | -9097.92 |

`temp=0.01` (crisper/less-soft target than the paper-default 0.03 tried earlier) is the first
config in this entire algorithm-swap effort (this section, all of PPO and default-hyperparameter
Munchausen) to beat the DQN baseline on any measure — it wins on mean reward (-7490.44 vs
-8016.66) though still loses on best-round (-6382.16 vs -5933.60). Single trial, no seed
replication yet — read as a lead, not a result, same standing caution as everywhere else in this
document. Also observed directly in this batch: both temp=0.1 and alpha=0.5 showed the
confident-lock-in signature mid-run (std collapsing to 1.8-5.7 across rounds 1-3) then escaped it
by round 3-4 (std back up to 150-175) -- the same lock-in/escape pattern already documented at
length in §32-34/§51-53, now also reproduced under Munchausen-DQN, not just plain DQN.

**Batch 2 launched (exploiting the temp=0.01 lead):** temp=0.01+dueling, temp=0.01+n_step=3,
temp=0.005 (pushing the temperature axis further). Results to follow.

## Open questions / next steps

1. ~~**Run-to-run non-determinism (the big open one).**~~ **Resolved — see §5, confirmed with a
   real multi-seed run in §6.** Parallel workers were never seeded; fixed and verified
   deterministic 2026-08-04, then verified on a real 3-seed/20-round run: per-city training loss
   is now consistently well-behaved across seeds (§6). Runs from before the fix (everything in
   §3, and the `no_federation_vs_federated_comparison.md` A/B pair) should still be read as "one
   sample from an unseeded process," not as reproducible results.
1b. **New from §6: holdout-reward volatility persists even with the fix**, but the 2-city cheap
   roster used to test this has a scale mismatch (train on 3+7-intersection cities, eval on a
   16-intersection holdout) that's a plausible full explanation on its own. Before concluding
   there's a second, distinct instability source: rerun the same multi-seed check with a roster
   whose training cities are closer in scale to the holdout (e.g. swap in city_3 or city_7
   alongside city_4/city_6), or evaluate against a smaller holdout that's actually representative
   of what a 2-small-city model was trained for.
2. ~~**Validate the city_1 swap.**~~ **Done properly — see §7.** The `no_federation_vs_federated_comparison.md`
   run predated the seeding fix; §7's 3-seed/20-round run with the fix in place is the real
   answer: city_1's persistent-high-loss failure mode is gone, confirmed across 3 seeds.
2b. **Round-20-style reward regression — mechanism confirmed (weight deltas), but "clusters at
   16-18" was a small-sample artifact.** §12 found seed5's best round is round 5, not 16-19 —
   fix_on can swing to a near-optimal *or* a badly-degraded policy at any point in training, not
   specifically at "the end." The ~2.2-2.3x-faster-movement-on-specialized-rows mechanism (§11,
   still valid) explains the *volatility*, not a specific "always regresses near round 20" claim,
   which should be dropped.
3. ~~**Phase 1 masked-head ablation.**~~ **Done with 5 seeds — see §11 (3 seeds) then §12
   (correction with 2 more). Final result: ambiguous on mean reward** (not statistically
   distinguishable, |diff|/SE≈0.68), **with a real, mechanistically-understood volatility
   tradeoff**: masked-head aggregation usually reaches a better peak (better median best-round,
   4/5 seeds) but has a real chance of an outright failure fix_off never produces (seed4). Not a
   basis for "fix wins" in a paper claim — report as a genuine conditional/negative result per the
   plan's own guidance for this decision-gate outcome, rather than scaling to Phase 2 or running
   further seeds on this same question. `fix_on` seed4's specific failure (reward degrades over 20
   rounds despite healthy-looking local loss) is a new, separate open item — see below.
3b. ~~**`fix_on` seed4's failure mode.**~~ **Minimally reproduced and localized to federation
   itself — see §13.** `city_1` alone (same seed) is stable and near-optimal; `city_1`+`city_4` (2
   cities, the minimum possible federation) reproduces severe instability immediately, more
   violently than the original 3-city case. Confirms the cause is aggregation/client-drift, not
   something specific to `city_1`'s own training or to the particular 3-city combination. `city_1`
   +`city_4` is now the standing cheap test bed for any fix aimed at this (e.g. FedProx below).
4. ~~**FedProx proximal term.**~~ **Tested 2026-08-06 — negative result, see [§14](#14-fedprox-swept-across-mu-and-a-3rd-city-no-stabilizing-effect).**
   Swept mu in {0, 0.01, 0.03, 0.1} plus a 3-city generalization check; no stabilizing effect at
   any strength, mu=0.1 measurably worse. Not the fix.
   ~~**Dueling network head.**~~ **Tested 2026-08-06/07 — clean, generalizing win, see
   [§15](#15-dueling-network-head-the-first-intervention-that-actually-helps).** ~26-35% better
   mean reward, 5-8x better best round, on both the 2-city and 3-city rosters. The first
   intervention this session that's a structural win rather than a wash or tradeoff.
   ~~**Server-side momentum.**~~ **Tested 2026-08-06/07 — modest, mixed benefit, see
   [§16](#16-server-side-momentum-fedavgm-style-modest-mixed-benefit).** ~9-10% better mean on
   both rosters, but damps peaks along with crashes (worse best-round on 2-city). Weaker than
   dueling.
   ~~**Dueling + momentum combined.**~~ **Tested 2026-08-07/09 — net-negative, see
   [§18](#18-dueling--momentum-combined-net-negative-interaction-confirmed-on-both-rosters).**
   Dueling alone always beats dueling+momentum on mean and best round, on both rosters. A hard
   CLI check now blocks this combination (`experiments/federated_training.py`).
   ~~**n-step returns, pseudo-gradient clipping, EMA eval snapshot.**~~ **All tested 2026-08-09/10,
   alone and combined with dueling — see
   [§19](#19-three-more-new_ideeas-alone-and-combined-with-dueling-n-step-is-the-new-headline-result).**
   **`dueling+n_step` is the new best-known config** — a genuine synergy, #1 on mean and best
   round on both rosters, superseding "use dueling alone." gradclip and ema_eval (alone or with
   dueling) are unconvincing — gradclip's fixed threshold doesn't transfer across roster sizes,
   ema_eval only compresses reported variance without improving training (by design — it's
   eval-only). **Current recommendation: `--dueling --n_step 3`.**
5. ~~**`baseline_max_pressure`'s implausible numbers**~~ **Investigated — see
   [§24](#24-major-bug-found-while-starting-phase-2-the-fixed_time-rule-based-baseline-never-actually-ran-fixed-time-control--every-fixed_time-number-in-this-projects-historyis-invalid).**
   `max_pressure` itself was fine; `fixed_time` was the actually-broken one (three-layer
   attribute-forwarding + missing multi-agent guard bug), and fixing it revealed the DQN currently
   loses to both rule-based baselines on 7-city holdout. Bigger finding than the original flag.
6. ~~**Bring 7-city Phase 1 ablation from 2 to 5 seeds.**~~ **Done — see
   [§23](#23-7-city-phase-1-ablation-brought-to-5-seeds-the-fixs-mean-reward-benefit-doesnt-just-shrink-with-roster-size-it-vanishes--but-the-best-round-win-survives-at-every-scale-tested).**
   Mean-reward benefit is fully gone at 7-city (|diff|/SE=0.23); best-round win persists (gap 1722,
   fix-on still clearly better). All three roster sizes now have 5 seeds each — Phase 1 is
   complete, no more seeds needed on this question. (§22's `MAX_CONCURRENT=1` caution for 7-city
   turned out to be conservative — §23 ran this batch at `MAX_CONCURRENT=2` successfully, ~7h vs
   the ~12h sequential estimate.)
7. **NEW, most important open item: does the trained federated DQN actually beat simple rule-based
   control?** §24 found `fixed_time` (-2.73) and `max_pressure` (-0.34) both dramatically
   outperform the 7-city trained DQN (mean -6918.4, best-round -2182.0) on the same holdout city,
   once `fixed_time` was measuring real fixed-time control instead of a broken degenerate policy.
   This was invisible before today. Needs: (a) multi-seed rule-based baseline numbers (currently
   single-episode, deterministic, §24 flagged this as not yet done), (b) the same check on
   2-city/3-city rosters, (c) a decision on whether this blocks/reshapes Phase 2's strategy
   comparison — comparing aggregation strategies against each other is less interesting if none of
   them currently beat a simple heuristic. Don't scale up Phase 2 compute assuming the trained
   model is competitive until this is checked.
8. ~~**Is mean-pooling actively worse than no-neighbor-info, or was that one lucky/unlucky
   seed?**~~ **Resolved (as "it was one lucky/unlucky seed") — see
   [§31](#31-30s-5-seed-follow-up-the-single-seed-story-doesnt-replicate--no-clean-win-for-c-no-clean-loss-for-d-bcd-are-statistically-indistinguishable-from-each-other).**
   §30's one-seed finding was seed3-driven noise on both sides: D's catastrophic seed3 (best round
   -836.87) doesn't recur in seeds 1/2/4/5 (all between -19.58 and -49.87), and C's excellent seed3
   was its best of five, not typical. At 5 seeds, B/C/D are pairwise statistically indistinguishable
   on best-round (all |diff|/SE ≤ 0.73). Two real, narrower signals survived: C beats `fixed_time`
   on best-round (|diff|/SE=2.97) but not `max_pressure`; B is significantly *worse* than
   `max_pressure` on final-round (|diff|/SE=2.28). The code split (`--disable_head_fix` vs
   `--disable_neighbor_attention`, committed in `23c38c1`) is confirmed functionally real (C
   and D produce visibly different trajectories) — that part of §30 holds even though the
   attention-vs-pooling performance conclusion doesn't. Not run at 3-city/7-city — deprioritized,
   the 2-city signal is too weak to justify the compute right now.
9. **NEW from §32: weight-divergence/gradient-conflict is ruled out as the mechanism, not
   confirmed as it.** §28 flagged this as the priority diagnostic before trying more
   hyperparameters; run against 3 existing runs' checkpoints (`diagnostics/weight_divergence.py`,
   new and reusable). Neither cross-city cosine similarity nor weight-delta magnitude predicts a
   round's reward swing — correlations are weak and flip sign across runs, both whole-network and
   head-only. One real but non-predictive structural finding: the two cities' updates are
   persistently mildly opposed in the shared backbone (mean cos_sim -0.06 to -0.08, negative in
   16-18/19 rounds every run) but not in the output head (-0.02 to +0.05) — a constant background
   tension, not a crash signal. Since the obvious weight-space explanation is ruled out, the next
   candidate mechanisms are downstream of the weights: (a) small weight changes flipping the greedy
   action at a handful of pivotal intersections, amplified by SUMO's traffic dynamics into large
   queue swings, or (b) eval-episode/SUMO-seed noise being mistaken for policy regression in the
   first place.
   ~~(b)~~ **Rejected — see
   [§33](#33-32s-hypothesis-b-tested-and-rejected-the-crashes-are-real-reproducible-policy-failures-not-eval-noise--they-survive-6x-more-episodes-almost-unchanged).**
   Re-evaluated 3 checkpoints (1 "good" round, 2 "crashed" rounds) at 30 episodes instead of 5;
   both crashed rounds reproduced their bad reward almost exactly (round 20: -4089→-4095 mean,
   every one of 30 different SUMO seeds landing in the same narrow bad band). Genuine policy
   failures, not sampling noise.
   ~~(a)~~ **Refined and substantiated via Q-gap, not action-distribution diffing — see
   [§34](#34-33s-action-flip-hypothesis-a-tested-via-q-gap-crashed-rounds-show-a-genuinely-degenerate-near-seed-independent-policy--and-its-confidence-not-uncertainty-thats-the-signature-of-the-failure).**
   Crashed-round checkpoints show a genuinely degenerate policy (18/30 and 13/30 episodes producing
   byte-identical rewards across 30 *different* SUMO seeds), and `min_gap` correlates -0.884 with
   reward within the one checkpoint whose episodes span a real gap range — the network gets
   confidently locked into a bad repeating action, and rare low-confidence episodes are what let it
   escape. **New standing next step:** test whether softmax/stochastic action selection at
   eval/deployment time (instead of pure argmax) reliably breaks this lock-in — untested, needs a
   new evaluator policy branch (`federated/evaluator.py:132` currently hardcodes
   `explore=False`).
   **First direct test of the aggregation-vs-generic-instability question — see §48, corrected by
   [§49](#49-48s-5-seed-follow-up-and-a-correction-no-federation-training-does-show-the-same-confident-lock-in-as-federated-training-once-checked-properly--the-single-seed-read-was-an-artifact-of-trusting-a-5-episode-std-as-sufficient).**
   §48's single-seed pilot found no round with a training-time 5-episode std near the ~0.07-2 range
   federated crashed rounds show, tentatively reading that as evidence the lock-in was
   aggregation-specific. **§49 (5 seeds) reversed this**: a 30-episode `reeval_checkpoint.py` check
   on the lowest-std no-federation round found (5.64 at 5 episodes) landed on the exact same
   confident-lock-in signature as federated crashed rounds — 30 different SUMO seeds producing only
   two near-identical reward values. **The lock-in is not aggregation-specific** — independent,
   never-aggregated single-city DQN training shows it too, reframing §28's original question (it
   isn't "why does aggregation cause this," the lock-in looks like it predates aggregation
   entirely). ~~Still open: whether aggregation changes the lock-in's *frequency or severity* even
   though it isn't the root cause — not yet measured.~~ **Measured — see
   [§50](#50-49s-open-follow-up-answered-aggregation-does-not-measurably-change-the-confident-lock-ins-frequency--federated-and-no-federation-show-statistically-indistinguishable-lock-in-rates).**
   Matched, threshold-confirmed lock-in rate: federated 7/100 model-rounds (7.0%), no-federation
   12/200 model-rounds (6.0%), |diff|/SE = 0.34 — no statistically supportable difference.
   Aggregation neither causes nor measurably changes the frequency of this failure mode.
10. **NEW from §35: two concrete, cheap, literature-motivated experiments this project has never
    run.**
    (a) Train against `sumo_rl`'s built-in `pressure` reward instead of the default
    `diff-waiting-time` — PressLight/MPLight both train against pressure specifically because it's
    theoretically tied to throughput maximization (max-pressure control theory), and this project
    currently only uses pressure as a rule-based eval baseline, never as the training signal.
    ~~**In progress**~~ **Confound fixed and rerun — still doesn't beat `diff-waiting-time`, see
    [§37](#37-35s-experiment-a-pilot-result-pressure-reward-looks-worse-than-diff-waiting-time--but-reward_clip100-is-hardcoded-and-almost-certainly-destroys-most-of-pressures-signal)
    and
    [§38](#38-37s-fix-applied-pressure_norm-pilot-rerun-properly-scaled--still-doesnt-beat-diff-waiting-time-and-shows-the-same-degenerate-lock-in-signature-from-34).**
    §37's raw-`pressure` pilot was clip-saturated (confirmed empirically: 26% of ticks exceed the
    hardcoded `reward_clip=10.0` under random actions). Added `pressure_norm` (§38,
    `sumo_rl/environment/traffic_signal.py`, confirmed 0% clip-saturation) and reran — best round
    472.61s vs `diff-waiting-time`'s 3.20s, still two orders of magnitude worse, on one seed. Also
    surfaced an unplanned finding: 6/20 rounds show the same eval-episode-std~0 degenerate-lock-in
    signature §34 found in crashed `diff-waiting-time` rounds, suggesting that failure mode isn't
    specific to the default reward. Single seed only — not a settled verdict on pressure reward,
    would need the same multi-seed treatment as everything else here before trusting the direction.
    ~~(b)~~ **Tested — see
    [§36](#36-35s-experiment-b-softmax-eval-on-crashed-checkpoints-a-good-policy-is-reachable-from-the-exact-same-weights--pure-argmax-just-never-finds-it).**
    Softmax(Q/0.2) at eval time recovers near-optimal episodes (reward -36 to -104, on par with
    this doc's best results anywhere) from a checkpoint that pure argmax never once escaped in 30
    seeds — direct proof a good policy is reachable from the same weights. Not a clean win though:
    roughly a wash on mean reward (the non-escaped episodes get worse, not just neutral), so this
    is evidence of the mechanism, not yet a deployable fix. Next candidates, none tried: multi-
    sample-and-select at deployment, temperature annealing, or checking whether more training-time
    exploration (epsilon is ~0.05 by round 16-20) would let training itself find this branch
    instead of needing an eval-time patch.
11. **From the §36 discussion — two training-time (not eval-time) follow-ups on "would training
    with more exploration find the good branch on its own?"**
    ~~(a)~~ **Tested — a real but not uniform fix, see
    [§39](#39-item-11as-recovery-finetune-test-a-short-burst-of-reset-exploration-reliably-walks-locked-checkpoints-out-of-the-bad-regime--the-strongest-positive-result-in-this-document)
    and the relapse-risk follow-up,
    [§40](#40-relapse-risk-check-on-39s-recovery-durable-for-the-moderately-locked-checkpoint-not-durable-for-the-fully-locked-one--the-fix-isnt-uniform).**
    5-round recovery bursts (epsilon reset to 1.0, not `--resume`) reached a good round on both
    crashed checkpoints (§39). Extending to 15 rounds (§40) showed this only *durably* fixes the
    moderately-locked checkpoint (round 16: 9/10 good rounds in the back half, real convergence) —
    the fully-locked one (round 20, 0/30 escapes under pure argmax in §33) keeps relapsing through
    round 15, including a fresh crash to -4114.05 at round 14. Severity-dependent, not a universal
    one-shot cure.
    ~~(b)~~ **Cheap variant built and now fully tested (5/5 seeds) — clean null, not turning it on
    by default. Invasive variant still not worth building on this evidence.** `--epsilon_reset_every
    N` (periodic epsilon-greedy reset) tested first on the worst seed alone (§41: seed 5, no
    meaningful change) then across all 5 seeds of §21's validation set
    ([§42](#42-41-brought-to-all-5-seeds-epsilon_reset_every-5-is-a-clean-null-across-the-full-2-city-validation-not-just-the-worst-seed)):
    |diff|/SE = 0.07 (mean), 0.18 (best round) — both far below this project's ≥2 significance bar,
    mixed-sign per-seed deltas consistent with pure noise. Combined with §40's checkpoint-level
    result (durably fixes a moderately-locked case, doesn't durably fix a severely-locked one), the
    picture is now consistent and reasonably complete: exploration resets are a *targeted* tool for
    an already-detected locked/degenerate round, not a standing training-schedule improvement.
    Replacing epsilon-greedy with softmax(Q/T) as the exploration policy itself (needing a new code
    path in `agents/dqn.py`) remains untested, but §42's clean null on the cheaper variant makes it
    a lower-priority next spend than the still-open §28 aggregation-dynamics question.
12. ~~NEW, top priority from §43: the 2-city "best-round beats baselines" claim (§21, §29) does not
    survive a genuine holdout~~ **Resolved — see
    [§45](#45-43-brought-to-full-5-seed-rigor-2-city-plus-a-3-city-true-holdout-check-the-cleanest-most-statistically-overwhelming-result-in-this-entire-document).**
    (a) 5-seed 2-city confirmation: |diff|/SE = 5.05 (best-round), 13.79 (mean) vs `max_pressure` —
    the cleanest, most decisive result in this document, every seed's best round 3-4 orders of
    magnitude worse than either baseline. (b) 3-city pilot lands in the same range, doesn't change
    the picture. (c) correction notes added to §21, §29, §31. **At every roster size and every seed
    tested with a true holdout, the trained DQN loses decisively to both rule-based controllers.**
    Not yet done: 3-city multi-seed (low priority — nothing suggests it would tell a different
    story), and the B/C/D neighbor-attention conditions (§30/§31) haven't been re-run with
    `--pad_to_true_holdout` specifically, only flagged with a caveat.
13. **From §44: reward shaping's first pilot (7-city, `wait_weight=0.001`, 1 seed) looked worse
    than the unshaped baseline, not better — inconclusive (weight may just be too conservative),
    not a rejection.** Same degenerate-lock-in signature (§34/§38) showed up again under yet a
    third reward design. Worth a follow-up at a larger `wait_weight` (and/or `stopped_weight`)
    before concluding the idea doesn't work, but given §43's much bigger finding, probably lower
    priority than re-auditing existing claims and extending the true-holdout check to 3-city.
14. **NEW from §56: §55's z=2.71 lock-in-rate reduction for `--q_entropy_weight=0.05` is not
    reliable as reported — the std<50 screen has substantial false negatives on both the baseline
    and qew=0.05 arms.** A targeted near-threshold check (4 rounds per side) found 3/4 qew=0.05
    "clean" rounds and 4/4 baseline "clean" rounds were actually genuine confident lock-ins at
    30-episode rigor. Corrected minimum-bound counts (11/99 baseline, 3/100 qew=0.05) give z=2.24 —
    still above this project's bar, but a much weaker and less clean number, and only a *lower*
    bound since just 4 of each side's ~95-96 remaining non-flagged rounds were checked. **Not yet
    done: a full 30-episode recount of every non-flagged round on both sides** (~190 checkpoints,
    multi-hour batch) — the only way to get the real corrected rate and know whether the qew=0.05
    lock-in-reduction effect survives at all. Also from §56: phase-switch rate and dominant-action
    fraction on a non-locked checkpoint are essentially the same as `max_pressure`'s, ruling out
    switching behavior as the primary driver of the baseline gap — the open "what's the primary
    driver" question (§51) narrows further but is still unanswered.
15. **NEW from §57: reward-clip saturation ruled out (0.00% of ticks on the true holdout, even for
    a maximally-locked checkpoint) — the real failure is a small, persistent, whole-episode
    per-tick reward deficit (mean -0.899 vs `max_pressure`'s ~0.0000) that compounds toward the end
    of the episode (2.3x worse in the second half than the first) — and it's present essentially
    from round 1 of training, not something that grows in.** Three candidate mechanisms (lock-in
    §51, switching behavior §56, reward-clip saturation §57) are now ruled out as the primary
    driver. The open question has sharpened to: **why can't this architecture/observation design
    ever learn `max_pressure`-level moment-to-moment reactivity, at any point in training** — worth
    checking next whether the DQN's observation actually contains what `max_pressure` uses
    (approach/exit queue counts) with the fidelity `max_pressure` gets by computing its action
    directly from that signal, since the DQN only ever sees it indirectly through reward. This is
    the most promising concrete next step for the primary-driver search.
16. **NEW, top priority from §58/§59: the "trained DQN loses to baselines by 3-4 orders of
    magnitude" framing does not hold once training budget and evaluation protocol are matched to
    what RESCO's own published DQN numbers use — the single most important correction in this
    document, supersedes the framing (not the mechanism findings) of §43 onward.** A budget-matched
    (240 episodes vs. the standard 40), in-distribution (matching RESCO's own protocol, not the
    true-holdout generalization task), single-city (`environments_c4_only`, cologne3/Cologne
    Corridor, `--no_federation`) run reaches reward=-2.01/waiting_time=37.4s at its best checkpoint
    — 6.2x better than `fixed_time` (230.6s) and within 1.4x of `max_pressure` (27.3s), vs. RESCO's
    own published IDQN number for the identical scenario (8.5s) landing about 4.4x off. Round-to-
    round volatility (confident lock-in, §32-34/§51-53) is still fully present and unexplained, but
    the *ceiling* this run reaches is categorically different from anything in §43-§57's true-holdout
    numbers. **Single seed, single scenario, single budget point — needs 5-seed replication before
    trusting the magnitude, per this document's own standing pattern (§11→§12, §30→§31, §46→§47).**
    Next steps, in order: (a) 5-seed replication of this exact run, (b) the federated 2-city version
    at the same extended budget (does federation-with-adequate-budget also close most of the gap?),
    (c) revisit whether Phase 2 scaling should still be on hold given this changes why the baseline
    gap exists.
17. **NEW from §60/§61: item 16(b) done — the federated true-holdout gap improves significantly
    with more training budget (|diff|/SE 2.38 best-round, 2.43 mean) but stays enormous (~6700-
    17700x vs `max_pressure`), categorically unlike §59's near-total in-distribution closure.**
    Confirms the two comparisons this document has run (in-distribution vs. true-holdout) are not
    interchangeable — training budget was nearly the whole story in-distribution, but the
    cross-topology generalization penalty is real and budget-resistant on the true holdout. Not yet
    done: (a) ~~a robust 15+-episode re-evaluation of the standout checkpoint (seed3 round 50/59,
    launched, pending as of §61)~~ **done, see §61's own update**; (b) ~~the symmetric
    no-federation-at-63-rounds comparison~~ **done, see §64 — still no significant difference,
    |diff|/SE 1.78/1.43, extending §49/§50's finding to the new budget (3-seed, scoped down, see
    §60's addendum)**; (c) a real budget-vs-performance curve (more than the 2 points now available: round 20, round
    63) before trusting any extrapolation about how much more training would close the remaining
    true-holdout gap, or whether it's asymptoting well short of baseline performance.
18. **NEW from §62: the first intervention actually targeted at the true-holdout generalization gap
    (rather than training-instability symptoms) — added `max_pressure`'s exact input signal
    (outgoing-lane pressure/density, confirmed structurally absent from the DQN's observation, not
    just underused) to `own_obs`.** `own_dim` 115→117. Pilot launched matching §60/§61's exact
    protocol (seed 3, `environments_c1_4`, `--pad_to_true_holdout`, 63 rounds) for a clean
    before/after comparison against that seed's known result (best=-327.10). Result pending. Given
    this is a genuine architecture change (`own_dim` changed), no existing checkpoint can be resumed
    into it — every run from here trains fresh.
19. **NEW from §63: the pressure-feature pilot result is in — worse than baseline on this seed
    (best -3844 vs -327, mean(21-63) -5116 vs -2906), not better.** Single seed, and not a perfectly
    matched pair even at the same nominal seed (`own_dim` change shifts the whole downstream RNG
    stream — see §63 for why). Genuinely uncertain, not a confirmed refutation. **Not yet decided:
    whether to run multi-seed replication of this specific feature, or deprioritize it now in favor
    of the other discussed levers (clustered federation — code already exists, cheapest; few-shot
    calibration; wider/more diverse training roster) — open decision point for next session.**
