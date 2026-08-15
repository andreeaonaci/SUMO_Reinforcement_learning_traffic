# Federated Traffic-Signal RL — Execution Plan

## How to use this document

This file is the source of truth for what to do next on this project. Each
phase has a **Goal**, a **What to investigate / implement** section written
at concept level (not exact code, not exact hyperparameters — use judgment
and the existing codebase conventions), an **Expected output**, and a
**Decision gate** that tells you whether to move to the next phase, iterate,
or stop and report back to the user.

Rules to follow throughout:
- Do not change the core aggregation mechanism under test (the masked-head
  fix) unless a phase explicitly says to. Infrastructure fixes (target
  network, LR overrides, buffer persistence, etc.) are orthogonal to the
  research contribution and should not touch `masked_head_weighted_average`
  or its sibling `weighted_average` unless told to.
- Prefer the cheapest experiment that can answer the question. Full-roster,
  many-seed, many-round runs are expensive — earn the right to run them by
  passing the decision gate on a cheap version first.
- Every run should log enough to be analyzed later (per-round metrics,
  config used, seed). If a phase's output can't be distinguished from noise
  without re-running, log more before declaring the phase done.
- When a phase's decision gate is ambiguous, don't guess — report findings
  and open questions back to the user rather than silently proceeding.
- Update the "Status" line under each phase heading as work progresses
  (Not started / In progress / Blocked / Passed gate / Failed gate — see
  note below).

## Parallel work tracks

Two tracks can run independently. Track A is compute-heavy and strictly
sequential (each phase's output gates the next). Track B has no dependency
on training stability and can start immediately, in parallel with Track A.

**Track A (sequential, compute-bound):**
Phase 0 → Phase 1 → Phase 2 → (Phase 3's learned-baseline runs)

**Track B (parallelizable, start anytime):**
- Phase 3's rule-based baselines (fixed-time, max-pressure) — no DQN
  training involved at all.
- Phase 4's code implementation (clustered aggregation strategy) — can be
  written and smoke-tested now; just don't trust its *results* until
  Track A's Phase 0 has passed its gate.
- Phase 4's literature/related-work writing — zero compute dependency.
- Feature Development ideas below — pick opportunistically when Track A is
  blocked on a long-running experiment and there's spare time.

---

## Phase 0 — Infrastructure Stabilization

**Status:** In progress (target-network gap identified, not yet fixed)

### Goal
Establish that a single city, trained without any federation involved, can
actually learn — i.e. loss trends down and holdout reward stabilizes over
enough rounds. Right now it does not: loss increases round over round and
reward oscillates by thousands of points even after exploration has fully
decayed. This is a training-stability bug, not a "needs more rounds"
problem, and it invalidates every experiment run so far.

### What to investigate / implement (concept level)
- **Target network.** Confirmed missing entirely. A DQN bootstrapping its
  target off the same network it's updating is a textbook divergence cause.
  Add a standard target network: a copy of the online network used only for
  computing the next-state value in the TD target, updated on a delay
  (hard copy every N steps, or soft/Polyak update — pick whichever fits the
  existing agent class most naturally). Keep it out of the aggregation
  path — only the online network's weights should be averaged across
  clients; the target network is a per-agent training-stability device, not
  something the federation logic needs to know about.
- **Optimizer and replay-buffer persistence across rounds.** Verify that
  the same agent object (with its optimizer state and replay buffer) is
  reused round over round for a given city, not rebuilt from scratch each
  round. If it's rebuilt, Adam's momentum/variance history dies right after
  the weights just got perturbed by averaging — compounding instability.
  Same logic for the replay buffer: if it resets every round, the agent has
  almost nothing to learn from before its weights are overwritten again.
- **Per-city learning-rate overrides.** A hardcoded override was found
  still active in diagnostic runs despite an earlier attempt to remove it.
  Find and eliminate any hardcoded per-city LR logic so that a `--lr` flag
  (or equivalent) uniformly controls every client's learning rate, with no
  silent exceptions. This is a correctness bug independent of the target
  network fix and must be confirmed fixed before trusting any LR-related
  experiment.
- **Reward magnitude.** Rewards are currently on the order of -6000 to
  -10000. Check whether any normalization or scaling reaches the TD error
  before backprop. Very large unscaled rewards produce very large gradients
  and can compound instability on top of the missing target network. Not
  necessarily a blocker on its own, but worth a quick check — if reward
  scaling is trivial to add, do it as part of this phase rather than a
  separate one later.
- **Parallel execution vs. libsumo.** Already diagnosed: libsumo does not
  support multiple concurrent simulation instances within one process,
  which conflicts with thread-based `--parallel` city workers. A working
  fallback (real traci) was already used to unblock this. If revisiting for
  speed later, the correct long-term fix is process-based parallelism (one
  OS process per city, each with its own libsumo instance) rather than
  threads — but this is a performance nice-to-have, not a correctness
  blocker, and can be deferred.
- **Incremental checkpointing.** Already implemented (partial history saved
  every round). Confirm it's still working correctly after any changes in
  this phase — don't regress it.

### Expected output
A single-city, non-federated training run (existing `--no_federation`-style
mode, or equivalent) over a meaningful number of rounds shows:
- Training loss trending downward, not upward, across rounds.
- Holdout eval reward variance shrinking over time rather than oscillating
  at a constant or growing magnitude.
- The action distribution genuinely diversifying/stabilizing into a
  sensible policy rather than collapsing onto a single dominant action that
  changes arbitrarily round to round.

### Decision gate
- **Pass →** loss decreases and reward stabilizes on the single-city,
  non-federated test. Move to Phase 1. Also re-enable federation (multiple
  cities + aggregation) and confirm the same qualitative pattern holds
  before fully trusting Phase 1's results — federation could still
  reintroduce instability even after the per-agent fix, which would itself
  be a notable and reportable finding.
- **Fail / ambiguous →** do not proceed to Phase 1. Report which of the
  above fixes were applied, what the resulting loss/reward curves look
  like, and flag remaining candidates from the list above that haven't been
  tried yet.

---

## Phase 1 — Core Claim Validation (cheap subset)

**Status:** Complete — all three roster sizes now have 5 seeds per arm. See
`fidings/divergence_investigation.md` sec 20 (2-city, 3-city, 2026-08-11/12)
and sec 23 (7-city brought to 5 seeds, 2026-08-13), on top of the validated
`dueling+n_step` config (sec 19). Final pattern: the fix's *mean-reward*
benefit shrinks monotonically with roster size and is gone by 7 cities
(|diff|/SE: 3.42 at 2-city → 0.71 at 3-city → 0.23 at 7-city — a genuine null,
not a data gap). Its *best-round* benefit is real and survives at every
roster size tested (fix-on reaches a meaningfully better peak at 2, 3, and
7 cities), though the relative margin also shrinks with scale (~95% → 76% →
44% better best-round mean). Honest framing for the paper: the masked-head
fix reliably helps a run reach a better peak policy at any scale tested, but
its effect on expected/average performance is roster-size-dependent and not
distinguishable from noise at full scale.

### Goal
Determine, cheaply, whether the masked-head aggregation fix actually
produces a distinguishable, non-noise improvement over naive FedAvg on the
head layer — before spending the compute to prove it at full scale.

### What to investigate / implement (concept level)
- Use a small subset of cities that still preserves real action-space
  heterogeneity (so the row-sparse-starvation mechanism the fix targets can
  actually manifest), plus the (topology-corrected) holdout city.
- Compare the fix on vs. off, holding the aggregation strategy itself fixed
  to plain FedAvg (i.e. isolate the head-layer mechanics from the
  client-weighting question — that's a separate, later comparison).
- Run enough seeds to distinguish signal from noise — not one, and enough
  that per-condition variance can be judged against the gap between
  conditions.
- Check per-seed results individually, not just the aggregate mean/std — a
  single outlier seed can otherwise masquerade as a real effect or wash out
  a real one.

### Expected output
A small table (metric × condition × seed) showing whether "fix on" and
"fix off" produce meaningfully separated distributions on the core
evaluation metrics (reward, waiting time, stopped vehicles, arrived
vehicles), with enough seeds that the separation (or lack of it) isn't
plausibly just noise.

### Decision gate
- **Clear separation, fix wins →** proceed to Phase 2 (scale up).
- **Clear separation, fix loses or reverses →** this is itself an important
  finding — don't discard it. Report it; the paper's framing may need to
  shift toward a negative/conditional result (see the Feature Development
  section for how to reframe this constructively).
- **Ambiguous / overlapping distributions →** do not scale up yet. Consider
  whether more seeds would resolve the ambiguity cheaply, or whether the
  mechanism needs to be re-examined (e.g. is the effect real but small,
  or was the original diagnosis — row-sparse gradient starvation under
  FedAvg — actually correct?).

---

## Phase 2 — Full-Scale Validation & Strategy Comparison

**Status:** Unblocked (Phase 1 complete) but not yet started — decision gate outcome is mixed, not
a clean "fix wins," see Phase 1 status above. 2-city passed the gate cleanly; 7-city (the real
paper roster) shows a null mean-reward result and a real-but-shrinking best-round win. Whether to
proceed to full Phase 2 scale-up as-is, reframe around the best-round-only claim, or investigate
the mean-reward vanishing point further first is a call for the user, per the plan's own
instruction not to guess on an ambiguous gate outcome.

### Goal
Confirm the Phase 1 result holds across the full, real city roster with
enough seeds to report a defensible statistic in the paper, and compare all
registered aggregation strategies against each other now that the fix's
value is established.

### What to investigate / implement (concept level)
- Repeat the fix-on/fix-off comparison from Phase 1 on the full city roster
  (not just the cheap subset), with more seeds than Phase 1 used, since
  this is the number that goes in the paper.
- Separately, compare all registered aggregation strategies against each
  other with the fix enabled, on the full roster. This answers a secondary
  but real question: does smarter client weighting (loss-based, alignment-
  based, novelty-based, survival-based) add anything on top of the
  structural head-layer fix, or is the fix doing all the work?
- For any strategy that produces per-round client weights, check whether
  the dominant client rotates across rounds (healthy) or fixates on the
  same client every round (a red flag worth investigating and reporting,
  not silently ignoring).
- Plot convergence curves with confidence bands across seeds for the
  headline comparison — this becomes a paper figure.

### Expected output
- A full-roster fix-on/fix-off result table with enough seeds to support a
  significance claim (or an honest "not significant, here's the effect
  size and confidence interval" if that's what the data shows).
- A strategy-comparison table (all registered strategies, fix on, full
  roster) with the same rigor.
- A rotation-vs-fixation characterization for each weighting strategy that
  produces per-round weights.

### Decision gate
- **Effect confirmed at scale →** proceed to Phase 3/4 (can now run in
  parallel — baselines and literature positioning). The core claim is
  established; remaining work strengthens the paper around it.
- **Effect shrinks or disappears at scale →** report honestly. This does
  not mean the project has no paper — see Feature Development for framing
  options (conditional effects, mechanism-level findings, negative results
  are still publishable when the diagnosis is rigorous).

---

## Phase 3 — Baselines

**Status:** Rule-based `fixed_time` baseline was found broken and fixed 2026-08-13
(`fidings/divergence_investigation.md` §24) — every prior `fixed_time` number in the project's
history measured a degenerate "never switch phase" policy, not real fixed-time control. Corrected
`fixed_time` (-2.73) and `max_pressure` (-0.34) both now beat the 7-city trained DQN (mean -6918.4)
by a wide margin on the same holdout city. This is a live, urgent open question (not yet multi-seed
or checked on other rosters) that should be resolved before trusting Phase 3's baseline table or
scaling Phase 2 compute on the assumption the trained model is already competitive. Rule-based
portion can otherwise proceed (Track B). Learned portion blocked on Phase 0 (Phase 0 itself is
fine — see status above).

### Goal
Establish what the federated/fix-enabled approach is actually being
compared against, so "our method is better" has a meaningful reference
point rather than only being compared against ablated versions of itself.

### What to investigate / implement (concept level) — Track B, no blocker
- **Fixed-time control.** A traffic signal cycling through its native/
  default phase program with no learned policy involved at all. Cheapest
  possible baseline, and a sanity floor — if a trained policy loses to
  fixed-time, something upstream is badly wrong.
- **Max-pressure control.** A simple rule-based controller that, at each
  decision point, favors the phase serving the movement with the highest
  "pressure" (roughly: upstream queue/demand minus downstream capacity).
  Standard reference baseline in the traffic-signal-RL literature —
  reviewers will expect to see it.
- Both of these should be evaluable through the same evaluation path used
  for trained agents, so their numbers land in the same table with no
  special-casing needed downstream.

### What to investigate / implement (concept level) — blocked on Phase 0
- **Independent (non-federated) DQN per city.** Same agent code, same
  fixed infrastructure, but no cross-city averaging at all. Tells you
  whether federation is worth doing in the first place, independent of the
  head-fix question — a federated method that loses to independent local
  training would undercut the paper's premise regardless of the fix.
- **Centralized upper bound (optional, higher cost).** Pooling all
  cities' data into one training run. Useful as a ceiling reference, but
  lower priority than the independent-DQN baseline — only pursue if time
  allows after everything else in this phase and Phase 2 are done.

### Expected output
A baseline table (fixed-time, max-pressure, independent DQN, optionally
centralized) on the same metrics and same holdout city as the main results,
ready to sit alongside the Phase 2 table in the paper.

### Decision gate
- Rule-based baselines: done once numbers are captured and sanity-checked
  (trained policies should generally beat fixed-time; if they don't,
  investigate before treating Phase 0-2 results as trustworthy).
- Learned baselines: only run these after Phase 0 has passed its gate —
  otherwise they inherit the same instability and are not meaningful.

---

## Phase 4 — Related-Work Positioning & Clustering Comparison

**Status:** Code and writing can start now (Track B). Result trust blocked
on Phase 0/2.

### Goal
Turn "our method is different from prior federated-TSC work" from a
citation-only claim into a tested empirical comparison, and produce the
related-work section.

### What to investigate / implement (concept level)
- **Clustering-based aggregation strategy**, modeled on the general idea
  used in prior federated-TSC work that groups cities by similarity and
  runs ordinary (non-masked) averaging within each group rather than
  per-row masking globally. Implement this as one more entry alongside the
  existing registered strategies, using whatever grouping signal is
  simplest to justify (action-space size is a reasonable first pass since
  it's directly on-axis with the mechanism under study; more sophisticated
  traffic-pattern similarity is a possible refinement, not a requirement).
  Note explicitly in any writeup that this is an approximation of the
  general clustering idea, not a faithful reproduction of any single
  paper's exact method.
- Because this strategy naturally produces one aggregated model per
  cluster rather than one global model, make sure the broadcast step
  correctly routes each cluster's model back only to its own members —
  this is a structural difference from every other registered strategy and
  needs explicit handling, not an assumption that the existing broadcast
  logic already covers it.
- **Related-work writing**, in parallel with the above, covering the
  general shape of the field: graph/attention-based non-federated
  coordination approaches, federated approaches using parameter sharing,
  federated approaches using clustering, and any standardized benchmarking
  frameworks relevant to this problem class. For each, describe the
  mechanism in one's own words and state explicitly how this project's
  approach differs (per-row/per-client gradient-sparsity-aware weighting
  inside the aggregation step, orthogonal to architecture) rather than just
  asserting difference.

### Expected output
- A working clustering-based strategy in the registry, smoke-tested for
  correctness (no crashes, sensible cluster assignments logged).
- A related-work section draft.
- Once Phase 0/2 have passed their gates: a real comparison run (same
  seeds/roster/eval protocol as the main results) showing where the
  head-fix approach lands relative to the clustering approach.

### Decision gate
- Code and writing: done when implemented/drafted and smoke-tested.
- Comparison numbers: don't report or trust them until Phase 0 has passed
  and this strategy has been run through the same rigor as Phase 2
  (multi-seed, full roster).

---

## Feature Development — Optional SOTA-adjacent extensions

These are not required for a publishable result — Phases 0-4 alone produce
a complete paper. Pick these up opportunistically (e.g. while a long
compute run is in progress) if there's spare time, and only if they don't
risk destabilizing what's already working. Rank roughly by effort-to-value:

- **Dueling network head + n-step returns — CURRENT BEST CONFIG:
  `--dueling --n_step 3`.** Implemented and tested 2026-08-06/10
  (`agents/networks.py::NeighborAttentionQNetwork`, `agents/dqn.py`'s
  `_remember_step`/`_flush_nstep`). Dueling alone is a clean, generalizing
  win (~26-35% better mean, 5-8x better best round, both rosters, sec 15).
  n-step alone is even stronger on its own (beats dueling alone on every
  metric on 2-city). **Combined, they're a genuine synergy** — #1 on mean
  AND best round on both rosters, beating both ingredients individually,
  not just averaging them (`fidings/divergence_investigation.md` sec 19).
  Best round on 3-city (-18.2) is the best number recorded anywhere in
  this investigation. **This supersedes "use dueling alone" as the
  recommendation — use both flags together.**
- **Server-side momentum (FedAvgM-style).** Implemented and tested
  2026-08-06/07 (`--server_momentum`, see
  `federated/aggregation.py::apply_server_momentum`). Targets a different
  symptom than the masked-head fix or FedProx: the *aggregated* global
  model swinging sharply round to round with no matching spike in any
  individual client's local loss (`fidings/divergence_investigation.md`
  sec 9) — applies this round's aggregated update through an
  exponentially-weighted velocity buffer instead of jumping straight to
  the raw aggregate, damping the applied update itself at the server
  rather than anything client-side. Result: modest, mixed benefit at
  beta=0.9 — ~9-10% better mean on both rosters, but damps peaks along
  with crashes (worse best-round on the 2-city roster) — see sec 16. Real
  but weaker than dueling. Cheap (a few lines, one buffer, no client-side
  change), beta=0 is an exact no-op.
- **Dueling + momentum combined.** Tested 2026-08-07/09, both rosters —
  net-negative interaction (`fidings/divergence_investigation.md` sec 18).
  Dueling alone beats dueling+momentum on mean and best round on both
  rosters; momentum's update-damping works against the fast, undamped
  advantage-head movement that makes dueling effective on its own.
  **Recommendation: use `--dueling` alone, do not combine with
  `--server_momentum`.**
- **EMA-averaged eval snapshot.** Implemented and tested 2026-08-09/10
  (`--eval_ema_decay`, see `federated/server.py` and
`federated/parallel_server.py`'s eval-time state-swap). Scoped to 
  evaluation only (not checkpoint files) — maintains a separate EMA copy
  of the aggregated weights, temporarily swapped in for the evaluator call
  and swapped back out immediately after, so training/broadcast is
  untouched. `eval_ema_decay=0` (default) is an exact no-op. Result: does
  exactly what its mechanism predicts and nothing more — compresses
  *reported* variance (best worst-case of anything tested, worst
  best-case of anything tested, both rosters) without improving training,
  since it can't by construction. Combined with dueling: decent on
  2-city, net-negative on 3-city — not reliable enough to recommend
  generally (`fidings/divergence_investigation.md` sec 19).
- **Server-side pseudo-gradient clipping.** Implemented and tested
  2026-08-09/10 (`--pseudo_grad_clip`, see
  `federated/aggregation.py::clip_pseudo_gradient`). Caps the round's
  applied update's total L2 norm, rescaling uniformly if over the cap;
  applied before `--server_momentum` if both are set. Clip threshold
  (1.5) chosen from real data (round-to-round delta norms ~1.2-3.3 on an
  existing dueling run). `pseudo_grad_clip=0` (default) is an exact
  no-op. Result: helps on 2-city, barely distinguishable from baseline on
  3-city — the fixed threshold doesn't transfer across roster sizes the
  way dueling/n-step's mechanisms do. Combined with dueling: no clean win
  on either roster (`fidings/divergence_investigation.md` sec 19).
- **Double DQN.** Once the target network exists (Phase 0), Double DQN is
  a small conceptual change (use the online network to *select* the
  best next action, the target network to *evaluate* it) that further
  reduces overestimation bias. Cheap to add right after Phase 0's fix and
  worth strongly considering as part of that same fix rather than a
  separate pass.
- **Communication-efficient federated updates.** The paper's whole premise
  is that per-city head rows are sparse (many action rows barely touched
  by a given city). That's a natural bridge to a bandwidth-savings angle:
  instead of transmitting full head-layer weights every round, transmit
  only the rows a city actually updated meaningfully. This is a genuinely
  novel extension of the existing contribution rather than a new project —
  worth flagging to the user as a possible "bonus" result section rather
  than pursuing unprompted, since it changes what gets measured
  (bandwidth) in addition to what's already measured (accuracy).
- **Personalization-layer comparison (FedPer-style).** An alternative to
  the masked-head fix: instead of fixing how the head layer is averaged,
  don't average it at all — keep it fully local/personalized per city and
  only share the body of the network. This is conceptually adjacent to the
  paper's core question (how should a shared network handle
  per-client-heterogeneous output spaces) and would make a strong
  additional baseline/strategy in the registry alongside clustering.
- **Uncertainty- or variance-aware aggregation weighting.** A strategy
  that weights clients not just by loss/alignment/novelty but by the
  estimated variance/confidence of their local updates — conceptually a
  Bayesian FedAvg variant. Same registry slot as the existing strategies;
  moderate effort, uncertain payoff, but a reasonable ablation if the
  existing five strategies leave an open question about *why* one wins.
- **Adaptive local-episode count per client.** Cities with more
  intersections (more local data per round) may benefit from a different
  number of local training episodes before aggregation than small cities.
  Currently a global flag; making it per-client and adaptive (e.g. scaled
  by intersection count or by a convergence signal) is a plausible
  contribution to training stability specifically for heterogeneous
  federated settings — worth considering only after Phase 0's core fixes
  are confirmed to be enough on their own; don't add this complexity
  preemptively.
- **GNN-based neighbor coordination.** Out of scope as a codebase change —
  this would be a different architecture, not an aggregation-mechanism
  paper. Mention only as future work in the paper's conclusion, do not
  implement.

If any of these get picked up, treat them the same way as the numbered
phases: define a concept-level goal, an expected output, and a decision
gate before starting, and don't let them block Track A's critical path.