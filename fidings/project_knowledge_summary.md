# Everything we know — master summary (as of 2026-09-09)

> **READ THIS FIRST — the framing below sections 1-6 is superseded.** Between 2026-09-08 and
> 2026-09-09 (`divergence_investigation.md` §95-§100) the cross-topology gap was traced to the
> **action representation** — the indexed Q-head — and a replacement (`--phase_relational`) now
> beats `max_pressure` zero-shot on the unseen holdout, confirmed at 6 seeds on four separate
> configurations. **Section 7 at the bottom of this file has it.** Sections 3-6's ~20 null results
> were all measured on the broken readout and should be read as floor effects, not as evidence
> that architecture/aggregation/curriculum don't matter.

**Purpose of this document:** a single entry point for "what has this project actually found,"
spanning the whole research arc, not just one session or one campaign. Other documents hold the
full detail this one condenses:

- `fidings/divergence_investigation.md` — the complete, dated, section-by-section investigation
  log (79 sections as of this writeup). The source of truth for exact numbers and derivations.
- `fidings/algorithm_swap_summary.md` — a standalone deep-dive on the PPO/Munchausen-DQN/
  architecture-search campaign (§73-77) specifically.
- `README.md` — the condensed, paper-oriented public summary (DQN-era results + the algorithm-swap
  campaign's headline finding).
- `CLAUDE.md` — engineering/architecture reference for the codebase itself, not results.

Treat this document as a map, not the territory — if a number here ever disagrees with
`divergence_investigation.md`, the investigation log wins; update this file to match.

---

## 1. What this project is

A federated reinforcement learning pipeline: one shared DQN policy ("foundation model") controls
traffic signals across multiple SUMO-simulated cities of different intersection topologies
(3-way/4-way/5-way, different neighbor counts), trained via FedAvg-style aggregation, with
topology differences expressed purely through observation/action masking (never per-topology
code). The central research question: can one shared policy generalize across genuinely different
intersection topologies, and if it struggles, why?

---

## 2. The DQN-era findings (the original research arc, through §71)

**Headline finding:** a trained DQN policy loses to simple rule-based control (`fixed_time`,
`max_pressure`) by 3-4 orders of magnitude when evaluated on a truly unseen intersection topology
— but this gap is **mostly a training-budget/evaluation-protocol artifact, not evidence DQN
"doesn't work."** Evaluated in-distribution with a training budget matched to the RESCO benchmark's
own published numbers, the same architecture comes within 1.4x of `max_pressure`. The gap that
remains under true cross-topology evaluation is real, large, and only partially closed by more
training budget (best-round mean -5278.1 → -2285.2 going from 20 to 63 rounds of training, still
~6,700-17,700x off baseline at the higher budget).

**Two compounding, independently-verified mechanisms:**
1. **Confident lock-in** — the policy sometimes converges to a state where it repeats one action
   regardless of actual traffic conditions, confirmed by re-evaluating checkpoints across 30
   different random seeds and finding byte-identical rewards. Happens whether or not federation is
   involved (turning federation off entirely reproduces the identical failure rate, 7.0% vs 6.0% of
   checkpoints, statistically indistinguishable) — it's a property of DQN training against this
   task, not an aggregation artifact.
2. **Cross-topology generalization gap** — the policy must control layouts it never trained on;
   rule-based baselines apply the same fixed formula everywhere and never face this problem.

**What was tried to close the gap (training/aggregation-time), and all came back null or negative:**
alternative aggregation strategies (clustered/EMA-loss/EMA-alignment/gradient-survival/velocity-
novelty), architecture changes beyond dueling+n-step, an extra `max_pressure`-style input feature,
reward shaping (the *first* attempt, ad hoc weights — see item 22 below for the theory-grounded
retry), and widening the training roster from 3 to 14 cities / 26 to 116 intersections (clean null
— ruling out "not enough training data/diversity").

**One lever that did help, decisively: test-time fine-tuning.** Briefly fine-tuning a trained
checkpoint on synthetic randomized traffic on the target topology itself (never the real
evaluation route file), then evaluating on real holdout traffic. Multi-seed replicated across
4 seeds, every seed/round beating zero-shot by at least 3.9x (|diff|/SE up to 72.78). Still
20-3200x off rule-based baselines in absolute terms — real and well-replicated, not a solved
problem.

**The open, uncomfortable finding that reframed everything:** a **randomly-initialized** network,
fine-tuned on the holdout city, **beat** the federated-**pretrained** network fine-tuned the same
way (-406.85 vs -693.84, single seed at the time). Combined with the roster-widening null, this
pointed at the sharpest form of the project's central negative result: **the binding constraint is
not the training data's quantity or diversity — it's the algorithm's failure to retain and
transfer what it learns.**

---

## 3. The algorithm/architecture-search campaign (§73-77, one autonomous ~8h session)

Direct response to finding #2 above: if DQN itself can't retain/transfer what it learns, is a
different algorithm or architecture the fix? Result, after ~30+ training runs across 6 independent
axes, several with genuine 3-6-seed multi-seed rigor: **no.**

| axis | what was tried | result |
|---|---|---|
| Algorithm | PPO, Munchausen-DQN (multiple hyperparameter configs) | statistically tied with DQN at best |
| Network width | `d_model` 64, 256, 512 | all worse than the 128 default |
| Normalization/activation | BatchNorm1d + relu6/leaky_relu | statistically tied |
| Network depth | `encoder_depth` 3, 4, at both 5-round and 20-round budgets | confirmed worse at both budgets, not just underbudgeted |
| Attention structure | stacked multi-layer attention (`n_attn_layers` 2, 3) | looked promising on 3 seeds, died at 6 |
| Training procedure | `local_episodes` doubled | null, slightly worse on mean |

**This session's own most valuable finding is methodological, not architectural:** three separate
few-seed "leads" each looked like a real, clean win on 3 seeds — each driven by exactly one
dramatic outlier seed — and each collapsed to a null once 3 more seeds were added
(`temp=0.01+n_step=3`; `n_attn_layers=2`, twice over). **Nothing under ~5-6 seeds should be
trusted at this roster/budget, full stop**, regardless of how clean or dramatic a smaller sample
looks. This lesson has been applied consistently to every result reported since.

**Standing conclusion:** the confident-lock-in / retention-failure problem is not fixable by
changing the network's algorithm, size, or structure. It looks like a property of the training
*dynamic* itself — off-policy bootstrapping against a non-stationary, federated-aggregated target
— which no architecture variant tested touches.

---

## 4. Current phase: mechanism-hunting via genuinely different paradigms (§78+, in progress)

Per direct user request, six directions "as far from the tested axis as possible" were queued
(`divergence_investigation.md`, "Open questions" items 20-25) and are being implemented and
validated **in this order**: replay-buffer reset → SWA/ensemble → potential-based reward shaping →
recurrent policy → meta-learning aggregation → evolution strategies.

### Item 20 — Replay-buffer reset on detected lock-in: **null, closed out**

Hypothesis: a locked policy's own self-generated, increasingly homogeneous transitions perpetuate
the lock via TD-bootstrapping off stale data. Implemented `--lockin_reset_std_threshold` (clears
every worker's replay buffer when the cheap std<50 lock-in screen fires). Verified triggering
correctly on real training (confirmed via logs on all 3 seeds). **Result: |diff|/SE = 0.30
(best-round), 0.38 (mean) vs. baseline — null.** The mechanism worked exactly as designed; clearing
the buffer just didn't change outcomes. Rules out this specific hypothesis about *why* lock-in
perpetuates, without needing to abandon the lock-in diagnosis itself (§51 already established
lock-in is a secondary factor in the gap, not the primary one — consistent with this null).

### Item 21 — SWA-style checkpoint averaging/ensembling at eval time: **closed — real effect,
not a deployable fix**

Every prior lever changed something about *training*. This was the first eval-time-only lever
tested: average (true SWA) or majority-vote-ensemble several consecutive round checkpoints instead
of picking one. Built `diagnostics/swa_reeval.py`.

**Original window** (5 consecutive checkpoints from a volatile run, confirmed at 30 episodes):

| | reward |
|---|---:|
| Individual checkpoints | -6083, -4651, **-4002 (best)**, -5715, -6849 |
| SWA weight-average | **-4390** |
| Majority-vote ensemble | **-4464** |

Both combination methods landed within ~10-11% of the best individual checkpoint — confirmed, this
direction held from 10 to 30 episodes, unlike most single-window leads in this project.

**Independent generalization window** (different run: `encoder_depth=3`, seed 7 — a *stable*
stretch, not a volatile one): combination did NOT help — SWA landed near the mean (-8546 vs. best
-8452), and the ensemble did worse than **every single individual checkpoint** (-8945 vs. -8452
best / -8925 worst).

**Verdict: a real, mechanistically-sensible effect — but conditional on volatility, not a general
fix.** It rescues near-best performance when a good round is surrounded by bad ones; it adds
noise-from-disagreement (and can actively hurt) when checkpoints are already converged and merely
differ by chance. The original pitch — "near-best performance without knowing which round was
best" — doesn't survive: telling a volatile window from a stable one requires the same per-round
eval sweep that would let you pick the best round directly. **Item 21 closed as a non-deployable
but real finding.** Moving to item 22.

**Two real bugs found and fixed while building this (both now committed):**
1. §76's attention-stacking refactor had silently broken loading of *every* checkpoint saved before
   that commit (a state-dict key rename, not a shape change) — would have hit `--resume` and every
   other checkpoint-loading script, not just this new one. Fixed with a backward-compatible
   key-remapping shim.
2. The shared `infer_arch_from_checkpoint` helper didn't detect this session's newer architecture
   knobs (`encoder_depth`, `n_attn_layers`), so it silently assumed old defaults — fine for old
   checkpoints, a shape-mismatch crash for anything trained with those flags. Fixed to detect both
   directly from the state dict.

### Item 22 — Potential-based reward shaping using `max_pressure`'s own formula: **CONFIRMED, the
first real training-time win in the item-2X series**

Unlike the earlier ad hoc shaping attempt (§44, arbitrary weights, inconclusive), potential-based
shaping (Ng, Harada & Russell 1999) is mathematically guaranteed not to change the optimal policy
— `F(s,a,s') = gamma*Phi(s') - Phi(s)` added to the training reward, with `Phi(s) = weight *
{ts}_pressure` (the same signal `max_pressure` itself maximizes). Isolates a learning-*dynamics*
effect (denser, better-aligned per-tick signal) from a different-optimal-policy effect.

At 6 seeds (3/7/11/17/21/25, `weight=0.1`, layered on top of the already-adopted
`q_entropy_weight=0.05`): best-round |diff|/SE=2.53, mean-reward |diff|/SE=2.49, both clearing this
project's bar — and **5 of 6 seeds individually favor it, the 6th is a near-exact tie, not a
reversal**, unlike every architecture-search lead that turned out to be one outlier seed. Magnitude
is real but modest (~4-7.5% better), nowhere near closing the multi-order-of-magnitude gap to
`fixed_time`/`max_pressure`. **This is the first training-time lever in the entire item-2X series
(20-25) to confirm as a genuine, non-outlier-driven improvement** — every other training-time
attempt in this project (aggregation strategy, architecture, extra features, ad hoc reward shaping,
roster diversity, replay-buffer reset) came back null or negative. Not yet tested at a longer
training budget or other weight values — flagged as a good follow-up once the item-2X queue is
finished, not urgent right now.

### Item 23 — Recurrent policy (GRU): **inconclusive, not confirmed**

A GRUCell-based hidden state per intersection (stored-state DRQN, `agents/recurrent_dqn.py`,
`--algo recurrent`) gave the network actual memory across ticks — every architecture tried in
§73-76 was still a purely reactive function of one tick's snapshot. At 6-seed rigor: |diff|/SE 1.63
(best-round), 1.98 (mean) — mean sits right at the bar without clearing it, best-round stays clearly
under. 4 of 6 seeds favor it (two substantially) but 2 of 6 are actively worse — a real direction
split, not a clean win. Also costs real extra compute (a forward pass on every intersection every
tick, no skip-on-explore shortcut). Closed as inconclusive.

### TC-FedAvg (Topology-Conditioned FedAvg) — a bespoke design, added mid-queue per direct user
request for something purpose-built rather than an existing named method: **null at 6-seed rigor,
another few-seed mirage**

Motivated by the accumulated evidence that every AGGREGATION-strategy tweak tried in this project
came back null and federation-vs-no-federation makes no difference either — the problem was never
*how* weights get combined, but that the one shared function being averaged has no way to behave
differently for a 3-way vs. 5-way intersection. A small shared hypernetwork
(`NeighborAttentionQNetwork.topo_hyper`) maps a 4-dim structural descriptor (valid-action/-neighbor
fraction, mean/max hop distance — computable for any intersection, including one never trained on)
to a FiLM scale/shift on the fused representation. FedAvg itself is completely unchanged; only the
shared function being averaged gains topology-awareness. Zero-initialized so it's an exact identity
transform at the start of training.

Looked promising at 3 seeds (|diff|/SE 1.98 best-round, 2.18 mean) but **did not hold at 6**:
1.63 (best-round), **1.22 (mean, down from 2.18)**. 4 of 6 seeds favor it, 2 reverse it — the exact
same two seeds (11, 21) that also hurt item 23's recurrent variant, suggesting those two draws are
just harder for any new intervention on this roster rather than something specific to either idea.
`topo`'s own seed-to-seed standard deviation roughly doubled vs. baseline on both measures. Closed
as null — a real, carefully-verified idea that didn't survive scrutiny, same category as items 20/
21/23, reinforcing item 22 as the one confirmed exception rather than the start of an easy streak.

### Item 24 — Meta-learning aggregation (Reptile-style): **confirmed null, not just a stale
single-seed miss**

Already implemented in this codebase as `--fedavg_blend` (the exact Reptile damped-update rule:
`global_new = blend*aggregate + (1-blend)*global_old`) and already tried once (§72 pilot C) under
an older protocol — a clean single-seed miss. Re-tested at 6-seed rigor under the CURRENT protocol
(matching items 22/23/TC-FedAvg) rather than trusting that stale result: |diff|/SE 0.82 (best-round),
**0.07 (mean)** — essentially zero effect, mean reward differing by 0.17%. The theory (fine-tuning
is what works, so optimize the global model to be fine-tune-friendly) was sound but doesn't move
this task's numbers. Confirmed null, not deprioritized-on-a-hunch.

### Item 25 — Evolution strategies (gradient-free policy optimization): **inconclusive, not a
confirmed miss**

The most radical departure of the six: OpenAI-ES (`diagnostics/evolution_strategies.py`), no
Q-values, no TD-bootstrapping at all — a population of policies perturbed and selected by total
episode reward, reusing `DQNAgent` purely as a stateless policy container. First pilot (population=8,
5 generations, 1 seed): generation 3 genuinely beat random init (-6581 vs. -8586, ~23% better), but
generations 4-5 relapsed to worse-than-initial — the same "reachable but not retained" pattern
§51-53 characterized in gradient-based training, now showing up in an optimizer with no TD-
bootstrapping at all (interesting evidence the instability may be more about the task than about
Q-learning specifically). But 8 individuals × 1 episode/generation is genuinely under-powered by ES
standards (published implementations use hundreds of episodes/generation) — this screen can't
distinguish "doesn't work" from "too small to see it work." Closed as inconclusive, not confirmed
either way; a real test would need substantially more compute per generation.

**This closes the full six-item queue (20-25) plus the two ad-hoc additions (TC-FedAvg, item 24's
re-test) from this session.** Final scorecard: item 22 is the one confirmed, replicated win; items
20 and 24 are confirmed clean nulls; item 21 is real-but-non-deployable; items 23 and TC-FedAvg are
inconclusive-leaning-null at 6-seed rigor; item 25 is inconclusive due to being under-powered as
tested.

### Sequential (non-federated) curriculum training — CONFIRMED real, at a genuinely smaller
magnitude than originally reported (a bug inflated the first result)

**RESOLVED, 2026-09-07** (superseding the "major correction, pending" note that stood earlier): a
real RNG-isolation bug in `HoldoutEvaluator` (§88 in the investigation log) was silently reducing
effective seed independence across every seed's training trajectory in every single-process
diagnostic script this session. Fixed at the source. Full 6-seed re-verification under the fix
(§89) is now complete:

| | \|diff\|/SE vs. best-round | \|diff\|/SE vs. mean |
|---|---:|---:|
| Sequential FINAL checkpoint (6 seeds, fixed code) | **1.92** | **2.26** |
| Sequential BEST checkpoint (6 seeds, fixed code) | **3.87** | **4.22** |

5 of 6 seeds are clearly positive on the best-checkpoint measure (the 6th is flat, not a reversal).
**The effect is real and confirmed at the standard training budget** — genuinely smaller than the
original (buggy) claim of 5.71/6.32, but not remotely eliminated; the original 3-seed re-check
(1.88/2.04) that looked like it might be trending toward null turned out to be an unrepresentative
partial sample, not the final answer — the 3 additional seeds pulled the aggregate back to solid
significance. Separately, the escalation to a 3x larger training budget (§86) did NOT hold up at
all on re-verification (a complete reversal on its one seed) — that specific "bigger budget helps
even more" claim should be treated as unresolved/likely an artifact, not as evidence against the
standard-budget finding, which rests on a full, genuinely-independent 6-seed sample.

Instead of training every city in parallel and averaging weights each round (FedAvg), fully train
on city_1, then CONTINUE the same weights on city_4, then city_6 — one pass, no aggregation step at
all. Motivated directly by two standing findings: federation vs. no-federation makes no difference
(§49/50/64), while sequential adaptation (fine-tuning) is the one thing that's reliably worked
(§66-69). `diagnostics/sequential_training.py`, reusing 100% existing training/eval code — no new
architecture, no new algorithm.

At 6-seed rigor, matched total training volume against the existing parallel-FedAvg baseline:

| | \|diff\|/SE vs. baseline best-round | \|diff\|/SE vs. baseline mean |
|---|---:|---:|
| Sequential FINAL checkpoint | **3.48** | **4.14** |
| Sequential BEST checkpoint (chosen in hindsight, same convention as this project's own "best-round" stat) | **5.71** | **6.32** |

**Every single one of six seeds shows a positive best-checkpoint improvement (+23% to +80%), and
the final-checkpoint comparison — the practically deployable one — also clears the bar (5 of 6
positive).** Unlike every other lever tried this session, this one got MORE significant going from
3 to 6 seeds, not less — the opposite of the standing "few-seed mirage" pattern, and the clearest
sign yet of a real, robust effect rather than a lucky sample.

A real cost comes with it: catastrophic forgetting, measured directly — city_1's own in-distribution
performance drops 64% by the time city_4 and city_6 have also been trained (§85). The net holdout-
generalization gain survives this cost, but it means the mechanism is trading away some retained
competence on earlier cities for a much better final/peak policy overall — consistent with, and the
sharpest demonstration yet of, this document's standing diagnosis (§70/§71) that RETENTION, not
search or data, is the binding constraint: this result shows a much better policy is easily
reachable via a completely different, much simpler training procedure, it just isn't perfectly kept.

**§86 (3x budget) update: re-verified and did NOT hold up — a complete reversal on its one seed.**
The original claim (training city_1 alone for 30 episodes reaching -2453.37, +74.0%/+75.8% best-
checkpoint edge) was entirely an artifact of the RNG bug: under the fix, the same seed's trajectory
monotonically got WORSE through every training phase, ending as a statistical wash vs. baseline.
Single seed, so this doesn't prove the large-budget variant never helps, but the specific dramatic
claim should be treated as unresolved/likely spurious, not confirmed, pending its own multi-seed
replication (not yet done).

**Current honest status, FINAL (2026-09-07, all 6 seeds re-verified):** sequential training at the
STANDARD budget is a real, confirmed generalization benefit — |diff|/SE 1.92/2.26 (final checkpoint),
3.87/4.22 (best checkpoint), 5 of 6 seeds positive on the best-checkpoint measure. Genuinely smaller
than the original (buggy) claim, but a real, citable, methodologically clean result (a different
training paradigm, zero new architecture). The 3x-budget escalation is NOT part of this confirmed
claim and should not be cited. See `divergence_investigation.md` §89 for the full reconciliation of
all three re-verification data points.

---

## 5. Bottom line, right now

- The cross-topology generalization gap is real, large, and well-characterized.
- It is not an architecture problem (§3) — extensively, multi-seed tested.
- It is not (solely) a lock-in-via-stale-replay-data problem (item 20, this session).
- Test-time fine-tuning is a real, replicated mitigation, not a fix.
- Eval-time checkpoint combination (item 21) is a real, replicated effect but not a deployable
  mitigation — it only helps on volatile windows, and you can't tell which kind of window you're
  in without the eval sweep that would let you just pick the best round directly. Closed.
- Potential-based reward shaping using `max_pressure`'s own signal (item 22) is a real, confirmed,
  modest training-time improvement (|diff|/SE 2.5 on both measures, 6 seeds, no single-outlier
  dependence) — the first training-time lever in this whole document to hold up. Still a small
  effect relative to the baseline gap, not a fix.
- Recurrent policy (item 23) is inconclusive — a real direction split across seeds (4 favor it,
  2 against), neither measure clears the bar cleanly. Closed, not a confirmed finding.
- TC-FedAvg (bespoke topology-conditioning design) also closed as null at 6 seeds — promising at 3
  (both measures near/above the bar), evaporated at 6 (mean dropped from 2.18 to 1.22), the same
  4-favor/2-reverse split as item 23, on the same two seeds.
- Meta-learning aggregation (item 24, `--fedavg_blend`) also confirmed null at 6-seed rigor
  (|diff|/SE 0.82/0.07) — a sound theory that doesn't move this task's numbers.
- Evolution strategies (item 25) is inconclusive, not a confirmed miss — the one first-generation
  result that beat random init didn't persist, but the pilot (8 individuals × 1 episode/generation)
  is genuinely under-powered by ES standards, not a fair test of the paradigm yet.
- **All six originally-queued "genuinely different paradigm" items (20-25), plus two ad-hoc
  additions (TC-FedAvg, item 24's protocol re-test), are now done.** Five genuinely different
  training/aggregation-time mechanisms were tried (replay reset, recurrent memory, topology-
  conditioned FiLM, Reptile-style blending, evolution strategies); exactly one (item 22,
  potential-based reward shaping) confirmed as a real, replicated win.
- **Sequential (non-federated) curriculum training, tried after the queue closed, is CONFIRMED as a
  real second positive finding — at a genuinely smaller magnitude than first reported, after a real
  bug (§88) was found and fixed.** The original |diff|/SE 3.48-6.32 at "6-seed rigor" was generated
  under a since-fixed RNG-isolation bug that silently reduced effective seed independence. Full
  6-seed re-verification under the fix (§89): |diff|/SE 1.92/2.26 (final checkpoint), 3.87/4.22
  (best checkpoint) — genuinely smaller than the original claim, but real and solidly above this
  project's bar on 3 of 4 measures. 5 of 6 seeds positive on the best-checkpoint measure. The
  separate "3x budget makes it even better" escalation (§86) did NOT survive re-verification (a
  complete reversal on its one seed) and should be treated as unresolved, not part of this confirmed
  claim.
- **A bespoke new mechanism (Self-Anchoring Training with Confidence-Gated Reversion, §90), built
  directly for this project's own diagnosed retention bottleneck per direct user request, is CLOSED
  as inconclusive at 6-seed rigor** — |diff|/SE 1.53 (best-round) / 0.90 (mean), 4 of 6 seeds
  positive (one strong outlier +36.7%) but not clearing the bar. The calibration fix (default
  threshold too conservative to engage at all for some seeds → a more sensitive one that reliably
  fires 10-27 times per run) was a real, necessary correction, but even properly engaging, the
  aggregate effect falls short of significance — same honest treatment as items 23/TC-FedAvg. The
  same seed (11) underperformed across four different mechanisms this session (TC-FedAvg, recurrent
  policy, and both anchor-revert thresholds), suggesting a generically hard training draw for this
  roster rather than evidence against any one intervention.

**For a paper:** the framing now has TWO confirmed positive results, not one: potential-based reward
shaping (item 22, |diff|/SE ~2.5) and sequential curriculum training (this section, |diff|/SE
1.9-4.2 depending on measure) — both modest relative to the overall gap, both methodologically clean
and properly replicated at 6-seed rigor under corrected code. Combined with the characterized
cross-topology generalization gap, the confident-lock-in mechanism, the replicated fine-tuning
mitigation, and the thorough elimination of the architecture/aggregation axes as culprits, this is a
stronger paper than before this session: two genuine positive findings plus a rigorous, honest
account of what doesn't work and why — including transparently reporting and correcting a real bug
that briefly inflated one of the two positive results, which is itself evidence of the rigor this
project applies to its own claims.

---

## 6. Four candidate "significantly improve" ideas (2026-09-07, `divergence_investigation.md` §91),
   tested unsupervised per the user's explicit standing delegation before going to sleep

Four mechanisms genuinely different from everything above, all targeting the confident-lock-in/
retention bottleneck directly, pre-registered before results were known:

- **Conservative Q-Learning (CQL, discrete form, `--cql_weight`)** — penalizes Q-value
  overestimation directly (logsumexp over valid actions minus taken action's Q). **3-seed screen was
  the cleanest, most unanimous positive result of the whole session (2.35/2.80, all 3 seeds positive,
  no exceptions) — but weakened to 1.05/1.14 at 6 seeds, with seed 21 reversing hard (-14.4%/-4.1%).
  Closed as NOT CONFIRMED.** The strongest illustration yet that even a perfectly clean 3-seed screen
  is not sufficient — this project's 6-seed-before-confirming rule exists precisely for cases like
  this one.
- **Distributional RL (QR-DQN, `agents/qrdqn.py::QRDQNAgent`)** — learns a quantile distribution over
  returns per action instead of a scalar Q-value, structurally resisting collapse to an overconfident
  point estimate. **3-seed pilot came back negative, not merely unconfirmed** — 1.69/1.13 |diff|/SE,
  2 of 3 seeds worse than a matched no-q_entropy baseline on both measures, no positive trend to
  extend to 6 seeds. **Closed as a negative result at 3 seeds.** Unlike CQL/TC-FedAvg/anchor-revert
  (promising-then-null), this one never showed promise to begin with.
- **Proper MAML meta-learning aggregation** (`federated/maml.py` + `diagnostics/maml_fedavg.py`,
  genuine second-order meta-gradient via `torch.func.functional_call` + `create_graph=True`, NOT a
  repeat of item 24's already-null first-order `--fedavg_blend` proxy — verified distinct via a unit
  test checking the two gradients aren't numerically identical). **Single-seed pilot (seed 3) came back cleanly negative: monotonic decline every round** (random-init -8585.72 →
  R1 -8676.93 → R2 -9544.67 → R3 -10150.91) **converging into a fully stable confident lock-in — R3,
  R4, and R5 are byte-identical (std=0.00), the policy stopped changing at all.** **Closed as a
  negative result at n=1 seed**, matching QR-DQN's treatment — no positive trend anywhere to justify
  replicating on seeds 7/11 at several hours each. A real, unrelated correctness bug was also found
  and fixed in this script during a `/simplify` pass (its per-city gradient weighting used a constant
  instead of real sample counts, contradicting its own "FedAvg-style weighting" comment) — fixed, and
  confirmed not to change the qualitative verdict.
- **True ensemble of independently-trained seeds** (majority-vote across genuinely independent final
  checkpoints, distinct from item 21's same-run temporally-adjacent-checkpoint SWA) — finished after
  ~13 hours, partially usable. The individual-checkpoint and SWA-weight-average evals succeeded:
  SWA of the 6 independent checkpoints scored -9068.94, beating every individual checkpoint's mean
  (best individual: -9240.70) — a small but real-looking effect from combining independently-trained
  models. **The majority-vote ensemble itself crashed on all 30 episodes** — a real bug
  (`EnsemblePolicy.act()` in `diagnostics/swa_reeval.py` was missing the `ts_id` parameter
  `HoldoutEvaluator` always passes) that cascaded into a second crash (the evaluator's
  all-episodes-failed fallback dict was missing `std_reward` and other keys). Both bugs found and
  fixed; the majority-vote result itself still needs a re-run under the fix.

Running tally after all 4: **zero of the four pre-registered §91 candidates confirmed** —
CQL and QR-DQN closed not-confirmed/negative, MAML closed negative, and the ensemble's actual
majority-vote result never computed due to the bug above (only its SWA-average side-result is
usable). Consistent with this project's dominant pattern: most levers are null; the rare real ones
are modest, not transformative.

**Separately, the same night's Progressive Curriculum FedAvg (PCFT, §87, not one of the four
pre-registered candidates) is now CONFIRMED at full 6-seed rigor — |diff|/SE 2.42 (final round),
3.42 (best-ever round), 2.70 (mean), all three STRONGER than the initial 3-seed screen (2.40/3.03/
2.39), the opposite of CQL's and TC-FedAvg's fade.** 5 of 6 seeds positive on every measure. **This
makes PCFT the third confirmed training-time/curriculum result of the whole investigation** (with
item 22 and sequential training), and by two of three measures the strongest of the three. The
budget/fine-tuning-embedding confound (PCFT's curriculum includes per-city focus/fine-tune steps
already known to help, so this may not isolate curriculum ORDERING as the active ingredient) and the
continued enormous within-run volatility are both still open, unresolved caveats — not retracted by
the confirmation. See `divergence_investigation.md` §87 for full seed-by-seed numbers.

**Two more architecture-level ideas, proposed and tested live with the user (not pre-registered,
not part of §91) immediately after: `--bounded_q` and `--trunk_lr_scale`, both targeting the
confident-lock-in RETENTION mechanism directly (as opposed to representation capacity, which three
prior attempts — the base architecture, TC-FedAvg, and §71's wider roster — already failed to fix).**
`--bounded_q` (caps the Q-head's cross-action spread via a hard tanh ceiling, architecturally rather
than as a loss-level preference like `--cql_weight`/`--q_entropy_weight`) came back a **clean null**
at 3 seeds (|diff|/SE 0.22/0.07) — one seed showed a promising lock-in-free trend through round 4
that then relapsed at the final round, netting out flat. `--trunk_lr_scale` (differential learning
rate: the representation-building trunk learns slower than the Q-head, so a round's gradients can't
fully overwrite it before consolidating) came back a **real negative result**, unanimous across all
3 seeds (|diff|/SE 2.15/1.99, all three seeds worse, one by -13%) — likely a starvation effect, since
the trunk still needs to adapt quickly early in training when it has no good representation yet to
protect. Both closed after one 3-seed pilot each, per direct user instruction to move on to the next
idea rather than tune either flag further. See `divergence_investigation.md` §92.

**A third architecture-level idea, `--lora_adapter`, was tried immediately after and also came back
null.** Rather than restricting the trunk's learning (which starved it), this ADDS a small
zero-initialized low-rank residual correction on top of the fully-normally-trained trunk — pure
extra capacity, not a reallocation. 3-seed pilot: |diff|/SE 0.38 (best-ever round) / 0.28 (mean),
indistinguishable from noise. One seed hit a genuine standout round (-8269.51) that didn't hold,
same "reachable, not retained" pattern as everywhere else. **Along the way, the first real-SUMO
smoke test caught a genuine pre-existing wiring bug** (not just confirmed the new mechanism): the
`--parallel` path's `global_model` template was missing several flags entirely (`cql_weight`,
`anchor_revert`, `bounded_q`, `trunk_lr_scale`, and now `lora_adapter`) — harmless for all the
others since none change the network's parameter set, but `lora_adapter` does (adds `lora_down`/
`lora_up`), so its state_dict was missing those keys and crashed every worker's strict
`load_state_dict` on round 1. Fixed by threading the full flag set through that call site, not just
the two new ones — the exact "flag silently doesn't reach where it needs to" bug class this
project's `tests/test_flag_wiring.py` exists to catch.

**Final tally, all three architecture-level retention ideas tried this session: `--bounded_q` null
(0.22/0.07), `--trunk_lr_scale` negative (2.15/1.99), `--lora_adapter` null (0.38/0.28).** None held
up. Combined with the loss-level attempts (q_entropy_weight, CQL, distributional RL) and the
post-hoc attempt (self-anchoring) tried earlier this session, the confident-lock-in/retention
mechanism has now resisted every lever aimed at it directly, at every level of the stack tried so
far — only fine-tuning on real target-city data (§66-70), which sidesteps the zero-shot requirement
rather than fixing it, reliably helps. See `divergence_investigation.md` §92 for full numbers.

---

## 7. THE ACTION REPRESENTATION WAS THE PROBLEM (§95-§100, 2026-09-08/09) — supersedes the framing above

**The finding.** The shared Q-head indexed actions by *position* (row k = phase k). Row k is
therefore trained on contradictory targets across cities — index 1 is a protected left turn on
arterial4x4/grid4x4 and a through movement on ingolstadt7/cologne3 — so a fully-trained row still
encodes nothing transferable. `--phase_relational` replaces positional indexing with a
phase-relational readout.

**Result: phase-relational beats `max_pressure` zero-shot on an unseen holdout topology** — the
first time anything in this project has done that. Confirmed at 6 seeds on four distinct
configurations, 6/6 seeds every time:

| config | section | phase-relational | indexed | \|diff\|/SE (best/final/mean) | drop-1 floor |
|---|---|---:|---:|---|---:|
| grid4x4, 8-phase, light demand | §96 | -0.16 | -9296.84 | 6-seed confirmed | — |
| grid4x4 at 3x demand | §97 | -0.32 | -17952.96 | 46.31 / 50.59 / 64.80 | 38.62 |
| 3x3Grid2lanes 4-phase, no untrained rows | §98 | -1.41 | -9028.88 | 25.79 / 24.29 / 96.67 | 20.25 |
| RESCO-exact timing, standard roster | §99 | -0.16 | -9411.88 | 47.23 / 39.34 / 57.49 | 32.77 |
| **fully RESCO-exact roster** | **§100** | **-0.13** | **-8525.72** | **23.63 / 30.93 / 53.82** | **20.24** |

**§98 is the control that makes this a representation result rather than a bug fix.** §95b had
found that on unseen-topology rosters, 37.5% of the holdout's action space is scored by Q-head rows
never trained by any city. §98 removed that defect entirely — a holdout whose phase count does not
exceed the training maximum, so every row it can use is fully trained — and **the indexed head
still gridlocked, at -9028.88, indistinguishable from its score with dead rows.** The defect was not
the cause. The representation is.

**Consequence for sections 3-6 of this document.** The ~20 interventions of §73-§95 (algorithm
swaps, capacity, aggregation strategies, curricula, retention levers) were all evaluated on a
readout that cannot express a transferable policy regardless of how it is optimised. **Their null
results are floor effects, not evidence about those mechanisms.** Re-running that corpus on the
phase-relational head is open; PCFT (§87, confirmed but modest) is the first candidate.

**§99: a measurement error found in the same stretch, worth carrying.** This project had never run
RESCO's scenario configuration — wrong route files, wrong evaluation windows, and `yellow_time=2`
against RESCO's 3, i.e. 50% more usable green per phase change than the benchmark it was being
compared to. §59's "~4.4x behind published IDQN" claim is **retracted**. Every *relative* result in
the document is unaffected (both arms always shared the configuration); only absolute numbers quoted
against external work were invalid. `environments_rescofull/` (§100) fixes all three mismatches, and
the headline result survives all of them.

**Standing caveat: the reward numbers above are this project's internal `diff-waiting-time` unit and
are not comparable to published figures.** Only Avg. Delay and Avg. Trip Time reconcile with RESCO;
this project's `wait` and `queue` do not (§99's metric caveat). The in-distribution
literature-metric comparison against RESCO's Cologne/Ingolstadt numbers — the first like-for-like
external comparison this project will have — was running as of 2026-09-09 and is **pending**.
