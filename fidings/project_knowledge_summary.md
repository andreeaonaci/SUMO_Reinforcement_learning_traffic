# Everything we know — master summary (as of 2026-09-05)

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

### Sequential (non-federated) curriculum training — the strongest confirmed finding of the entire
investigation, added per direct user request after the item-2X queue closed

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

**Escalation to 3x training budget (§86) confirms and sharpens this further — now the single most
important empirical finding of the whole project.** Training city_1 ALONE for 30 episodes reaches a
holdout reward of -2453.37 — better than any other result anywhere in this document outside
rule-based baselines. The matched parallel-FedAvg baseline at the same budget got WORSE with more
rounds (best -9450.14 vs. the smaller budget's -9390.30), widening the gap from both directions:
sequential's best checkpoint beats it by **+74.0%** (best-round) / **+75.8%** (mean); even the final
checkpoint, after relapsing, still beats it by +19-25%. Single seed at this larger budget, but
combined with §85's 6-seed confirmation at the smaller one, this is a well-triangulated result
across two budgets, not a one-off. It also sharpens the "search vs. retention" diagnosis into its
cleanest form yet: a policy far better than anything federated training has ever produced is
trivially reachable by just training on ONE city for a while — the entire remaining problem is
keeping it.

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
- **Sequential (non-federated) curriculum training, tried after the queue closed, is the strongest
  confirmed result in the whole investigation** — |diff|/SE 3.48-6.32 at 6-seed rigor (vs. item 22's
  2.5), every single seed agreeing in direction on the best-checkpoint measure, and — unlike
  everything else this session — the effect got MORE significant with more seeds, not less. Comes
  with a real, measured catastrophic-forgetting cost, but the net holdout gain survives it.

**For a paper:** the defensible framing has shifted meaningfully with this last result. Instead of
"federated DQN traffic control fails," the strongest available framing is now: "a characterized
cross-topology generalization gap; a mechanism (confident lock-in); a replicated test-time
mitigation (fine-tuning); a thorough, rigorous elimination of the architecture and training-dynamics
axes as culprits (five mechanisms tried, one confirmed modest win); and a genuinely surprising
positive result — that abandoning federated averaging in favor of simple sequential curriculum
training substantially improves cross-topology generalization at matched compute, at the cost of
measurable forgetting on earlier-seen cities." That last piece, if it continues to hold at larger
scale, is a real, citable, non-obvious empirical finding in its own right, not just a negative
result — potentially the headline finding of the whole project.
