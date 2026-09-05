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

### Item 21 — SWA-style checkpoint averaging/ensembling at eval time: **the first genuinely
promising result of the entire investigation, confirmation in progress**

Every prior lever changed something about *training*. This is the first eval-time-only lever
tested: average (true SWA) or majority-vote-ensemble several consecutive round checkpoints instead
of picking one. Built `diagnostics/swa_reeval.py`.

**First result** (5 consecutive checkpoints from an existing run, 10 episodes each):

| | reward |
|---|---:|
| Individual checkpoints | -6223, -4737, **-4229 (best)**, -5699, -6968 |
| SWA weight-average | **-4263** |
| Majority-vote ensemble | **-4324** |

**Both combination methods landed almost exactly at the single best individual checkpoint's
performance, without needing to know in advance which round was best** — the actual practical
value proposition, since in real deployment you can't pick the best round ahead of time.

**Not yet confirmed**, per the hard-learned lesson in §3 above: this is one window (one specific
5-round stretch, one run), 10 episodes only. Two confirmatory runs are in progress as of this
writeup: the same window at 30-episode rigor, and an independent window (different run, seed, and
architecture) to test generalization. **Do not cite the numbers above as final** — check back for
the confirmed result before relying on this.

**Two real bugs found and fixed while building this (both now committed):**
1. §76's attention-stacking refactor had silently broken loading of *every* checkpoint saved before
   that commit (a state-dict key rename, not a shape change) — would have hit `--resume` and every
   other checkpoint-loading script, not just this new one. Fixed with a backward-compatible
   key-remapping shim.
2. The shared `infer_arch_from_checkpoint` helper didn't detect this session's newer architecture
   knobs (`encoder_depth`, `n_attn_layers`), so it silently assumed old defaults — fine for old
   checkpoints, a shape-mismatch crash for anything trained with those flags. Fixed to detect both
   directly from the state dict.

### Items 22-25 — queued, not yet started

- **22. Potential-based reward shaping using `max_pressure`'s own formula.** Unlike the earlier ad
  hoc shaping attempt (§44, arbitrary weights, inconclusive), potential-based shaping (Ng, Harada &
  Russell 1999) is mathematically guaranteed not to change the optimal policy — isolates a
  learning-*dynamics* effect from a different-optimal-policy effect.
- **23. Recurrent policy (LSTM/GRU over recent ticks).** Every architecture tried in §73-76 was
  still a purely reactive function of one tick's snapshot. Temporal memory is a genuinely different
  axis (time, not capacity).
- **24. Meta-learning aggregation (Reptile/MAML-style) instead of plain FedAvg.** The one lever
  that's actually worked (fine-tuning) succeeds *despite* FedAvg optimizing for the wrong thing (a
  good fixed policy on average, not a good starting point for adaptation) — this targets that
  mismatch directly.
- **25. Evolution strategies (gradient-free policy optimization).** The most radical departure: no
  Q-values, no TD-bootstrapping at all — sidesteps the entire diagnosed confident-lock-in mechanism
  class rather than patching around it.

---

## 5. Bottom line, right now

- The cross-topology generalization gap is real, large, and well-characterized.
- It is not an architecture problem (§3) — extensively, multi-seed tested.
- It is not (solely) a lock-in-via-stale-replay-data problem (item 20, this session).
- Test-time fine-tuning is a real, replicated mitigation, not a fix.
- The first result suggesting a further, practical mitigation is eval-time checkpoint combination
  (item 21) — promising but **not yet confirmed**.
- Four more genuinely different mechanism hypotheses are queued and will be tested with the same
  rigor (3+ seeds from the start, or multiple independent windows for eval-time tricks) before any
  of them gets cited as a real finding.

**For a paper:** the defensible framing remains what §58-61 of the investigation log established —
not "federated DQN traffic control fails," but "a characterized, budget-resistant cross-topology
generalization gap, with a well-understood partial mechanism (confident lock-in), a replicated
test-time mitigation (fine-tuning), a thorough negative result on the architecture axis, and an
ongoing, methodologically rigorous search for a training-dynamics-level fix." Same category of
contribution as the RESCO benchmark paper itself.
