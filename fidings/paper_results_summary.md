# Paper-Ready Results Summary

**Status of this document, rewritten 2026-09-06:** the previous version of this file dated from
2026-08-15 and predates the entire §43-onward reframing of this project's central story (see
`CLAUDE.md`'s "RESUME HERE" section and `divergence_investigation.md` §43-86). Its old section A's
headline claim ("2-city trained policy beats both rule-based baselines") is a **confirmed-superseded
artifact** — §43 later found that claim was an in-distribution evaluation, not a true cross-topology
holdout result (the 2-city roster's action_dim was too narrow to reach the real holdout, so
evaluation silently fell back to one of the roster's own training cities). That old content is kept
below, in section F, with its correction notice intact, purely as a historical record — **do not
cite section F in a paper**. Everything above it (sections A-E) reflects the current, corrected
state of knowledge and is safe to draft from directly. All numbers are re-derivable from the cited
`divergence_investigation.md` sections; if anything here ever disagrees with that document, the
investigation log wins.

---

## Novelty / contribution framing

Four distinct layers of contribution, useful for structuring a paper's framing (introduction /
related work) and for keeping straight what's actually new here versus what's supporting
infrastructure:

1. **Architecture (established before the 2026-09-06 session):** a genuinely topology-agnostic
   "foundation model" for traffic-signal control — one shared DQN controls intersections of
   arbitrarily different topologies (3-way/4-way/5-way, differing neighbor counts) purely through
   `action_mask`/`neighbor_mask`, no per-topology code paths, masked neighbor-attention handling any
   number of neighbors. Federated learning is applied across genuinely heterogeneous topologies, not
   just non-iid data on the same topology — an unusual combination in the FL literature, and
   different from how most traffic-signal RL work (including the RESCO benchmark this project's city
   configs are drawn from) trains per-network models rather than one model shared across
   structurally different networks.
2. **Mechanistic diagnostics (established before 2026-09-06, real contributions in their own
   right):** the "confident lock-in" failure mode, rigorously demonstrated via byte-identical
   rewards across 30 different random seeds and tied directly to Q-value confidence
   (`divergence_investigation.md` §32-34); separating training-budget/eval-protocol confounds from
   the genuine cross-topology generalization gap (§58-61) — showing most of the originally-reported
   "3-4 orders of magnitude worse than baselines" gap was a confound, not a pure algorithm failure;
   the counterintuitive finding that a randomly-initialized network fine-tuned on the target city
   *beats* a federated-pretrained one (§70).
3. **The 2026-09-06 session's central empirical discovery — the strongest, most novel, most
   citable result in the project as of this writing:** sequential (non-federated) curriculum
   training substantially and robustly beats parallel FedAvg on cross-topology holdout
   generalization, confirmed at 6-seed rigor and replicated at a second, 3x-larger training budget,
   with a real, precisely measured catastrophic-forgetting cost that does not erase the net gain.
   See section A below. This is the section to lead a paper's results with if it holds up under
   further validation.
4. **Bespoke method designs, built from scratch for this exact problem rather than adapted from an
   existing named method (2026-09-06 session, per direct user request for genuinely new designs):**
   TC-FedAvg (a shared hypernetwork conditions the network's internal computation on a structural
   descriptor computable for any intersection, including unseen ones — architecturally novel, did
   not survive 6-seed confirmation, see section C) and Progressive Curriculum FedAvg (simplest-to-
   complex city ordering + phased-in federation via focus-then-merge, synthesizing findings 2 and 3
   above — in progress, see section D).

**Suggested paper framing given all of the above:** not "federated DQN traffic control fails," but
"a characterized cross-topology generalization gap; a well-understood partial mechanism (confident
lock-in); a thorough, rigorous elimination of the architecture and training-dynamics axes as
culprits (six mechanisms tried in the 2026-09-06 session alone, one confirmed modest win); and a
genuinely surprising positive result — abandoning federated averaging for simple sequential
curriculum training substantially improves cross-topology generalization at matched compute, at the
cost of measurable, characterized forgetting." Same category of contribution as the RESCO benchmark
paper itself (whose own headline finding is also "published methods underperform simple baselines in
realistic scenarios"), strengthened by finding 3 above turning it from a purely negative result into
one with a genuine positive, actionable core.

---

## A. Sequential (non-federated) curriculum training beats parallel FedAvg — the project's
    strongest confirmed result

**The claim:** instead of training every city in parallel and averaging weights each round
(FedAvg), fully training on one city, then continuing the SAME weights on the next city, then the
next (one pass, no aggregation step at all) produces a policy that generalizes to the true holdout
city dramatically better than parallel FedAvg, at matched total training volume — confirmed at
6-seed rigor and replicated at a second, larger budget.

**Source:** `divergence_investigation.md` §85 (6-seed confirmation, `environments_c1_4_6`,
`city_1 -> city_4 -> city_6` order, 10 episodes/city) and §86 (3x-budget replication, seed 3, 30
episodes/city). Implementation: `diagnostics/sequential_training.py`.

### The numbers (§85, 6 seeds: 3/7/11/17/21/25, 10 episodes/city)

| | vs. baseline best-round | vs. baseline mean |
|---|---:|---:|
| Sequential FINAL checkpoint | \|diff\|/SE = **3.48** | \|diff\|/SE = **4.14** |
| Sequential BEST checkpoint (chosen in hindsight, same convention as "best-round" below) | \|diff\|/SE = **5.71** | \|diff\|/SE = **6.32** |

Per-seed, best-checkpoint vs. baseline's own best-round: +23.3%, +49.6%, +80.0%, +44.4%, +25.5%,
+35.5% — **every single seed positive, no exceptions.** Unlike every other lever tried in the
2026-09-06 session, this result got MORE significant going from 3 to 6 seeds, not less (3-seed
screen: 1.60/1.85 final, 4.20/4.49 best-checkpoint) — the opposite of the standing "few-seed mirage"
pattern this document has repeatedly warned about, and the clearest sign of a real, robust effect.

### The numbers (§86, 3x budget, seed 3, 30 episodes/city)

| Stage | Holdout mean_reward |
|---|---:|
| Random init | -8585.72 |
| After city_1 alone (30 episodes) | **-2453.37 — best result in the entire investigation outside rule-based baselines** |
| After city_1+city_4 | -6207.73 |
| Final (city_1+city_4+city_6) | -7642.36 |

Matched parallel-FedAvg baseline at the same budget (`--rounds 15 --local_episodes 2`) actually got
WORSE with more rounds: best-round -9450.14, mean -10143.20 (vs. -9390.30/-9687.10 at the smaller
budget) — widening the gap from both directions. Sequential's best checkpoint beats it by **+74.0%**
(best-round) / **+75.8%** (mean); the final checkpoint, after relapsing, still beats it by +19.1%/
+24.7%.

### Catastrophic forgetting, measured directly (§85, seed 3)

| City | Right after its own training | Final (all cities trained) | Change |
|---|---:|---:|---:|
| city_1 (trained first) | -2053.65 | -3369.62 | **64% worse** |
| city_4 (trained second) | -389.52 | -417.34 | ~7% worse |
| city_6 (trained last) | -3.01 | -3.01 | unchanged |

Real, substantial forgetting, worse the earlier a city was trained — the expected continual-learning
signature. The net holdout-generalization gain survives this cost, but it is a genuine tradeoff, not
a free win.

### Reading

Sequential training reliably discovers a much better region of weight space than parallel FedAvg
ever reaches at matched total training volume, and — while it does not perfectly retain that peak
through the end of training — retains enough of it that even the FINAL checkpoint significantly
beats FedAvg. Combined with the forgetting measurement, this is the sharpest demonstration yet of
this project's standing diagnosis (`divergence_investigation.md` §70/§71): the binding constraint on
this task is the algorithm's failure to RETAIN what it learns, not its ability to find a good policy
in the first place, which turns out to be comparatively easy — a much better-than-anything-federated
policy is reachable by literally just training on one city for a while.

### What would make this section fully paper-ready

1. §86's larger-budget result is currently single-seed — replicate at 3+ seeds at that budget for
   the same rigor already applied at the smaller one.
2. Test whether holdout-monitored checkpoint selection (picking the best-so-far checkpoint during
   sequential training, exactly the "best-round" convention already used throughout this document)
   turns this into a directly deployable recipe, independent of ever fixing the underlying retention
   problem — flagged as the natural next experiment in §85/86, not yet run.
3. Section D below (Progressive Curriculum FedAvg) is a direct attempt to get this method's search
   advantage while reducing its forgetting cost via phased-in federation — pending results.

---

## B. Potential-based reward shaping using `max_pressure`'s own signal — confirmed, modest
    training-time improvement

**The claim:** adding a potential-based reward shaping term (Ng, Harada & Russell 1999 — provably
does not change the optimal policy, unlike ad hoc reward shaping) using `max_pressure`'s own
pressure signal as the potential function gives a small but real, statistically confirmed
improvement over plain FedAvg.

**Source:** `divergence_investigation.md` §80. Implementation: `--potential_shaping_weight`/
`--potential_shaping_gamma` in `experiments/federated_training.py`, `RewardShapingWrapper` in
`environments/federated_env.py`.

### The numbers (6 seeds: 3/7/11/17/21/25, `environments_c1_4_6`, 5 rounds x 2 episodes)

| | best-round (6-seed avg) | mean (6-seed avg) |
|---|---:|---:|
| baseline (`q_entropy_weight=0.05` only) | -9296.84 | -9675.20 |
| `+ potential_shaping_weight=0.1` | -8602.09 | -9277.03 |
| \|diff\|/SE | **2.53** | **2.49** |

5 of 6 seeds favor it, the 6th a near-exact tie (not a reversal). ~4-7.5% improvement — real and
replicated, but modest relative to the overall gap to rule-based baselines. The first training-time
lever in the entire item-2X campaign to hold up (four others — replay-buffer reset, recurrent
policy, Reptile-style aggregation blending, evolution strategies — all came back null or
inconclusive; see `divergence_investigation.md` §78/§81/§83/§84 for the full negative-result
tally).

### Reading

Since `Phi(s)` tracks exactly the signal `max_pressure` itself greedily maximizes, this shaping term
teaches the DQN a denser, better-aligned per-tick learning signal without changing what the optimal
policy actually is — a purely dynamics-side intervention that moved the needle, consistent with the
project's standing diagnosis that the core problem is a learning-dynamics/retention issue, not
insufficient data or capacity.

---

## C. TC-FedAvg (Topology-Conditioned FedAvg) — a real, well-verified negative result

**The claim tested:** a small shared hypernetwork maps a 4-dim structural descriptor (valid-action/
-neighbor fraction, mean/max hop distance — computable for any intersection including one never
trained on) to a FiLM scale/shift on the network's internal representation, so the ONE shared
function FedAvg averages gains explicit topology-awareness, without changing the aggregation
mechanism itself.

**Source:** `divergence_investigation.md` §82. Implementation: `agents/networks.py`'s
`topology_conditioned` flag, `agents/topology_conditioned_dqn.py`.

### The numbers (6 seeds, same protocol as section B)

| | best-round | mean |
|---|---:|---:|
| \|diff\|/SE at 3 seeds | 1.98 | 2.18 |
| \|diff\|/SE at 6 seeds | **1.63** | **1.22** |

Promising at 3 seeds, evaporated at 6 — the standard "few-seed mirage" pattern this document has
repeatedly documented. **Do not cite as a working method.** Useful negative result: it rules out
"the network just needs an explicit topology signal at the representation level" as a fix, given a
mathematically clean, well-motivated, carefully verified implementation of that exact idea.

---

## D. Progressive Curriculum FedAvg — in progress, not yet resolved

**The design:** a bespoke synthesis of sections A and B/§66-70's fine-tuning mechanism, per direct
user request. Orders training cities from simplest to most complex by intersection count
(`city_4` (3) -> `city_6` (7) -> `city_1` (16) — the opposite of section A's incidental ordering,
which happened to start with the most complex city). Warms up solo on the simplest city, then for
each new city gives it a short focus fine-tune phase (starting from the current shared weights, the
exact §66-70 mechanism) before folding it into genuine multi-city FedAvg for several rounds, with
already-active cities keeping persistent replay buffers/optimizer state across rounds (matching the
real pipeline's warm-start convention) to test whether phased-in federation can retain sequential
training's search advantage while reducing its forgetting cost.

**Source:** `divergence_investigation.md` §87. Implementation:
`diagnostics/progressive_curriculum_fedavg.py`. Verified via a real SUMO smoke run; a 3-seed pilot
(seeds 3/7/11, `--warmup_episodes 10 --focus_episodes 5 --fedavg_rounds 3 --local_episodes 2`) is
running as of this writing. **Do not cite any result for this section until it lands and is
reported in `divergence_investigation.md` §87's follow-up.**

---

## E. Everything else tried in the 2026-09-06 session's item-2X campaign: confirmed nulls or
    inconclusive, not deployable

For completeness — these were tried with real rigor and should be citable as "ruled out," not
omitted, since the thoroughness of elimination is itself part of this project's contribution
(framing point 2 above):

| Item | Result | \|diff\|/SE | Source |
|---|---|---:|---|
| Replay-buffer reset on detected lock-in | Confirmed null | 0.30 / 0.38 | §78 |
| SWA/ensemble checkpoint combination (eval-time) | Real but not deployable — only helps on volatile training windows, can't tell in advance | n/a | §79 |
| Recurrent policy (GRU hidden state) | Inconclusive — seeds disagree on direction | 1.63 / 1.98 | §81 |
| Meta-learning aggregation (`--fedavg_blend`, Reptile-style) | Confirmed null | 0.82 / 0.07 | §83 |
| Evolution strategies (OpenAI-ES) | Inconclusive — genuinely under-powered pilot (8 individuals x 1 episode/generation), not a fair test | n/a | §84 |

---

## F. HISTORICAL, SUPERSEDED — pre-2026-09-06 content, do not cite

**Everything below this line predates the §43 correction and is kept only as a historical record of
what this document used to claim.** Section A below (the old section A) is the specific claim §43
found to be an artifact — see `CLAUDE.md`'s "RESUME HERE" section for the full correction. Sections
B and C below (old numbering) describe Phase 2 aggregation-strategy comparisons and a reward-shaping
candidate fix that were both superseded by the much more thorough item-2X campaign and the §58-61
budget/protocol reframing. None of this should appear in a paper draft.

<details>
<summary>Old section A (superseded, click to expand)</summary>

### [SUPERSEDED] 2-city trained policy beats both rule-based baselines on best-achieved performance

**Correction, 2026-09-06:** this claim was confirmed at full 5-seed rigor to be an artifact of
evaluating in-distribution, not on a true holdout — see `divergence_investigation.md` §43. The
2-city roster's action_dim was too narrow to reach the real `city_5_holdout`, so evaluation silently
fell back to `city_1`, one of the roster's own training cities. At true holdout, the trained DQN
loses decisively to both rule-based baselines at every roster size tested. Do not cite the numbers
that originally appeared in this section.

</details>

<details>
<summary>Old sections B and C (superseded, click to expand)</summary>

### [SUPERSEDED] Phase 2 aggregation-strategy comparison

Superseded by the much more thorough aggregation-strategy elimination in the 2026-09-06 session
(section E above, and `divergence_investigation.md`'s broader aggregation-strategy history) — the
Phase 2 sweep's conclusion (no strategy clears plain FedAvg) still holds directionally, but the
numbers below are from an earlier, less-corrected protocol and should not be cited directly. See
`divergence_investigation.md` for the current, correct comparisons.

### [SUPERSEDED] Task #7 reward-shaping candidate fix

The ad hoc `--reward_shaping_wait_weight`/`--reward_shaping_stopped_weight` mechanism described here
was tested once (§44) and found inconclusive; it has been superseded as a research direction by the
POTENTIAL-BASED reward shaping in section B above, which is mathematically guaranteed not to change
the optimal policy (unlike this ad hoc version) and IS confirmed to work. Cite section B instead.

</details>
