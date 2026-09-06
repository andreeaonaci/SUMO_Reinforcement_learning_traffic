# Paper-Ready Results Summary

**Status of this document, rewritten 2026-09-06:** the previous version of this file dated from
2026-08-15 and predates the entire §43-onward reframing of this project's central story (see
`CLAUDE.md`'s "RESUME HERE" section and `divergence_investigation.md` §43-86). Its old section A's
headline claim ("2-city trained policy beats both rule-based baselines") is a **confirmed-superseded
artifact** — §43 later found that claim was an in-distribution evaluation, not a true cross-topology
holdout result (the 2-city roster's action_dim was too narrow to reach the real holdout, so
evaluation silently fell back to one of the roster's own training cities). That old content is kept
below, in section G, with its correction notice intact, purely as a historical record — **do not
cite section G in a paper**. Everything above it (sections A-F) reflects the current, corrected
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
3. **A second confirmed positive result, at a smaller magnitude than first measured:** sequential
   (non-federated) curriculum training beats parallel FedAvg on cross-topology holdout
   generalization. A real bug in the evaluation harness (a global-RNG leak between evaluation and
   subsequent training, present in every single-process diagnostic script this session — see
   `divergence_investigation.md` §88) was found and fixed; the full 6-seed re-verification under the
   fix (§89) confirms the effect survives at |diff|/SE 1.92-4.22 depending on measure (down from an
   inflated 3.48-6.32 originally) — real, replicated, methodologically clean, genuinely smaller. See
   section A. A separate 3x-budget escalation did not survive re-verification and is not part of
   this claim.
4. **Bespoke method designs, built from scratch for this exact problem rather than adapted from an
   existing named method (2026-09-06 session, per direct user request for genuinely new designs):**
   TC-FedAvg (a shared hypernetwork conditions the network's internal computation on a structural
   descriptor computable for any intersection, including unseen ones — architecturally novel, did
   not survive 6-seed confirmation, see section C), Progressive Curriculum FedAvg (simplest-to-
   complex city ordering + phased-in federation via focus-then-merge, synthesizing findings 2 and 3
   above — in progress, see section D), and Self-Anchoring Training with Confidence-Gated Reversion
   (an active, in-the-loop mechanism that snapshots a round's starting weights and pulls training
   back toward them when a validated internal signal — Q-gap growth — indicates drift toward the
   project's diagnosed confident-lock-in failure mode; closed as inconclusive at 6-seed rigor, see
   section E).

**Suggested paper framing given all of the above:** two confirmed positive results — potential-
based reward shaping (finding 2 above, |diff|/SE ~2.5) and sequential curriculum training (finding 3,
|diff|/SE 1.9-4.2) — sitting inside a characterized cross-topology generalization gap, a well-
understood partial mechanism (confident lock-in), a replicated test-time mitigation (fine-tuning),
and a thorough, rigorous elimination of the architecture/aggregation axes as culprits (six further
mechanisms tried in the 2026-09-06 session, all null or inconclusive). Same category of contribution
as the RESCO benchmark paper itself (whose own headline finding is also "published methods
underperform simple baselines in realistic scenarios"), strengthened by two genuine positive results
instead of zero, and by the transparent finding-and-fixing of a real bug that briefly inflated one of
them — itself evidence of the rigor applied to every claim in this document.

---

## A. Sequential (non-federated) curriculum training vs. parallel FedAvg — CONFIRMED real, at a
    genuinely smaller magnitude than first reported

**Provenance note (safe to skip if you just want the numbers):** the first version of this section
reported |diff|/SE 5.71-6.32 at "6-seed rigor." A real RNG-isolation bug (`divergence_investigation.md`
§88 -- `HoldoutEvaluator`'s per-episode determinism reset was leaking into subsequent training in
every single-process diagnostic script this session) was found and fixed, and the full 6-seed
re-verification under the fix (§89) is what's reported below. The bug inflated the original numbers;
the effect itself is confirmed real, just smaller. The separate "3x training budget makes it even
better" escalation (§86) did NOT survive re-verification (a complete reversal on its one seed) and is
NOT part of this claim -- see the caveat at the end of this section.

**The claim:** instead of training every city in parallel and averaging weights each round
(FedAvg), fully training on one city, then continuing the SAME weights on the next city, then the
next (one pass, no aggregation step at all) produces a policy that generalizes to the true holdout
city better than parallel FedAvg, at matched total training volume -- confirmed at 6-seed rigor.

**Source:** `divergence_investigation.md` §85 (original result), §88 (bug found and fixed), §89
(6-seed re-verification under the fix -- this section's numbers). `city_1 -> city_4 -> city_6`
order, `environments_c1_4_6`, 10 episodes/city. Implementation: `diagnostics/sequential_training.py`.

### The numbers (6 seeds: 3/7/11/17/21/25, 10 episodes/city, re-verified under the RNG-isolation fix)

| | vs. baseline best-round | vs. baseline mean |
|---|---:|---:|
| Sequential FINAL checkpoint | \|diff\|/SE = **1.92** | \|diff\|/SE = **2.26** |
| Sequential BEST checkpoint (chosen in hindsight, same convention as "best-round" elsewhere in this document) | \|diff\|/SE = **3.87** | \|diff\|/SE = **4.22** |

Per-seed, best-checkpoint vs. baseline's own best-round: +51.7%, -0.7%, +75.0%, +75.1%, +20.9%,
+64.2% -- **5 of 6 seeds clearly positive, the 6th essentially flat (not a reversal).**

### Catastrophic forgetting, measured directly (seed 3)

| City | Right after its own training | Final (all cities trained) | Change |
|---|---:|---:|---:|
| city_1 (trained first) | -2053.65 | -3369.62 | **64% worse** |
| city_4 (trained second) | -389.52 | -417.34 | ~7% worse |
| city_6 (trained last) | -3.01 | -3.01 | unchanged |

Real, substantial forgetting, worse the earlier a city was trained -- the expected continual-learning
signature. The net holdout-generalization gain survives this cost, but it is a genuine tradeoff, not
a free win.

### Reading

Sequential training reliably discovers a better region of weight space than parallel FedAvg reaches
at matched total training volume, and retains enough of it that the FINAL checkpoint significantly
beats FedAvg too, not just an interim one. Combined with the forgetting measurement, this supports
this project's standing diagnosis (`divergence_investigation.md` §70/§71): the binding constraint on
this task is closer to the algorithm's ability to RETAIN what it learns than its ability to find a
good policy in the first place.

### Caveat: the 3x-budget escalation did not hold up

An initial follow-up scaled the training budget 3x (30 episodes/city instead of 10) and found an
extremely dramatic result (city_1 trained alone reaching holdout reward -2453.37, the best number
anywhere in this document). **This did not survive re-verification under the fixed code** -- the
same seed's corrected trajectory monotonically got WORSE through every training phase, ending as a
statistical wash vs. baseline. Single seed, so this doesn't prove larger budgets never help, but
that specific claim is unresolved, not confirmed, and should not be cited.

### What would make this section fully paper-ready

1. Replicate the 3x-budget question properly (3+ seeds) before drawing any conclusion about how
   training budget interacts with this effect -- the one data point available reversed completely.
2. Test whether holdout-monitored checkpoint selection (picking the best-so-far checkpoint during
   sequential training, exactly the "best-round" convention already used throughout this document)
   turns this into a directly deployable recipe, independent of ever fixing the underlying retention
   problem -- not yet run.
3. Section D below (Progressive Curriculum FedAvg) is a direct attempt to get this method's search
   advantage while reducing its forgetting cost via phased-in federation -- pending results.

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
`diagnostics/progressive_curriculum_fedavg.py`. Verified via a real SUMO smoke run; the first 3-seed
pilot was killed mid-run when it exposed the §88 RNG-isolation bug (two seeds produced byte-identical
results) rather than because of anything wrong with PCFT's own design -- the fix applies
automatically on re-run (same shared `HoldoutEvaluator`), no PCFT-specific changes needed. **Not yet
re-run as of this writing -- no result to cite for this section yet.**

---

## E. Self-Anchoring Training with Confidence-Gated Reversion — bespoke mechanism targeting the
    project's own diagnosed bottleneck; CLOSED as inconclusive at 6-seed rigor

**The design:** built per direct user request for a genuinely new solution to this project's own
central bottleneck (a good policy is reachable but not retained -- see finding 3's mechanism, and
the identical symptom in evolution strategies, §84) rather than an adaptation of an existing
continual-learning method. Each federated round snapshots the round-start weights as an anchor and
tracks a running EMA of the batch Q-gap (top1-top2 masked Q-value gap) during that round's local
training -- reusing `q_values` already computed for the TD loss, no extra forward pass. This project
already established (§32-34/§53) that Q-gap growth is the validated signature of drift toward
confident lock-in, so the threshold is self-calibrated per round (a multiple of that round's own
baseline Q-gap) rather than a fixed, environment-specific magic number. If Q-gap grows past the
threshold, the weights are blended partway back toward the round-start anchor -- a partial pull-back,
not a hard reset, so local training doesn't lose all forward progress.

**Source:** `divergence_investigation.md` §90. Implementation: `agents/dqn.py`'s `anchor_revert`
flag (default off), wired through the real `--parallel` pipeline as
`--anchor_revert`/`--anchor_qgap_growth_threshold`/etc.

### The numbers (`environments_c1_4_6`, standard protocol)

| Threshold | Seeds | \|diff\|/SE best-round | \|diff\|/SE mean | Notes |
|---|---:|---:|---:|---|
| Default (3.0x growth) | 3 | 0.83 | 0.06 | Too conservative -- 1 of 3 seeds never triggered at all |
| Sensitive (2.0x growth) | 3 | 1.07 | 0.58 | Engages reliably; 2 of 3 positive |
| Sensitive (2.0x growth) | **6** | **1.53** | **0.90** | Engages reliably (10-27 triggers/run); **4 of 6 seeds positive** (one outlier +36.7%), 2 mildly negative |

**Verdict: inconclusive, not confirmed** -- the calibration fix (default too conservative → a
threshold that reliably engages) was a real, necessary correction, but even properly calibrated, the
6-seed aggregate doesn't clear this project's bar. Same honest treatment as items 23/TC-FedAvg: a
majority-positive but non-significant result, not cited as a working method. Notably, seed 11
underperformed across FOUR different mechanisms this session (TC-FedAvg, recurrent policy, and both
anchor-revert thresholds) -- suggestive of a generically hard training draw for this roster, not
specific evidence against any one intervention.

---

## F. Everything else tried in the 2026-09-06 session's item-2X campaign: confirmed nulls or
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

## G. HISTORICAL, SUPERSEDED — pre-2026-09-06 content, do not cite

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
