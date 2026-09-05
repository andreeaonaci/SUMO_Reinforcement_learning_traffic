# Algorithm swap: PPO / Munchausen-DQN vs. DQN (2026-09-04/05 overnight session)

**Status of this document:** standalone, consolidated summary of one overnight experimental
campaign — a condensed, better-organized version of the same material scattered incrementally
across `fidings/divergence_investigation.md` §73 (batches 1-6). Read this for the full story in
one place; read §73 for the exact per-round numbers and commit-by-commit trail if needed.

## Motivation

By 2026-09-04, this project's standing diagnosis (`divergence_investigation.md` §70/§71) was that
the federated DQN pipeline's binding constraint is **retention/transfer failure**, not training
data quantity or diversity: a randomly-initialized network, fine-tuned on the holdout city, beat
the federated-pretrained network fine-tuned the same way, and widening the training roster from 3
to 14 cities produced no improvement. The user asked directly: keep FedAvg, but replace DQN itself
with something better suited to this task — specifically, an algorithm whose policy can't
collapse into the "confident lock-in" failure mode already characterized at length in this
project (§32-34/§51-57: `argmax(Q)` sometimes locks onto one repeating action regardless of actual
traffic state, byte-identical rewards across 30 different SUMO seeds on some checkpoints).

## What was built

Two alternative agents, both implemented to match `DQNAgent`'s external interface exactly
(`start_round`/`current_epsilon`/`train`/`decay_lr`/`state_dict`/`load_state_dict`/`act`) so the
federated round-loop code (`federated/client.py`, `FederatedServer`, `ParallelFederatedServer`)
needed zero changes beyond a new `--algo {dqn,ppo,munchausen}` CLI flag:

- **`agents/ppo.py`** — on-policy actor-critic (clipped surrogate objective, GAE, entropy bonus),
  reusing the existing `NeighborAttentionQNetwork` trunk via new `policy_head`/`ac_value_head`
  outputs (`actor_critic=True` mode, mutually exclusive with `dueling`).
- **`agents/munchausen_dqn.py`** — Munchausen-DQN (Vieillard et al. 2020). Keeps DQN's **off-policy
  replay buffer** (same sample efficiency as DQN, unlike PPO) but replaces the hard
  `argmax(Q)`/epsilon-greedy policy with a Boltzmann distribution over Q-values and an
  entropy-regularized soft-Bellman target (a "Munchausen" bonus term — the target network's own
  clipped log-policy of the action taken — plus a soft state value via `logsumexp`).

Two real pre-existing bugs were found and fixed while wiring this up: `federated/server.py` and
`federated/parallel_server.py` both hardcoded `global_model.q.state_dict()`/`.q.parameters()`
(DQNAgent-specific attribute access) for checkpoint-saving and weight-norm logging, which would
have crashed for any non-DQN agent. Switched to the already-existing, agent-agnostic
`state_dict()` interface both agents implement.

Also added, as CLI flags (previously hardcoded): `--d_model`/`--n_heads` (network capacity, for a
"does a bigger/smaller network help at all" sanity check independent of algorithm) and
`--munchausen_temp`/`--munchausen_alpha` (Munchausen's own hyperparameters).

## The experimental campaign

All runs: `environments_c1_4_6` (3-city roster: arterial4x4/cologne3/ingolstadt7, action_dims
5/4/3), `--pad_to_true_holdout`, true `city_5_holdout` evaluation, `--parallel`. Six batches,
escalating from cheap single-seed screens to a rigorous multi-seed comparison at full budget.

### Batches 1-2: hyperparameter screening (5-round budget, seed 3)

| config | best | mean |
|---|---:|---:|
| DQN + q_entropy=0.05 (reference) | -5933.60 | -8016.66 |
| Munchausen temp=0.1, alpha=0.9 | -7932.27 | -9127.86 |
| Munchausen temp=0.01, alpha=0.9 | -6382.16 | -7490.44 |
| Munchausen temp=0.03, alpha=0.5 | -8016.12 | -9097.92 |
| Munchausen temp=0.01 + dueling | -9738.03 | -9862.08 (locked the whole run) |
| **Munchausen temp=0.01 + n_step=3** | **-5514.84** | **-6561.87** |
| Munchausen temp=0.005 | -6516.09 | -7286.82 |

`temp=0.01 + n_step=3` looked like a clean win — beat DQN on both measures, improved
monotonically every round, no lock-in.

### Batches 3-4: the methodology lesson — this "win" was a single-seed mirage

Seed replication (seeds 7, 11, 17) and extending the same seed-3 config to the full 20-round
budget both undercut it:

| check | result |
|---|---|
| Seed 7 (5 rounds) | best -7879.46, mean -8571.60 — worse than DQN reference |
| Seed 11 (5 rounds) | best -9550.65, mean -10015.82 — locked most of the run |
| Seed 17 (5 rounds) | best -6529.73 (late escape only), mean -9302.95 |
| **Seed 3, extended to 20 rounds** | best -3592.65, mean -6065.33 — vs. DQN's own 20-round -3288.73/-6093.66: **a wash, not a win** |

None of the three new seeds reproduced the clean early improvement. At full budget, even the
original seed only ties DQN instead of beating it, and shows the same late-training regression
pattern documented throughout this project (peaked at round 9, degraded through round 20).
**Reading: the batch-1/2 result was a favorable single-seed draw, not a real property of the
config.** Two more useful negatives from the same batch: `n_step=3` alone (no Munchausen) added to
plain DQN+q_entropy produces no change (best -5945.46, mean -8173.02 — statistically the same as
the -5933.60/-8016.66 reference); PPO with a 4x larger episode budget (`--local_episodes 8`,
removing its on-policy sample-inefficiency handicap) partially closes its own gap
(best -8795.28 vs. -10190.73 at the smaller budget) but remains far behind DQN-family results
regardless.

### Batch 5: a real 3-seed comparison reframes the whole campaign

The seed-3 DQN number used as "the baseline to beat" throughout batches 1-4 was itself checked
against two more seeds — and turned out to be a favorable outlier for DQN too, not just for
Munchausen:

| algo | seed | best | mean |
|---|---:|---:|---:|
| DQN+qew | 3 | -5933.60 | -8016.66 |
| DQN+qew | 7 | -9141.15 | -9396.41 |
| DQN+qew | 11 | -9008.52 | -9463.45 |
| Munchausen-default | 3 | -7198.59 | -8539.36 |
| Munchausen-default | 7 | -7885.15 | -8806.43 |
| Munchausen-default | 11 | -8701.05 | -9394.84 |

Averaged across 3 seeds, DQN+qew (-8027.76 best / -8958.84 mean) and Munchausen-default
(-7928.26 best / -8913.54 mean) are within ~100 points of each other on both measures — dwarfed
by DQN's own ~1800-point seed-to-seed spread.

A capacity-axis check on plain DQN (never done before this session) completed the picture started
by Munchausen's earlier `d_model=512` failure: `d_model=64` locked immediately and never escaped
(clean dud); `d_model=256` underperformed the existing default of 128. **Every capacity deviation
tried, on every algorithm, underperformed the existing 128-width default.**

### Batch 6 (final): the same 3-seed comparison at the full 20-round budget

| algo | seed | best | mean(1-20) |
|---|---:|---:|---:|
| DQN+qew | 3 | -3288.73 | -6093.66 |
| DQN+qew | 7 | -6406.89 | -8147.82 |
| DQN+qew | 11 | -7752.10 | -9252.74 |
| **DQN+qew, 3-seed avg** | — | **-5815.91 (std 2289.6)** | **-7831.41 (std 1603.1)** |
| Munchausen-default | 3 | -6349.36 | -7862.28 |
| Munchausen-default | 7 | -6155.44 | -7817.16 |
| Munchausen-default | 11 | -6802.57 | -8609.05 |
| **Munchausen-default, 3-seed avg** | — | **-6435.79 (std 332.1)** | **-8096.16 (std 444.7)** |

**|diff|/SE = 0.46 (best-round), 0.28 (mean)** — both far below this project's own ≥2
significance bar. **DQN+q_entropy and Munchausen-DQN are statistically indistinguishable at the
full training budget, with real (if small, n=3) multi-seed grounding, not a single-seed anecdote.**

## Two things worth carrying forward

1. **The methodology lesson is as valuable as the result.** A clean-looking single-seed win
   (batch 1-2) evaporated under both seed replication and extended budget (batches 3-4) — the
   *n*-th instance of this exact pattern in this project's history (§11→12, §30→31, §46→47), now
   demonstrated inside a brand-new algorithm rather than an existing one. Reinforces: never trust
   a single-seed result in this codebase, regardless of how clean the trajectory looks.
2. **An open, specific hypothesis for later:** Munchausen-DQN's seed-to-seed variance was 5-7x
   smaller than DQN's (best-round std 332.1 vs. 2289.6) even though its mean wasn't better — DQN's
   one good seed (3) was doing most of the work of making it look ahead of Munchausen at all. This
   is a genuine consistency-vs-peak-performance question, distinct from "which has the better
   mean," and not yet tested rigorously (n=3 per side is a start, not a confirmation). Worth a
   dedicated multi-seed variance comparison (e.g. Levene's test or a simple F-test on 5+ seeds
   per side) in a future session, if the project's confident-lock-in/instability story
   (§32-34/§51-57) is ever revisited.

## Bottom line

**No algorithm swap earned its keep.** PPO underperformed clearly (mitigated but not fixed by a
larger episode budget). Munchausen-DQN, at every hyperparameter/architecture/capacity combination
tried, never showed a statistically supportable improvement over plain DQN+q_entropy — the one
config that looked like a win was a single-seed mirage. **Recommendation: keep DQN+q_entropy as
the standing default going forward; do not adopt PPO or Munchausen-DQN as a replacement based on
this campaign.**

## Where the raw data lives

Every run's full log: `results/overnight_sweep_logs/` (numbered `01_...` through `23_...`, plus
the earlier same-night pilots under `results/pilot72_logs/`). Every run's checkpoints and
`federated_history.json`: `results/run_2026_09_04-*` / `results/run_2026_09_05-*` (matched to log
files by PID in the directory name — see each log's own startup banner for the exact PID). All
untracked local output, same caveat as every other reproducibility index in this project.
