# No-Federation vs Federated Aggregation — Comparison (2026-08-04)

Numbers below are pulled directly from `results/run_2026_08_03-21_11_36/federated_history.json`
(run A) and `results/run_2026_08_04-09_12_09/federated_history.json` (run B), plus checkpoint
file timestamps and `training.log` — re-derivable from those files if anything here looks off.

## Why we ran this

Two federated trainings (A, B) were launched back-to-back on 2026-08-03/04, both using
`--no_federation`. A PC restart killed run B partway through round 2. While diagnosing what was
lost, we found that `--no_federation` was also the dominant cause of a round-time regression the
user had separately flagged (~14 min/round expected vs ~40 min/round observed). The `city_1` map
swap (single 2-way intersection → 16-intersection RESCO `arterial4x4`, done to fix the
worst-local-loss-growth issue described in `divergence_investigation.md`) was suspected as the
cause but turned out to be a minor factor: per-city wall time for 1 simulated hour showed
`city_1`'s new map (10.0 min) sitting right in line with the other 16-intersection RESCO nets
(`city_3` 9.5 min, `city_7` 9.7 min) — see the table near the bottom. The real driver is
`_evaluate_multiple_models` (`federated/parallel_server.py:267`): under `--no_federation`, every
round's holdout evaluation runs **once per city** (6 cities × `eval_episodes=3` = 18 evaluation
episodes/round) instead of once for the single aggregated model (3 episodes/round) that the
federated path uses. That alone accounts for the bulk of the ~40 vs ~20 min/round gap measured
below.

Rather than resume the crashed run B in place (not supported by the current code — see
[Resume capability](#resume-capability-investigated-not-implemented)), we relaunched B from
scratch with `--no_federation` removed. This both got B running at the expected pace and turned
what was originally an accidental duplicate run into a useful first-look comparison of the two
aggregation regimes.

## What we're comparing

- **Run A** (`results/run_2026_08_03-21_11_36`, `--no_federation`): each of the 6 training
  cities keeps and updates its own independent model; no weight aggregation across cities. The
  logged eval numbers are the **average of 6 separate holdout evaluations**, one per city's
  independently-trained model (`_evaluate_multiple_models`).
- **Run B** (`results/run_2026_08_04-09_12_09`, plain `--aggregation_strategy fedavg`): cities
  train locally, weights are aggregated (masked-head weighted average) into one shared global
  model each round, and the holdout eval is a **single evaluation of that one aggregated model**.

Both runs share: the same 7-city `environments/` roster (`city_1` already swapped to
`arterial4x4`), `--seed 42`, `--rounds 10`, `--local_episodes 2`, `--lr 3e-4 --lr_decay 0.97
--min_lr 1e-5`, `--eval_episodes 3`, `--eval_sumo_seed 12345`, `--parallel`.

**Caveat:** this is not a strictly controlled "federated vs not" ablation. A's metric (mean of 6
independently-trained models) and B's metric (one shared model) are different quantities by
construction, not just the same kind of number produced under different training regimes. Treat
this as a first orientation, not a paper-ready comparison — see
[Open questions](#open-questions--next-steps).

## Round-by-round results

| round | A reward | A wait(s) | A stopped | A arrived | B reward | B wait(s) | B stopped | B arrived |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | -9558.78  | 1809.36 | 1032.1 | 233.9 | -9515.74  | 1773.12 | 1049.7 | 236.0 |
| 2  | -9728.36  | 1814.40 | 1045.4 | 216.6 | -9546.59  | 1826.17 | 1013.3 | 247.7 |
| 3  | -8958.18  | 1708.35 | 1001.9 | 267.6 | -9666.55  | 1864.11 | 1027.3 | 243.0 |
| 4  | -9469.99  | 1821.14 | 1032.7 | 215.3 | -8607.88  | 1501.19 |  960.7 | 452.3 |
| 5  | -8927.44  | 1767.75 |  978.4 | 299.0 | -10244.02 | 1791.66 | 1093.3 | 208.0 |
| 6  | -9709.66  | 1868.19 | 1040.8 | 220.2 | -7138.74  | 1353.50 |  931.3 | 448.7 |
| 7  | -9506.99  | 1785.21 | 1031.7 | 247.7 | -10135.37 | 1832.81 | 1068.3 | 198.0 |
| 8  | -9732.46  | 1805.17 | 1057.8 | 198.9 | -10118.64 | 1692.37 | 1086.3 | 272.3 |
| 9  | -9130.97  | 1770.68 | 1002.6 | 265.4 | -8626.09  | 1508.60 |  960.7 | 469.7 |
| 10 | -8647.95  | 1721.51 |  947.7 | 342.1 | -8577.08  | 1532.51 |  954.7 | 466.7 |

Summary stats across the 10 rounds:

| | mean reward | std reward | best round | worst round |
|---|---:|---:|---:|---:|
| A | -9337.1 | 370.5 | -8648.0 (r10) | -9732.5 (r8) |
| B | -9217.7 | 925.2 | -7138.7 (r6)  | -10244.0 (r5) |

## Wall-clock

| round | A duration | B duration |
|---:|---:|---:|
| 1  | 36.9 min | 18.6 min |
| 2  | 39.6 min | 20.9 min |
| 3  | 41.4 min | 20.6 min |
| 4  | 41.0 min | 21.4 min |
| 5  | 43.4 min | 22.1 min |
| 6  | 41.2 min | 22.4 min |
| 7  | 40.7 min | 22.2 min |
| 8  | 41.1 min | 22.6 min |
| 9  | 40.5 min | 22.6 min |
| 10 | 41.2 min | 22.3 min |
| **avg** | **40.7 min/round** | **21.6 min/round** |
| **total** | **407.1 min (6h47m)** | **215.5 min (3h36m)** |

## Analysis

- **Round time:** B is ~1.9x faster than A (21.6 vs 40.7 min/round), essentially entirely
  attributable to the `--no_federation` eval multiplier (18 vs 3 holdout episodes/round); the
  `city_1` map size is a minor contributor at most (see the per-city benchmarking table below).
- **Final performance:** comparable at round 10 (A -8647.95, B -8577.08), and round 10 is close
  to each run's session-best in both cases.
- **Volatility:** B is much noisier round-to-round (std 925 vs A's 370), swinging from -7138.74
  (round 6, best) to -10244.02 (round 5, worst) on consecutive-ish rounds. This is consistent
  with what you'd expect structurally: B's number reflects one shared model directly, so
  whichever cities dominate that round's aggregation weighting show up immediately in the
  holdout score, while A's "average of 6 independent models" metric smooths over any single
  city's bad round.
- **No clean monotonic improvement in either run** over 10 rounds — both bounce around a fairly
  flat band, with round 10 landing near session-best for both rather than being the tail of a
  clear downward trend. This echoes the open non-determinism question already on record in
  `divergence_investigation.md` (§3): 10 rounds isn't enough here to cleanly separate "still
  learning" from "round-to-round noise" for either aggregation mode.

## Resume capability (investigated, not implemented)

- No `--resume` / `--init_checkpoint` flag exists in `experiments/federated_training.py`.
- The saved `global_round_NNN.pth` files are just `agent.q.state_dict()`
  (`federated/parallel_server.py:535`) — directly loadable via `DQNAgent.load()` /
  `load_state_dict()` (`agents/dqn.py:328-341`), so a "load these weights as the starting point"
  flag would be a small change (~15 lines).
- What that quick version would **not** recover, because none of it is persisted anywhere today:
  per-client epsilon schedule state (`steps_done`), Adam optimizer momentum, replay buffer
  contents, or LR-decay progress. A byte-for-byte faithful resume would need all four serialized
  per client — real but currently-missing work.
- We chose not to build this for the crash recovery itself (round 2 of the crashed run had only
  ~10 minutes of local training in it) and instead just relaunched run B from scratch. Flagging
  this as a known gap for the next time a long run needs to survive an interruption.

## Per-city 1-hour-simulation wall time (city_1 map swap sanity check)

Measured from run A's `training.log`: per-worker time from env-ready to first `ep=1/2 steps=720`
(= 3600s simulated / `delta_time=5`).

| city | map | intersections | wall time / 1h sim |
|---|---|---:|---:|
| city_4 | cologne3 | 3 | 7.8 min |
| city_6 | ingolstadt7 | 7 | 8.5 min |
| city_3 | 4x4-Lucas | 16 | 9.5 min |
| city_7 | grid4x4 (dense) | 16 | 9.7 min |
| **city_1** | **arterial4x4 (new)** | **16** | **10.0 min** |
| city_2 | 3x3grid | 9 | 10.2 min |

`city_1`'s new map sits right in line with the other 16-intersection RESCO nets and isn't a
training-time outlier — confirms the `--no_federation` eval multiplier, not the map swap, drove
the round-time regression this investigation started from.

## Open questions / next steps

- Is B's higher round-to-round volatility a real structural property of single-model fedavg
  aggregation on this roster, or an artifact of a single seed? Worth a multi-seed comparison
  before drawing any conclusions for the paper.
- Neither run shows a clean learning curve over 10 rounds — same open question as
  `divergence_investigation.md`. Worth extending to 20 rounds (as the divergence-investigation
  runs did) before concluding either config has converged.
- If resumability becomes a recurring need (this is the second crash-adjacent incident on
  record), build the checkpoint/resume plumbing described above rather than re-deriving the
  analysis ad hoc each time.
