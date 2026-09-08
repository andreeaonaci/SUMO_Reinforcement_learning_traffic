---
name: launch
description: Preflight and launch a multi-seed training batch through analyse/run_concurrent_batch.sh — capacity/orphan-worker checks, matched-arm and true-holdout verification, correct seed sets and concurrency. Use whenever starting a seed sweep, flag ablation, screen, or 6-seed confirmation run.
---

# launch — start a batch without wasting it

A batch costs hours. Most of the ways this project has wasted one were
preventable in thirty seconds: an unmatched baseline, a silently-wrong holdout
city, a different `--rounds` between arms, or orphaned workers eating the RAM.

## Step 1 — preflight

```bash
bash .claude/skills/launch/preflight.sh <cities_per_run> <max_concurrent>
```

Checks `SUMO_HOME`/`PYTHONPATH`, live runs, **orphaned `spawn_main` workers**
(daemon workers from a crashed parent never exit on their own — ~11 of them once
held ~9GB for four days), RAM headroom against the ~1.5 + 0.65×cities GB/run
estimate, and disk.

RAM is the binding constraint, not CPU. Do not raise `MAX_CONCURRENT` on core
count: each concurrent run slows ~60%, and the net win at 3 is ~1.5x wall-clock,
not linear (§22).

## Step 2 — design the batch

**Always include a matched baseline arm in the same batch.** Comparing against a
historical number from the fidings log is how confounds get in — different code
revision, different roster, different budget. The baseline arm is the same
command with the lever at its no-op default.

**Seeds:** `3 7 11` for a screen, `+ 17 21 25` for confirmation. Reuse these —
they are what the existing record uses, so results stay comparable.

**Identical `--rounds` on both arms.** `compute_eps_decay` sizes the exploration
schedule from `--rounds`, so a 7-round run's early rounds are not comparable to a
5-round run's (§69).

**Reduced rosters need `--pad_to_true_holdout`.** Without it, `city_5_holdout`
silently falls back to a *training* city and you get an in-distribution number
wearing a cross-topology label (§25). Verify afterwards, don't assume —
`/seedcheck` flags `is_true_holdout=False`.

## Step 3 — launch

```bash
BASE_COMMON="--parallel --rounds 20 --local_episodes 2 --aggregation_strategy fedavg \
  --lr 3e-4 --lr_decay 0.97 --min_lr 1e-5 --dueling --n_step 3 --pad_to_true_holdout" \
bash analyse/run_concurrent_batch.sh results/<name>.log 3 \
  "base_s3|environments_c1_4_6|--seed 3" \
  "base_s7|environments_c1_4_6|--seed 7" \
  "base_s11|environments_c1_4_6|--seed 11" \
  "lever_s3|environments_c1_4_6|--seed 3 --mylever 0.1" \
  "lever_s7|environments_c1_4_6|--seed 7 --mylever 0.1" \
  "lever_s11|environments_c1_4_6|--seed 11 --mylever 0.1"
```

Job spec is `TAG|BASE_DIR|EXTRA_FLAGS`; `EXTRA_FLAGS` is appended after
`BASE_COMMON`, so it can override any shared default. Run it in the background —
these take hours. The log records `finished <tag> ... exit=0 run_dir=<path>` per
job, keyed to the exact PID, which is what `/seedcheck --batch-log` reads.

`--dueling --n_step 3` is the standing best-known config. Never combine
`--server_momentum` with `--dueling` (§18, net-negative; a CLI check blocks it).

## Step 4 — while it runs

Monitor with `/runstatus`. Host sleep freezes a run but does not kill it — it
resumes cleanly on wake (§30, §42), so a large wall-clock gap in a log is not a
reason to restart.

## Step 5 — after

`/seedcheck` to analyze, `/logfinding` to write up. Per standing preference:
write the fidings note and commit *before* launching the follow-up, so a lost
session never loses the result.
