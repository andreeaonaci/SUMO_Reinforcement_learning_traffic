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

## Writing your own driver — four bugs that cost this project real batches

`analyse/run_concurrent_batch.sh` covers the standard case. When a batch needs a
custom driver (a non-`federated_training` script, a growing city pool, a chained
queue), it must be **safe to stop and relaunch** — hosts sleep, sessions die,
machines get shut down mid-batch. Working examples:
`analyse/run_pcft_phase6.sh`, `run_budget_sensitivity.sh`, `run_rescofull_frap.sh`.

**Make every job skip-or-resume.** Record each run's `run_dir` in a sidecar;
on relaunch, skip if it already has the target rounds, else pass
`--resume <dir>`. For scripts with no resume of their own, add checkpointing
first — `progressive_curriculum_fedavg.py` had *none* (zero `torch.save`), so a
killed 70-minute run lost everything including the model.

Then these four, each of which actually bit:

1. **`local a=$1 b=$a` breaks under `set -u`.** Bash expands every word of a
   `local` before assigning any of them, so `$a` is unbound. Split into separate
   `local` statements.

2. **`grep … | head -1` on an appended log returns the wrong run.** These logs are
   appended across relaunches, so `head -1` can return an aborted stub from a
   killed launch that has no `federated_history.json` — making a complete arm
   look like it has fewer seeds. Use `tail -1`: a resumed run re-logs the same
   directory, so tail is correct in both cases.

3. **`$?` after any other command is that command's status.** `python …; grep …;
   log "exit=$?"` reports the grep's exit code, so every job looks successful.
   Capture `local rc=$?` on the line immediately after the command you care about.

4. **`wait` can return while jobs are still live**, logging BATCH DONE early and
   letting a chained queue advance over a running batch. Poll the real processes:

   ```bash
   drain() {
     wait 2>/dev/null || true
     while pgrep -f "experiments.federated_training" > /dev/null; do sleep 30; done
   }
   ```

Also: `chmod +x` the script you actually invoke (exit 126 means you didn't), and
**validate globs by counting** — a run-dir glob that silently matched 4 of 6
seeds would have produced a 4-seed mean presented as 6. Prefer explicit lists,
and make an analysis script *abort* on an incomplete arm rather than emit a
partial table.

## Keeping the machine busy across batches

Chain stages with a runner that drains, then launches the next:

```bash
while pgrep -f "experiments.federated_training" > /dev/null; do sleep 120; done
SEEDS="3 7 11 17 21 25" bash analyse/run_next_batch.sh
```

Because each stage is skip-or-resume, a killed chain costs nothing — relaunching
by hand re-runs no completed work. Expect to relaunch: the harness reaps
long-lived polling scripts under memory pressure. **It kills the poller, never
the training** — check `ps` before assuming a batch died.

## Stopping a batch cleanly (before a shutdown)

**Kill the driver first, or its loop immediately relaunches the next job.**

```bash
pkill -TERM -f "run_<name>.sh"          # driver
pkill -TERM -f "experiments.federated_training"
pkill -TERM -f "spawn_main"             # daemon workers outlive a killed parent
```

Then verify all four counts are zero, record the exact per-job state, and write
the single command that resumes it. In-flight `run_dir` markers are only written
on completion, so write them by hand before killing or the partial run restarts
from zero.

## Step 4 — while it runs

Monitor with `/runstatus`. Host sleep freezes a run but does not kill it — it
resumes cleanly on wake (§30, §42), so a large wall-clock gap in a log is not a
reason to restart.

## Step 5 — after

`/seedcheck` to analyze, `/logfinding` to write up. Per standing preference:
write the fidings note and commit *before* launching the follow-up, so a lost
session never loses the result.
