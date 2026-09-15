---
name: runstatus
description: Check progress of running or finished federated_training batches — per-job round count, best-so-far, pace, stalls vs. host-sleep, nonzero exits, and live confident-lock-in screening. Use when monitoring a launched batch, checking whether a long-running job is still healthy, or auditing which jobs in a batch actually completed.
---

# runstatus — is the batch healthy?

```bash
python .claude/skills/runstatus/runstatus.py --batch-log results/mybatch.log [--total-rounds 20]
python .claude/skills/runstatus/runstatus.py --dirs "results/run_2026_09_0*"
```

Reads each run's `federated_history.json`, which is rewritten every round, so
numbers are current mid-run. Per job: rounds done, best-so-far and which round it
came from, latest round, minutes since the last log write, and liveness (matched
via the run dir's PID suffix).

## Reading the output

- **`PROCESS GONE`** — the run died without a finish marker. Check the tail of its
  `training.log`. Its checkpoints up to the last completed round are still valid;
  every round is checkpointed.
- **Long idle time** — **not** automatically a stall. Host sleep freezes a run and
  it resumes cleanly on wake (§30, §42). Check liveness first: if the process is
  alive, leave it alone.
- **`lock-in screen fired`** — `eval_reward_std < 50` on those rounds. This screen
  has known false negatives (§33, §49), so it is a hint, not a verdict. Confirm a
  suspected lock-in with a real 30-episode re-eval:
  ```bash
  python diagnostics/reeval_checkpoint.py --pad_to_true_holdout --episodes 30 \
      --checkpoint results/<run>/global_round_0NN.pth
  ```
  A genuine lock-in shows rewards collapsing onto a handful of distinct values
  across 30 *different* SUMO seeds.
- **`is_true_holdout=False`** — the run is evaluating on one of its own training
  cities (§25). Any cross-topology framing of its numbers is wrong.
- **Nonzero exits** — treat the whole arm as incomplete. An arm with a missing seed
  is not a clean multi-seed comparison; relaunch the missing seed before analyzing.

## Confirming host sleep rather than guessing

A run that looks frozen for hours is usually the host sleeping. Prove it from the
round timestamps instead of assuming either way:

```bash
grep -oE "^[0-9-]+ [0-9:]+.*Federated round [0-9]+ /" results/<arm>.log | tail -8
```

A clean sleep looks like an abrupt jump between two otherwise evenly-spaced
rounds, with normal spacing resuming after:

```
2026-09-14 04:39:58   Federated round 13 / 20
2026-09-15 12:29:10   Federated round 14 / 20     <- ~32h gap, resumed fine
```

That is the documented behaviour (§30, §42) and needs no action. A *stall* looks
different: the process is alive but the log's mtime is also old, or the process is
gone entirely.

## Reading a batch whose driver mis-reported

The driver's own bookkeeping can be wrong even when every run is fine — both of
these happened here, and both made complete arms look incomplete:

- **`exit=0` on a job that did not finish.** `$?` captured after a later command
  reports that command's status. Trust `federated_history.json`'s round count over
  the driver's exit line.
- **A `run_dir` marker pointing at an aborted stub.** Logs are appended across
  relaunches; a marker written with `head -1` can name a killed launch that has no
  history. Check for multiple run dirs in one log and take the last:
  ```bash
  grep -oE "results/run_[0-9_-]+_[0-9]+" results/<arm>.log | sort -u
  ```

**Before concluding an arm is short a seed, verify against the run dirs**, not the
driver log.

## After the batch

`/seedcheck --batch-log <same log>` reads the same finish markers and does the
statistics.

## Cleanup

If a parent crashed, its daemon workers survive as orphans and hold RAM
indefinitely. `bash .claude/skills/launch/preflight.sh` detects them and prints
the exact `kill` command; SIGTERM is safe since results are checkpointed each round.
