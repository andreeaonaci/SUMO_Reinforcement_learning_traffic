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

## After the batch

`/seedcheck --batch-log <same log>` reads the same finish markers and does the
statistics.

## Cleanup

If a parent crashed, its daemon workers survive as orphans and hold RAM
indefinitely. `bash .claude/skills/launch/preflight.sh` detects them and prints
the exact `kill` command; SIGTERM is safe since results are checkpointed each round.
