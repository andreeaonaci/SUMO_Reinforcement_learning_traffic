---
name: seedcheck
description: Compute this project's |diff|/SE significance statistic between experimental arms of federated_training runs, with the multi-seed rigor guards (>=5-6 seeds, drop-1 outlier influence, per-seed direction, holdout-fallback and eps-decay confound checks) applied automatically. Use whenever comparing a lever/flag/architecture against a baseline, reading a finished batch, or deciding whether a result is confirmable or still a screen.
---

# seedcheck — is this result real?

The single most-repeated analysis in this project, plus the guards that catch the
failure mode it keeps hitting: a clean-looking few-seed win driven by one outlier
seed. `CQL` (2.35 → 1.05), `TC-FedAvg`, and `n_attn_layers=2` (twice) each looked
unanimous and clean at 3 seeds and evaporated at 6.

## Run it

```bash
python .claude/skills/seedcheck/seedcheck.py \
    --arm "baseline=results/run_2026_09_06-*_1?????" \
    --arm "mylever=results/mylever_seed*" \
    --baseline baseline
```

From a `run_concurrent_batch.sh` log (tags instead of globs — preferred, since the
log records the exact `run_dir` per job by PID):

```bash
python .claude/skills/seedcheck/seedcheck.py \
    --batch-log results/mybatch.log \
    --arm "baseline=tag:base_s3,tag:base_s7,tag:base_s11" \
    --arm "mylever=tag:lever_s3,tag:lever_s7,tag:lever_s11"
```

Options: `--window auto|all|21:|1:20` (default `auto` = rounds 21+ on runs of ≥30
rounds, matching the `mean(21-63)` convention in the fidings log; all rounds on
shorter runs), `--metric reward|waiting`, `--json`.

Note `--window` constrains **only the mean**. "Best-ever round" and "final round"
are over the whole run, matching how the fidings log uses those terms.

## What it reports, and how to read it

Three measures, matching the fidings log: **best-ever round**, **final round**,
**mean over the window**. For each:

- `|diff|/SE unpaired (pstdev)` — **the headline number**. Same formula
  (`sqrt(sd_a²/n_a + sd_b²/n_b)`) *and* the same population-std input as
  `experiments/analyze_phase2_strategies.py` and `analyze_phase1.py`, so it is
  comparable to the historical numbers in `fidings/divergence_investigation.md`.
  Quote this one in write-ups.
- `same, sample std` — the same statistic using the sample (n−1) std, which is the
  more defensible estimator for inference across seeds. It is **always lower** —
  ~18% at n=3, ~11% at n=5. Printed so a borderline result can't hide behind the
  choice of convention. If a result clears the bar on one and not the other, the
  tool says `CONVENTION-SENSITIVE` and it should be treated as not clearing.
- `|diff|/SE paired` — only when the arms share seeds. More powerful for
  matched-seed designs, but **not** the project's historical convention. Supporting
  detail, never the headline.
- `drop-1 range` — the headline statistic recomputed with each seed removed in turn.
- per-seed win count.

Bar: **|diff|/SE ≥ 2**.

## Non-negotiable rules this skill enforces

1. **Under 5-6 seeds is a SCREEN, never a confirmation.** The tool prints
   `SCREEN ONLY` and you must carry that word into any write-up or summary. Do not
   describe a 3-seed result as confirmed, real, or a win — regardless of how clean.
2. **If `drop-1 min` falls below 2 while the headline clears it, one seed is
   carrying the result.** Say so explicitly. This is the exact signature of the
   leads that later died.
3. **A `DIRECTION SPLIT` warning outranks a passing statistic.** A positive mean
   with half the seeds individually against it is the outlier pattern again.
4. **`is_true_holdout=False` invalidates cross-topology framing.** That is the §25
   silent-fallback trap (the run evaluated on one of its own training cities). The
   number may still be a valid in-distribution result — label it as such.
5. **More than one differing flag between arms = confounded.** The tool diffs the
   full argparse dump from each run's `training.log` and lists every difference.
   The flag under test should be the only entry.
6. **Different `--rounds` between arms is never a clean comparison.**
   `compute_eps_decay` sizes the exploration schedule from `--rounds`, so early
   rounds differ systematically (the §69 dose-response confound).
7. **A `tag:` that resolves to nothing means a MISSING seed, not a smaller arm.**
   A job's finish line carries an empty `run_dir` when it was `--resume`d or was a
   `--baseline_controller` run; the tool recovers the former and warns about the
   latter. Never analyze an arm the warnings say is incomplete — relaunch the
   missing seed first.

## After analysis

A resolved result — confirmed, null, or negative — gets written up with
`/logfinding`. A promising screen gets escalated to 6 seeds with `/launch`.
Report negative and null results with the same prominence as positive ones; the
value of this project's record is that it does that consistently.
