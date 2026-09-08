---
name: benchmark
description: Compare this project's results against published numbers (RESCO, AttendLight, MPLight, FRAP) without producing an invalid comparison. Use whenever quoting an external paper's numbers, positioning a result against prior work, or reporting absolute performance rather than a within-project delta.
---

# benchmark — comparing to published numbers without fooling yourself

On 2026-09-08 this project discovered it had **never run RESCO's scenario configuration**, despite
quoting RESCO's numbers since §58. Three independent mismatches, none of which announced itself.
One published claim (§59's "4.4x behind IDQN") had to be retracted.

The rule: **a within-project delta needs no scenario audit; an absolute number quoted against
external work needs all of it.**

## Step 1 — audit the configuration before running anything

External benchmarks ship their own config. Read it, don't assume.

```bash
cat sumo_rl/nets/RESCO/<scenario>/<scenario>.sumocfg     # route file, begin, end
```

Then check the benchmark's *runner* config too — the `.sumocfg` does not carry signal timing.
RESCO's lives in `resco_benchmark/config/config.yaml` in their repo.

The four that have actually differed here:

| what | how it bit |
|---|---|
| **route file** | ours used `*_shifted.xml`, the benchmark uses the original |
| **time window** | cologne3 demand spans 1.47h; RESCO takes the LAST hour, we took the FIRST |
| **`yellow_time`** | RESCO 3, `sumo_rl` default 2 — with a 5s step that is **50% more usable green**, for every controller, in every experiment |
| **metric definition** | see step 3 |

`environments_resco/` holds RESCO-exact configs; `environments_y3/` is the standard training roster
at `yellow_time: 3` so new results are comparable by construction.

## Step 2 — validate with a rule-based probe BEFORE evaluating any checkpoint

Run `max_pressure` on the reconstructed scenario and check it lands where a competent rule-based
controller should relative to the published methods.

```bash
python diagnostics/eval_paper_metrics.py max_pressure \
    --base_dir environments_resco --city city_4_resco --episodes 5
```

**If your rule-based controller beats every published RL method, the setup is wrong.** That is
exactly how the yellow-time mismatch was caught: at `yellow_time: 2`, `max_pressure` scored delay
19.3 against RESCO's best of 22.13. At 3 it scored 22.4 — between IPPO and IDQN, where it belongs.
The probe costs two minutes and is the only thing standing between you and an invalid table.

## Step 3 — only compare metrics that reconcile

Measured against RESCO on two scenarios:

| metric | reconciles? |
|---|---|
| **Avg. Delay** (SUMO `timeLoss`) | ✅ use it |
| **Avg. Trip Time** | ✅ use it |
| Avg. Wait | ❌ ours read 1.49s vs IDQN 8.5 on one scenario and 293.6s vs 8.71 on another — inconsistent in *both* directions, so it is a different quantity |
| Avg. Queue | ❌ runs ~3x high, different normaliser |
| **reward** | ❌ never. It is this project's internal `diff-waiting-time` sum in units of 100 vehicle-seconds |

Report delay and trip time. State the others as differently defined rather than quietly omitting
them.

## Step 4 — state the comparison honestly

- Published tables here are **in-distribution** (they train on the scenario they report). A
  zero-shot number is a *different and harder* claim — say which you are making.
- RESCO reports **best episode averaged over 5 seeds**. Match that, or say you didn't.
- Prefer *"improves delay by N% over a `max_pressure` reference measured under identical
  conditions"* to *"beats IDQN"*, unless the probe in step 2 came out clean.

## Step 5 — check the prior art is actually beaten

Beating `fixed_time` and `max_pressure` establishes nothing against methods designed for the
problem. For phase-based readouts the relevant prior art is **AttendLight** (NeurIPS 2020, varying
phase counts, unseen intersections), **MPLight** (AAAI 2020) and **FRAP** (CIKM 2019). If a claim
of novelty is being made, these are the baselines a reviewer will ask for.
