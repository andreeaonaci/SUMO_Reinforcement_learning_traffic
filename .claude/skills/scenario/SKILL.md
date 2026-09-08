---
name: scenario
description: Build or calibrate a SUMO evaluation scenario (new holdout, congestion test, demand sweep) so it can actually distinguish controllers. Use before adding any roster or route file to an experiment — it prevents ceiling and floor effects, and checks the holdout's topology is genuinely unseen.
---

# scenario — make a holdout that can measure something

A scenario that is too easy or too hard measures nothing, and both failures look like a finished
experiment. On 2026-09-08 two holdouts were built for one comparison and **both were wrong, in
opposite directions**, each caught only by a rule-based probe.

## Step 1 — never trust a filename, measure the demand

`grid4x4_dense.rou.xml` sounds heavy. It is a `randomTrips` file with 60 flows at `period=60` over
`begin=0 end=1200` — **1200 vehicles in the first third of an episode**, lighter than the
1473-vehicle baseline it was meant to stress. Both rule-based controllers scored exactly 0.00s
waiting: a ceiling where nothing can be distinguished.

```bash
grep -c "<vehicle" <route>.rou.xml                 # listed vehicles
grep -oE 'begin="[0-9.]+"|end="[0-9.]+"|period="[0-9.]+"' <route>.rou.xml | sort | uniq -c
```

Flow-based files need the arithmetic: `(end - begin) / period` vehicles per flow.

## Step 2 — calibrate by the BASELINE CONTROLLER'S waiting time, not per-signal counts

Vehicles-per-signal does **not** normalise congestion across networks of differing capacity.
Matching 3x3Grid2lanes to grid4x4 at 92 veh/signal produced `max_pressure` at 0.00s waiting — from
gridlock straight to free-flow in one step, because the two networks have very different capacity.

The scenario is usable when the rule-based probe shows:

- `max_pressure` **functional but not perfect** — measurable waiting, ~95%+ trips completed
- `fixed_time` **clearly worse** — that spread is the room a learned controller has to land in
- neither at 0.00s (ceiling) nor gridlocked (floor)

```bash
python diagnostics/eval_paper_metrics.py max_pressure fixed_time \
    --base_dir <roster> --episodes 3
```

Rescale until it sits in range:

```bash
python diagnostics/scale_demand.py IN.rou.xml OUT.rou.xml --factor 0.5
```

`--factor <1` subsamples on a uniform stride (keeps the temporal profile); `>1` duplicates with
headway-spread departures. Verify the burst structure actually scaled rather than smoothing away —
peak density should move with the factor.

## Step 3 — verify the holdout topology is genuinely unseen

`is_true_holdout=True` only means the evaluator resolved to `city_5_holdout` instead of falling
back to a training city (the §25 trap). It does **not** check whether a training city points at the
same road network under another name — and `city_7` does, sharing `grid4x4.net.xml` with the
holdout. Three rosters (`environments`, `environments_wide`, `environments_phase0`) were affected
for months.

`federated_training.py` now warns on this automatically. Confirm the warning is absent, and check
directly when building a roster:

```bash
grep -h net_file <roster>/*/config.yaml | sort | uniq -c   # holdout net must appear ONCE
```

## Step 4 — know which property you are varying

A control isolates one property. When the dead-rows control was first built it changed **two**
things at once — holdout phase count *and* congestion level — which would have confounded it.
Fix demand when varying topology; fix topology when varying demand.

For a phase-count control specifically: holdout phases **≤** training maximum means the indexed
head has no untrained rows (§95b), which is what separates "the representation transfers" from
"we repaired a defect".

## Step 5 — record the calibration

Put the probe numbers in the write-up. A scenario without its rule-based reference is not
interpretable later, and reward scale is scenario-specific — `max_pressure` scores −0.34 on one
holdout and −2.225 on another. **Never compare rewards across scenarios**; compare each arm to the
reference measured on its own.
