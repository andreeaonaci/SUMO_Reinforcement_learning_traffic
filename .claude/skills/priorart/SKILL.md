---
name: priorart
description: Check whether a result or mechanism is actually novel before spending more compute on it, and work out what can honestly be claimed. Use before writing up a finding as a contribution, before escalating a promising lever to more seeds, when deciding a paper's framing, or whenever someone asks "is this new?".
---

# priorart — is this actually new, and what can we claim?

On 2026-09-09 this project had a 6-seed-confirmed result on four configurations
and no idea that its architecture family had been published in 2019, its
zero-shot claim in 2022 and 2024, its federated setting in 2025, and its
curriculum design in 2023. **None of that was discoverable from the experiments.**
It took one afternoon of reading to find, and it changed the paper's framing
completely while costing no compute.

`/benchmark` compares *numbers* to published work. This compares *claims*.

## Rule: do this BEFORE the escalation, not after

The natural moment is when a screen looks good and you are about to spend 6 seeds
or build a baseline. That is the cheapest possible point to discover the idea is
twenty years old, and the most expensive point to discover it is after the
write-up.

## Step 1 — decompose the claim into components

A result is almost never one idea. Split it until each piece is separately
searchable. For this project's headline:

| component | what to search |
|---|---|
| phase-invariant / action-as-input readout | "phase invariant traffic signal", FRAP |
| per-phase pressure as a feature | MPLight, pressure control |
| handling any number of phases | AttendLight, universal TSC |
| zero-shot transfer to unseen topology | inductive / transferable TSC |
| federated across heterogeneous clients | federated RL traffic signal |
| curriculum over clients | curriculum federated learning |

**Every one of those came back occupied.** The novelty was in none of them — it
was in a property nobody had isolated (see step 4).

## Step 2 — search, and read the actual mechanism

```
WebSearch: "<component> <domain> <year range>"
WebFetch:  the arXiv abs page, then the HTML full text for the mechanism
```

Do not stop at the abstract. AttendLight's abstract says "any number of phases";
its full text says *"as long as a similar configuration is represented in the
training set"* — which is a materially weaker claim and is exactly the gap this
project's result fills. That distinction is invisible from the abstract.

When a method is vendored in the repo, **read its source, not its paper.**
RESCO's `agents/action_value/mplight.py` revealed that MPLight requires
hand-authored per-signal configuration — a fact stated plainly in their docs and
absent from every summary of the method.

## Step 3 — age cuts both ways

- **As a baseline, age is irrelevant.** FRAP (2019) and MPLight (2020) are still
  the canonical comparisons; reviewers expect them regardless of year.
- **As prior art, old is worse for you.** If a 2019 paper already did your
  mechanism, it is not new, and having missed something that old reads badly.
  "That's an old paper" is a reason to check it *more* carefully, not less.

## Step 4 — find what survives, and name it precisely

After everything occupied is stripped away, what is left is usually narrower and
more defensible than the original claim. Here, what survived was:

> Prior phase-invariant methods achieve topology transfer with **hand-authored
> per-signal movement configuration**. We derive it automatically from the
> simulator's topology.

That is checkable (it is in RESCO's own docs), it is not a performance claim, and
it turned out to be supported. **A narrow claim you can defend beats a broad one
you cannot.**

## Step 5 — separate capability from parity

These are different claims with different evidence bars:

- **Capability:** "we do not require X". Evidence: the other method's own
  documentation. Cheap, often already in hand.
- **Parity:** "and we match them anyway". Evidence: a run of their method in your
  protocol. Expensive — see `/lever`'s baseline-porting section.

State which one you are making. Claiming parity on capability evidence is the
most common way a positioning claim becomes indefensible.

## Step 6 — write the ledger

Produce three explicit lists, and put them in the fidings log via `/logfinding`:

1. **Can claim, fully supported** — with the statistic and seed count.
2. **Can claim with a stated caveat** — and the caveat in the same sentence.
3. **Cannot claim** — with the paper that owns it.

The third list is the valuable one. It is what stops a later session from
re-asserting something already known to be occupied.

## What this project has already checked (don't redo)

| claim | owned by |
|---|---|
| phase-invariant readout, transfer without retraining | FRAP, CIKM 2019 |
| per-phase pressure | MPLight AAAI 2020; G2P 2025 |
| any number of phases (with training-set caveat) | AttendLight, NeurIPS 2020 |
| zero-shot transfer to unseen networks | MuJAM 2022; TransferLight Dec 2024 |
| federated RL for TSC, incl. clustered aggregation | HFRL, Apr 2025 |
| curriculum over clients in FL | Vahidian et al., ICCV 2023 |

Full write-up and sources: `fidings/divergence_investigation.md` §101.
