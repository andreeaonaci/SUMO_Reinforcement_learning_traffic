# Paper source

IEEEtran conference-format draft of the action-representation result
(`fidings/divergence_investigation.md` §95–§104).

## Build

```bash
cd paper
pdflatex main.tex && pdflatex main.tex     # twice, for cross-references
```

Requires `IEEEtran.cls` (Debian/Ubuntu: `texlive-publishers`), plus `tikz`,
`booktabs`, `amsmath`, `xcolor`, `hyperref` — all in `texlive-latex-recommended`
and `texlive-science`. Verified compiling clean to 5 pages with no undefined
references.

All three figures are hand-authored TikZ; there are no external image files, so
the figures stay editable and vector-clean.

**Fig. 1 uses absolute millimetre coordinates**, not relative `below=of`
positioning. That is deliberate: the node heights differ enough that a relative
chain stacks them on top of one another. If you edit that figure, keep the
explicit `at (x,y)` placement.

## Where the numbers come from

Every figure in the tables is traceable to a numbered section of the
investigation log:

| Table | Content | Source |
|---|---|---|
| I | dead-row control | §98 |
| II | zero-shot at 20-round budget | §103 |
| III | cologne3 in-distribution | §103b |
| IV | ingolstadt7 in-distribution | §103b |
| V | curriculum re-run on the corrected readout | §102 |

Published IPPO/IDQN figures are RESCO's own, quoted only for the
benchmark-exact roster (`environments_rescofull`). **Do not substitute numbers
from `environments_c1_4_6`** — that roster carries the three scenario mismatches
§99 documents, and quoting it against published work is the error that retracted
§59.

## Claims deliberately not made

- No novel architecture — phase-invariant readouts predate this work (FRAP 2019,
  AttendLight 2020). The contribution is the control and its consequence.
- No comparison against a current zero-shot method (TransferLight, 2024). Until
  that is run, competitiveness with the recent state of the art is unestablished
  and the Limitations section says so.
- In-distribution superiority over `max_pressure` is **not** claimed: the paper
  states the Cologne margin is unadjudicable on throughput grounds and that
  Ingolstadt is an outright loss.
