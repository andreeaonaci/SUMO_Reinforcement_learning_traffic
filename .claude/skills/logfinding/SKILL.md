---
name: logfinding
description: Write up an experimental result into this project's lab-notebook chain — a numbered dated section in fidings/divergence_investigation.md, the compressed project_knowledge_summary.md, and CLAUDE.md's RESUME HERE block — then commit and push. Use whenever a run, screen, confirmation, bug fix, or null result is resolved and needs to survive the session.
---

# logfinding — make the result outlive the session

This project's real output is its record. A result that isn't written down before
the next launch is a result that can be lost — and per standing preference, the
write-up and commit come **before** launching the follow-up.

## The document chain

| file | what goes there |
|---|---|
| `fidings/divergence_investigation.md` | full detail, exact numbers, derivations. **Source of truth.** |
| `fidings/project_knowledge_summary.md` | compressed map of the whole arc. Update when a headline changes. |
| `CLAUDE.md` "RESUME HERE" | what a cold session needs first. Update when current status changes. |
| `fidings/paper_results_summary.md` | paper-facing ledger. Update when a result becomes confirmed/retracted. |
| `README.md` | public condensed summary. Only for headline-level changes. |

If a number in a summary ever disagrees with `divergence_investigation.md`, the
investigation log wins — fix the summary.

## Writing the section

Sections are `## <N>. <Title>` and go **immediately before the final
`## Open questions / next steps` section**, not at end of file. Find the number:

```bash
grep -n "^## " fidings/divergence_investigation.md | tail -3
```

A section must contain, in this order:

1. **Date** (`**2026-09-08.**`) and what was tested, in one sentence.
2. **Why** — which prior section or hypothesis this follows from, with `§NN` refs.
3. **Implementation** — what was built, which files, what was reused vs. new. Note
   explicitly when a primitive from the real pipeline was reused rather than
   reimplemented (that's what makes a diagnostic result trustworthy).
4. **Verification before compute** — the smoke test, and what it proved.
5. **Result** — the table. Per-seed values, then |diff|/SE on each measure, with
   the seed count stated in the same sentence as the statistic.
6. **Verdict** — one of: CONFIRMED / null / negative / inconclusive / SCREEN ONLY.
7. **Caveats that are not retracted by a positive result.** Budget confounds,
   mechanism confounds, single-seed limitations. This project's write-ups are
   trusted because they carry their own counter-evidence.
8. **What this closes and what it opens.**

## Rules

- **Report null and negative results with the same prominence as positive ones.**
  Most levers here come back null; the record's value is that it says so.
- **Never call a <5-seed result confirmed.** Write `SCREEN ONLY, N seeds` in the
  verdict line itself, not in a footnote.
- **Quote the unpaired |diff|/SE** from `/seedcheck` as the headline statistic —
  that's what every historical number in the log uses. Paired stats and drop-1
  ranges go in as supporting detail.
- **When a new result corrects an old section, edit the old section too** — add a
  correction note in place. Someone reading §46 must not be able to miss §47.
- **Convert relative dates to absolute.** "last night" is useless in six months.

## Commit and push

Per standing preference: push straight to `origin/next_phases`. **No PRs.**

```bash
git add -A && git commit -m "<what changed and what it means>" && git push origin next_phases
```

Commit messages state the finding, not the file operation — `Close out lora_adapter
as null at 3 seeds` beats `Update fidings`. Commit trailer per session config.

If working in a worktree, sync to `origin/next_phases` before editing shared docs
(the fidings file is append-heavy and conflicts easily); offline, merge the local
branch instead.
