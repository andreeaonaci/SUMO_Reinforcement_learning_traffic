---
name: lever
description: Implement a new training-time intervention (a CLI flag / lever) end-to-end through this repo's long, partly-positional plumbing chain, then verify it actually reaches the network before spending seed budget. Use when adding any new --flag to federated_training, changing DQNAgent/network behavior behind a switch, or auditing whether an existing flag is wired correctly.
---

# lever — add an intervention without it silently doing nothing

Three flags in this project's history parsed fine and did nothing:
`--disable_head_fix` (§10), `fixed_ts` (§24), `--lora_adapter`'s missing
`global_model` template entry. Each silently invalidated real runs — days of
ablations that were measuring nothing. The chain is long, spans four files, and
one leg of it is **positional**.

## Step 1 — audit first

```bash
python .claude/skills/lever/audit_flag.py --flag <name>   # one flag
python .claude/skills/lever/audit_flag.py                 # all knobs
python .claude/skills/lever/audit_flag.py --align         # positional check only
```

Static `ast` analysis — no SUMO, no torch, runs in a second. Do this before
writing code (to see a comparable flag's shape) and again after.

## Step 2 — the chain

For a flag affecting the **DQN + `--parallel`** path (the path every real run
uses), all of these must be touched:

**`experiments/federated_training.py`**
1. `parser.add_argument("--name", ...)` — help text states the exact no-op default.
2. `_make_agent(...)` signature — new keyword with a **no-op default**.
3. `_make_agent` body — forward into `DQNAgent(...)` (and only the agent classes
   the flag actually applies to; skipping PPO/QR-DQN is fine and intentional).
4. `global_model = _make_agent(...)` in the **`args.parallel` branch**.
   **This is the one that bit `--lora_adapter`.** This agent's `state_dict`
   becomes the round-0 broadcast that every worker's `load_state_dict(strict=True)`
   must match key-for-key — so if the flag adds or removes parameters, omitting
   it here crashes every worker on round 1.
5. `ParallelFederatedServer(...)` call — pass `name=args.name`.

   Find the current line numbers rather than trusting any written here — the
   auditor prints them, and so does:
   ```bash
   grep -n "_make_agent(\|ParallelFederatedServer(\|FederatedServer(" experiments/federated_training.py
   ```

**`federated/parallel_server.py`**
6. `ParallelFederatedServer.__init__` signature + `self.name = name`.
7. `_client_worker(...)` signature — keyword with the same no-op default.
8. The `ctx.Process(target=_client_worker, args=(...))` tuple. **This is
   positional.** Append at the end; inserting anywhere else silently shifts every
   later flag onto the wrong parameter, and nothing at runtime will notice —
   most of these are floats and bools that happily accept a wrong value.
   `audit_flag.py --align` checks this.

**`agents/dqn.py` / `agents/networks.py`**
9. `DQNAgent.__init__` param → forward to the network; network param + use.
   A loss-only lever (like `--cql_weight`) legitimately never touches
   `networks.py`.

**`tests/test_flag_wiring.py`**
10. Add an assertion. This file exists precisely because two of these bugs were
    caught by luck instead of tests.

### Known standing gap — the sequential path
`audit_flag.py` reports 3 gaps for nearly every lever added since ~§78: the
non-`--parallel` code path (its own `_make_agent` call site, the `make_agent`
closure inside the client builder, and `FederatedServer(...)`) never received
them. Real runs all use `--parallel`, so **no published result is affected** —
but a lever run without `--parallel` is silently inert, and ~60 of the auditor's
~68 reported gaps are this one defect repeated per flag.

**The right fix is one guard, not 20 threadings:** in `federated_training.py`,
if `not args.parallel` and any lever flag is off its no-op default, raise. That
closes the silent-inertness hazard and collapses the gap report to the handful
that mean something. Until that exists, the auditor is annotating a bug rather
than the codebase preventing it.

## Step 3 — no-op default is mandatory

The default value must make the code path **byte-identical** to before. Every
lever in this repo follows this (`bounded_q=False`, `cql_weight=0.0`,
`lora_adapter=False`). It is what makes the baseline arm of a comparison a
genuine control rather than a second treatment.

## Step 4 — smoke test before seed budget

```bash
export SUMO_HOME=/usr/share/sumo && export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
python -m experiments.federated_training --parallel --rounds 1 --local_episodes 1 \
    --base_dir environments_smoketest --<name> <value>
```

Then verify from the run's own `training.log`:
- the args dump at the top shows your flag at the intended value;
- the run reached round 1 and exited 0;
- **the flag visibly changed something** — a log line, a parameter count, a loss
  scale. "It didn't crash" is not evidence it did anything. Log a one-line
  confirmation from inside the code path if there is no other observable.

Also run `pytest tests/test_flag_wiring.py -q`.

## Step 5 — then, and only then, spend compute

Launch the 3-seed screen with `/launch`, analyze with `/seedcheck`, write up with
`/logfinding`. Remember what the record says: a 3-seed result is a screen, and
most levers in this project come back null. Design the write-up to be honest
about that before seeing the numbers.
