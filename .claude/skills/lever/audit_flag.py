#!/usr/bin/env python3
"""Audit whether a training flag actually reaches everywhere it must.

This project has been burned three times by a flag that parsed fine and did
nothing (``--disable_head_fix`` sec 10, ``fixed_ts`` sec 24, ``--lora_adapter``'s
missing global_model template entry). Each one silently invalidated real runs.
The chain is long and partly POSITIONAL, so a gap is invisible at runtime.

Static-analysis only (``ast``): no SUMO, no torch, no training. Run it after
adding a flag and before spending any seed budget on it.

Usage:
    python .claude/skills/lever/audit_flag.py                # audit every known flag
    python .claude/skills/lever/audit_flag.py --flag lora_adapter
    python .claude/skills/lever/audit_flag.py --align        # worker arg alignment only
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FT = os.path.join(REPO, "experiments", "federated_training.py")
PS = os.path.join(REPO, "federated", "parallel_server.py")
DQN = os.path.join(REPO, "agents", "dqn.py")
NET = os.path.join(REPO, "agents", "networks.py")
WIRING_TEST = os.path.join(REPO, "tests", "test_flag_wiring.py")

# Positional args whose name legitimately differs from the parameter they fill.
# Anything NOT in here that mismatches is a possible shift, which is the real
# hazard -- a warning that always fires is a warning nobody reads.
KNOWN_RENAMES = {
    ("lr", "city_lr"),
    ("seed", "city_seed"),
    ("in_queue", "self.in_queues[name]"),
}


def parse(path: str) -> ast.Module:
    with open(path, errors="replace") as fh:
        return ast.parse(fh.read(), filename=path)


def func_def(tree: ast.AST, name: str, cls: Optional[str] = None) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and cls and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
        if cls is None and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def param_names(fn: Optional[ast.FunctionDef]) -> List[str]:
    if fn is None:
        return []
    return [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]


def calls_to(tree: ast.AST, name: str) -> List[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        target = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if target == name:
            out.append(node)
    return out


def kwargs_of(call: ast.Call) -> Set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def assigned_attrs(fn: Optional[ast.FunctionDef]) -> Set[str]:
    """self.X = ... inside a function."""
    out: Set[str] = set()
    for node in ast.walk(fn) if fn else []:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    out.add(t.attr)
    return out


def unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is total in practice
        return "<?>"


def worker_args_tuple(ps_tree: ast.AST) -> Optional[ast.Tuple]:
    """The positional args=(...) tuple handed to ctx.Process(target=_client_worker)."""
    for call in calls_to(ps_tree, "Process"):
        if not any(kw.arg == "target" and getattr(kw.value, "id", None) == "_client_worker"
                   for kw in call.keywords):
            continue
        for kw in call.keywords:
            if kw.arg == "args" and isinstance(kw.value, ast.Tuple):
                return kw.value
    return None


# --------------------------------------------------------------------------
# One index pass, shared by every flag
# --------------------------------------------------------------------------
@dataclass
class Index:
    """Everything that does not depend on which flag is being audited.

    Built once. Without this the per-flag loop re-walks the same four trees
    ~12 times per flag (~330 walks / ~840k node visits for 26 flags).
    """
    cli_options: List[str] = field(default_factory=list)
    make_agent_params: List[str] = field(default_factory=list)
    make_agent_to_dqn: Set[str] = field(default_factory=set)
    make_agent_sites: List[Tuple[int, Set[str]]] = field(default_factory=list)
    server_kwargs: Dict[str, Optional[Set[str]]] = field(default_factory=dict)
    worker_params: List[str] = field(default_factory=list)
    server_init_params: List[str] = field(default_factory=list)
    server_attrs: Set[str] = field(default_factory=set)
    worker_tuple_exprs: List[str] = field(default_factory=list)
    dqn_params: List[str] = field(default_factory=list)
    net_names: Set[str] = field(default_factory=set)


def build_index(trees: Dict[str, ast.Module]) -> Index:
    ft, ps, dqn, net = trees["ft"], trees["ps"], trees["dqn"], trees["net"]
    idx = Index()

    idx.cli_options = [
        a.value[2:]
        for call in calls_to(ft, "add_argument")
        for a in call.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")
    ]

    mk = func_def(ft, "_make_agent")
    idx.make_agent_params = param_names(mk)
    if mk:
        for call in calls_to(mk, "DQNAgent"):
            idx.make_agent_to_dqn |= kwargs_of(call)
    idx.make_agent_sites = [(c.lineno, kwargs_of(c)) for c in calls_to(ft, "_make_agent")]
    for server in ("ParallelFederatedServer", "FederatedServer"):
        cs = calls_to(ft, server)
        idx.server_kwargs[server] = set().union(*(kwargs_of(c) for c in cs)) if cs else None

    idx.worker_params = param_names(func_def(ps, "_client_worker"))
    init = func_def(ps, "__init__", cls="ParallelFederatedServer")
    idx.server_init_params = param_names(init)
    idx.server_attrs = assigned_attrs(init)
    tup = worker_args_tuple(ps)
    idx.worker_tuple_exprs = [unparse(e) for e in tup.elts] if tup else []

    idx.dqn_params = param_names(func_def(dqn, "__init__", cls="DQNAgent"))
    idx.net_names = {
        n.attr if isinstance(n, ast.Attribute) else n.arg
        for n in ast.walk(net)
        if isinstance(n, (ast.Attribute, ast.arg))
    }
    return idx


def check_alignment(idx: Index) -> Tuple[List[str], List[str]]:
    """Positionally align the Process args tuple against _client_worker's params.

    The tuple is passed positionally, so inserting a parameter anywhere but the
    end silently shifts every later flag onto the wrong parameter. Nothing at
    runtime catches this -- most of these flags are floats/bools that will
    happily accept a wrong value.
    """
    errors: List[str] = []
    suspicious: List[str] = []
    params, elts = idx.worker_params, idx.worker_tuple_exprs
    if not params or not elts:
        return ["could not locate _client_worker and/or its ctx.Process args tuple"], []
    if len(elts) > len(params):
        errors.append(f"Process passes {len(elts)} positional args but _client_worker takes {len(params)}")
    for i, (expr, p) in enumerate(zip(elts, params)):
        norm = expr.split("[")[0].replace("self.", "").strip()
        if norm == p or (p, expr) in KNOWN_RENAMES:
            continue
        suspicious.append(f"  pos {i:>2}: param '{p}'  <-  '{expr}'")
    return errors, suspicious


def match_cli_option(flag: str, options: Sequence[str]) -> Optional[str]:
    """Map a _make_agent param name to its CLI option.

    Exact first, then a WORD-BOUNDARY relation (--fedprox_mu <-> mu,
    --batchnorm <-> use_batchnorm). A bare substring test is wrong here: it
    matched 'mu' to '--munchausen_temp', reporting a genuinely-unwired flag
    as OK -- the exact failure this tool exists to catch.
    """
    if flag in options:
        return flag
    for opt in options:
        if opt.endswith("_" + flag) or flag.endswith("_" + opt):
            return opt
    return None


def audit_flag(flag: str, idx: Index) -> List[Tuple[str, bool, str]]:
    """Returns [(touchpoint, ok, detail), ...] for one flag."""
    rows: List[Tuple[str, bool, str]] = []

    opt = match_cli_option(flag, idx.cli_options)
    rows.append((f"federated_training: --{flag} in argparse", opt is not None,
                 "" if opt == flag else (f"matched --{opt}" if opt else "")))
    rows.append(("federated_training: _make_agent param", flag in idx.make_agent_params, ""))
    rows.append(("federated_training: _make_agent -> DQNAgent(...)", flag in idx.make_agent_to_dqn, ""))
    for n, (lineno, kws) in enumerate(idx.make_agent_sites, start=1):
        rows.append((f"federated_training: _make_agent call site #{n} (line {lineno})", flag in kws, ""))
    for server, kws in idx.server_kwargs.items():
        rows.append((f"federated_training: {server}(...)", bool(kws) and flag in kws,
                     "" if kws is not None else "no call site found"))

    rows.append(("parallel_server: _client_worker param", flag in idx.worker_params, ""))
    rows.append(("parallel_server: server __init__ param", flag in idx.server_init_params, ""))
    # Spawned positionally either as self.<flag> (stored on the server) or as a
    # bare __init__ local -- both are live; only absence is a gap.
    in_tuple = any(e in (f"self.{flag}", flag) for e in idx.worker_tuple_exprs)
    stored = flag in idx.server_attrs
    rows.append(("parallel_server: in Process args tuple", in_tuple,
                 "POSITIONAL -- order matters" + ("" if stored else "; passed as __init__ local")))

    rows.append(("agents/dqn: DQNAgent.__init__ param", flag in idx.dqn_params, ""))
    rows.append(("agents/networks: referenced", flag in idx.net_names,
                 "only required if the flag changes the network"))
    return rows


def known_flags(idx: Index) -> List[str]:
    skip = {"own_dim", "neighbor_dim", "k_max", "action_dim", "eps_decay", "head_fix", "algo"}
    return [p for p in idx.make_agent_params if p not in skip]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flag", help="single flag to audit (default: every _make_agent knob)")
    ap.add_argument("--align", action="store_true", help="only run the worker-arg alignment check")
    ap.add_argument("--quiet-ok", action="store_true", help="only print flags with gaps")
    args = ap.parse_args()

    idx = build_index({"ft": parse(FT), "ps": parse(PS), "dqn": parse(DQN), "net": parse(NET)})

    print("=" * 78)
    print("POSITIONAL ALIGNMENT: ctx.Process(args=...) vs _client_worker signature")
    print("=" * 78)
    errors, suspicious = check_alignment(idx)
    for e in errors:
        print(f"  ERROR: {e}")
    if suspicious:
        print("  Unexpected name mismatches -- a SHIFT would look exactly like this:")
        for r in suspicious:
            print(r)
    if not errors and not suspicious:
        print(f"  {len(idx.worker_tuple_exprs)} positional args align with their parameters")
        print(f"  ({len(KNOWN_RENAMES)} known intentional renames skipped)")
    if args.align:
        return 1 if errors else 0

    flags = [args.flag] if args.flag else known_flags(idx)
    print("\n" + "=" * 78)
    print(f"FLAG WIRING: {len(flags)} flag(s)")
    print("=" * 78)

    gaps_total = 0
    for flag in flags:
        rows = audit_flag(flag, idx)
        gaps = [r for r in rows if not r[1]]
        gaps_total += len(gaps)
        if args.quiet_ok and not gaps:
            continue
        print(f"\n--{flag}  [{'OK' if not gaps else f'{len(gaps)} GAP(S)'}]")
        for label, ok, detail in rows:
            if ok and args.quiet_ok:
                continue
            print(f"  {'ok  ' if ok else 'MISS'}  {label}{f'   ({detail})' if detail else ''}")

    covered: Set[str] = set()
    if os.path.exists(WIRING_TEST):
        with open(WIRING_TEST, errors="replace") as fh:
            body = fh.read()
        covered = {f for f in flags if f in body}
    print("\n" + "=" * 78)
    print(f"tests/test_flag_wiring.py mentions {len(covered)}/{len(flags)} of these flags")
    missing = [f for f in flags if f not in covered]
    if missing:
        print("  not mentioned: " + ", ".join(missing))
    print("=" * 78)
    print(f"\n{gaps_total} total gap(s). A gap is not automatically a bug -- a flag that only "
          "affects\nthe DQN path legitimately skips PPO/QR-DQN, and a loss-only flag skips "
          "networks.py.\nBut every gap must be explained before the flag is trusted in a run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
