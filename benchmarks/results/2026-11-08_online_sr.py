#!/usr/bin/env python3
"""Re-derive the online AndroidWorld numbers from the episode JSONLs in backup-2026-11-06/.

The paper's §5.7 reports, under protocol `aw-main-v1`, 8 tasks x 2 conditions at E2@938 with a 10-step
budget: **SR 1/8 in both conditions** (Wilson 2.2-47.1%), invalid-action 20/71 pure vs 10/68 declared-fallback,
and `n_reachable = 0` so `SR_all`/`SR_reachable` cannot be split.

This script re-derives all of that from disk. It is deliberately explicit about the denominator: a task that
ends in a checker exception on a live device is NOT a success and NOT a crash, and it is counted separately.
"""
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "backup-2026-11-06", "gate_runs", "aw_runs_20261106")


def wilson(k, n, z=1.96):
    """Wilson score interval — the one the paper quotes; Wald is invalid at k=1, n=8."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load_condition(name):
    """One condition = a directory of per-task episode JSONLs."""
    d = os.path.join(RUNS, name, "episodes")
    if not os.path.isdir(d):
        return None
    eps = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        last = rows[-1]
        # the report fields the harness writes
        eps.append({
            "task": os.path.basename(f).replace(".jsonl", ""),
            "steps": len(rows),
            "success": bool(last.get("success")) if last.get("success") is not None else None,
            "checker_error": last.get("checker_error"),
            "env_crash": bool(last.get("env_crash")),
            "invalid": sum(1 for r in rows if r.get("invalid_action")),
            "actions": sum(1 for r in rows if r.get("action_ok") is not None),
        })
    return eps


def report(name):
    eps = load_condition(name)
    if eps is None:
        print(f"  {name:10s}  MISSING (no episodes dir)")
        return None
    n = len(eps)
    succ = sum(1 for e in eps if e["success"] is True)
    crash = sum(1 for e in eps if e["env_crash"])
    chk = sum(1 for e in eps if e["checker_error"])
    inv = sum(e["invalid"] for e in eps)
    acts = sum(e["actions"] for e in eps)
    lo, hi = wilson(succ, n)
    print(f"  {name:10s}  n={n}  SR={succ}/{n} = {succ/n*100:.1f}%  "
          f"Wilson {lo*100:.1f}-{hi*100:.1f}%  invalid={inv}/{acts} ({inv/max(acts,1)*100:.1f}%)  "
          f"env_crash={crash}  checker_error={chk}")
    return {"n": n, "succ": succ, "invalid": inv, "actions": acts, "crash": crash, "checker_error": chk}


def main():
    print("§5.7 — online AndroidWorld, re-derived from episode JSONLs\n")
    print("protocol aw-main-v1: 8 tasks x 2 conditions, E2@938, 10-step budget\n")
    a = report("pure8")
    b = report("fallback8")
    print()
    if a and b:
        print("PAPER CLAIMS: SR 1/8 in BOTH conditions, Wilson 2.2-47.1%, "
              "invalid 20/71 pure vs 10/68 fallback")
        got = (a["succ"], a["n"], b["succ"], b["n"])
        print(f"ARTIFACTS   : SR {got[0]}/{got[1]} pure, {got[2]}/{got[3]} fallback")
        ok = got == (1, 8, 1, 8)
        print("SR REPRODUCES:", "YES" if ok else f"NO — artifacts say {got}, paper says (1,8,1,8)")
        lo, hi = wilson(1, 8)
        print(f"Wilson      : {lo*100:.1f}-{hi*100:.1f}%  (paper says 2.2-47.1)")
        print(f"invalid     : {a['invalid']}/{a['actions']} vs {b['invalid']}/{b['actions']}  "
              f"(paper says 20/71 vs 10/68)")
        # the honest caveat the paper must carry
        print("\nCAVEAT (must stay in the paper): the two conditions ran DIFFERENT task instances, "
              "so the invalid-action comparison is unmatched and is a hypothesis, not an effect; "
              "and n_reachable = 0, so SR_all cannot be split into reachable.")
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
