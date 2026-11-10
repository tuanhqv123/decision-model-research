#!/usr/bin/env python3
"""POST-HOC, NOT PRE-REGISTERED (written after paired_sr_s0.md was read): why the official success rate is low.

    python3 posthoc2.py        (from this directory; needs ../../aw on the path for the frozen paired_sr helpers)

1. FLOOR SPLIT. Q11 ('status' at step 0, no model) succeeds on some tasks whose end state already holds; any arm's
   successes split into those floor tasks and tasks beyond the floor.
2. SUCCESS BY STEP BUDGET (task-completion tasks only; the 25 answer-only tasks score 0 by construction).
3. TERMINATION: how often each arm declares done, and what a wrong 'done' looks like.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "aw"))
from paired_sr import ANSWER_ONLY_FILE, TASKS_FILE, _excluded, _names, _success, load_results  # noqa: E402

BUDGETS = os.path.join(HERE, "..", "..", "aw", "hostscripts", "aw116_budgets.json")


def main():
    q = load_results(os.path.join(HERE, "q11_status_now_results.jsonl"))
    arms = {"B0'": load_results(os.path.join(HERE, "b0prime_results.jsonl")),
            "R1 s0": load_results(os.path.join(HERE, "r1s0_results.jsonl"))}
    suite, ao = _names(TASKS_FILE), set(_names(ANSWER_ONLY_FILE))
    bud = {k: v["max_steps"] for k, v in json.load(open(BUDGETS)).items()}
    floor = sorted(t for t in suite if t in q and not _excluded(q[t]) and _success(q[t]))
    print("1. Floor split. Q11 successes (%d of %d scored): %s" % (len(floor), sum(not _excluded(q[t]) for t in suite if t in q), floor))
    for nm, a in arms.items():
        succ = [t for t in suite if t in a and not _excluded(a[t]) and _success(a[t])]
        print("   %-6s %2d successes = %d on floor tasks + %d beyond; floor tasks missed: %s" % (
            nm, len(succ), sum(t in floor for t in succ), sum(t not in floor for t in succ), [t for t in floor if t not in succ]))
    print("2. Success by step budget (task-completion tasks)")
    for nm, a in arms.items():
        cells = []
        for lo, hi in ((0, 10), (11, 20), (21, 40), (41, 999)):
            sel = [t for t in suite if t in a and t not in ao and not _excluded(a[t]) and lo <= bud[t] <= hi]
            cells.append("%d-%d: %d/%d" % (lo, hi, sum(_success(a[t]) for t in sel), len(sel)))
        print("   %-6s %s" % (nm, " | ".join(cells)))
    print("3. Termination (task-completion tasks)")
    for nm, a in arms.items():
        tc = [a[t] for t in suite if t in a and t not in ao and not _excluded(a[t])]
        done = [x for x in tc if x["declared_done"]]
        wrong = sorted(x["n_steps"] / bud[x["task_name"]] for x in done if not _success(x))
        print("   %-6s declared done %d of %d; right %d, wrong %d (median %.0f%% of the step budget used when wrong)" % (
            nm, len(done), len(tc), sum(_success(x) for x in done), len(wrong), 100 * wrong[len(wrong) // 2]))
        aod = [a[t] for t in suite if t in a and t in ao and not _excluded(a[t])]
        print("          answer-only: %d tasks, declared done %d, successes %d" % (len(aod), sum(x["declared_done"] for x in aod), sum(_success(x) for x in aod)))
    for nm in ("r1s0", "b0prime"):
        eps = json.load(open(os.path.join(HERE, nm + "_export.json")))
        for label, keep in (("wrong", lambda e: e["declared"] and not (e.get("score") or 0) > 0.5 and not e.get("setup_error")),
                            ("right", lambda e: e["declared"] and (e.get("score") or 0) > 0.5)):
            E = [e for e in eps if keep(e)]
            n = sorted(len(e["steps"]) - 1 for e in E)
            last = Counter(e["steps"][-2]["action"] if len(e["steps"]) > 1 else "(none)" for e in E)
            print("   %-8s %-5s done: %2d episodes, median acting steps before done %d, last action before done %s" % (
                nm, label, len(E), n[len(n) // 2], last.most_common(3)))


if __name__ == "__main__":
    main()
