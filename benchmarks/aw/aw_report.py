#!/usr/bin/env python3
"""AndroidWorld results by the official difficulty labels, with average steps (2026-11-12).

Published tables (AndroidWorld paper, AndroidGen Table 9, ...) split success by easy / medium / hard. The labels and
optimal step counts come from upstream `android_world/task_metadata.json` (copied verbatim to
references/raw/androidworld-task_metadata.json: 61 easy, 36 medium, 19 hard).

Per difficulty: scored tasks n, score sum (official: partial credit counts, `score >= 1` is a full success), SR =
score/n, average steps taken (all, successful, failed) and the average optimal steps. Unscored or setup-error
episodes are excluded from n and counted separately, never as failures.

    python3 aw_report.py RUN_DIR[,RUN_DIR...] [--meta META.json] [--vs OTHER_RUN_DIR[,...]]
Also: SR by MobileRL complexity level (ceil(complexity), 4 = 4+) and, with --vs, the step win rate between two runs
(a win = solving in fewer steps; MobileRL Fig. 5 categories).
"""
import glob
import json
import math
import os
import sys

DIFFS = ("easy", "medium", "hard")
META_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "references", "raw",
                            "androidworld-task_metadata.json")


def load(dirs):
    """{task: result row} from the last `result` row of each non-abandoned episode file."""
    res = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "episodes", "*.jsonl"))):
            if ".abandoned." in f:
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            r = [x for x in rows if x.get("type") == "result"]
            if r:
                res[r[-1]["task_name"]] = r[-1]
    return res


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def table(res, meta):
    m = {t["task_name"]: t for t in meta}
    out = {}
    for group in DIFFS + ("all",):
        names = [n for n in m if group == "all" or m[n]["difficulty"] == group]
        scored = [n for n in names if n in res and not res[n].get("unscored") and not res[n].get("setup_error")
                  and res[n].get("score") is not None]
        score = sum(float(res[n]["score"]) for n in scored)
        steps = [res[n].get("n_steps") or 0 for n in scored]
        succ = [res[n].get("n_steps") or 0 for n in scored if float(res[n]["score"]) >= 1.0]
        fail = [res[n].get("n_steps") or 0 for n in scored if float(res[n]["score"]) < 1.0]
        out[group] = {"tasks": len(names), "n": len(scored), "excluded": len([n for n in names if n in res]) - len(scored),
                      "missing": len([n for n in names if n not in res]), "score": score,
                      "full_success": len(succ), "sr": 100.0 * score / len(scored) if scored else None,
                      "avg_steps": _mean(steps), "avg_steps_success": _mean(succ), "avg_steps_failure": _mean(fail),
                      "avg_optimal": _mean([float(m[n]["optimal_steps"]) for n in scored])}
    return out


BUDGETS_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hostscripts", "aw116_budgets.json")


def level(complexity):
    """MobileRL's complexity level (Fig. 4/5): rounded-up complexity, 4 = 4 and above."""
    return min(4, int(math.ceil(float(complexity))))


def _ok(r):
    return r is not None and not r.get("unscored") and not r.get("setup_error") and r.get("score") is not None


def step_wins(a, b, budgets):
    """Run a vs run b, per task scored in both: a 'win' solves the task (score >= 1) in fewer steps; if only one run
    solved it, that run wins; both wrong is a tie. Categories as MobileRL Fig. 5: All, C1-C4, BC, BW, Others."""
    cats = {k: {"n": 0, "a_win": 0, "tie": 0, "b_win": 0} for k in ("All", "C1", "C2", "C3", "C4", "BC", "BW", "Others")}
    for t in sorted(set(a) & set(b)):
        if not (_ok(a[t]) and _ok(b[t])) or t not in budgets:
            continue
        sa, sb = float(a[t]["score"]) >= 1.0, float(b[t]["score"]) >= 1.0
        if sa and sb:
            na, nb = a[t].get("n_steps") or 0, b[t].get("n_steps") or 0
            out = "a_win" if na < nb else ("b_win" if nb < na else "tie")
        else:
            out = "a_win" if sa else ("b_win" if sb else "tie")
        group = "BC" if (sa and sb) else ("BW" if not (sa or sb) else "Others")
        for k in ("All", "C%d" % level(budgets[t]["complexity"]), group):
            cats[k]["n"] += 1
            cats[k][out] += 1
    return cats


def fmt(t):
    f = lambda v, p="%.1f": "-" if v is None else p % v  # noqa: E731
    lines = ["| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | "
             "avg optimal steps | excluded |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in DIFFS + ("all",):
        r = t[g]
        lines.append("| %s | %d/%d | %s | %d | %s | %s | %s | %s | %s | %d |" % (
            g, r["n"], r["tasks"], f(r["score"], "%.1f"), r["full_success"], f(r["sr"]), f(r["avg_steps"]),
            f(r["avg_steps_success"]), f(r["avg_steps_failure"]), f(r["avg_optimal"]), r["excluded"]))
    return "\n".join(lines)


def level_table(res, budgets):
    meta = [{"task_name": t, "difficulty": "C%d" % level(v["complexity"]), "optimal_steps": 0} for t, v in budgets.items()]
    out = {}
    for g in ("C1", "C2", "C3", "C4"):
        sub = [m for m in meta if m["difficulty"] == g]
        r = table(res, [dict(m, difficulty="easy") for m in sub])["easy"]
        out[g] = r
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    meta_path = a[a.index("--meta") + 1] if "--meta" in a else META_DEFAULT
    budgets = json.load(open(BUDGETS_DEFAULT))
    res = load(a[0].split(","))
    print(fmt(table(res, json.load(open(meta_path)))))
    print()
    print("| Complexity level (ceil) | scored n | score | SR % | avg steps | avg steps (success) |")
    print("|---|---|---|---|---|---|")
    for g, r in level_table(res, budgets).items():
        f = lambda v: "-" if v is None else "%.1f" % v  # noqa: E731
        print("| %s | %d/%d | %.1f | %s | %s | %s |" % (g, r["n"], r["tasks"], r["score"], f(r["sr"]), f(r["avg_steps"]),
                                                    f(r["avg_steps_success"])))
    if "--vs" in a:
        other = load(a[a.index("--vs") + 1].split(","))
        print()
        print("| Category | n | this run wins (fewer steps) | tie | other run wins |")
        print("|---|---|---|---|---|")
        for k, v in step_wins(res, other, budgets).items():
            print("| %s | %d | %d | %d | %d |" % (k, v["n"], v["a_win"], v["tie"], v["b_win"]))
