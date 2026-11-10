#!/usr/bin/env python3
"""R1 vs B0' — the pre-registered paired comparison (PRE-REG-2026-11-10-contract-v1 §4, amendments A1 and A3).

    python3 paired_sr.py <r1_results.jsonl> <b0prime_results.jsonl> [--seed-label s0]

Each input holds one `result` line per episode (the last `type == "result"` line of every scored
`episodes/*.jsonl`, abandoned files excluded). Frozen BEFORE any R1 online outcome was looked at.

What it reports, all fixed in the pre-registration:
  * official SR per arm (`awenv.metrics.official_sr`), with its excluded list, the task-completion (TC) vs
    answer-only split (25 answer-only tasks score 0 by construction: no `answer` action), and a Wilson 95%
    interval on the number of fully successful tasks;
  * the paired counts on tasks scored in BOTH arms: w = R1 succeeds and B0' fails, l = the reverse, where
    success = official score > 0.5 (upstream prints "Task Successful" iff agent_successful > 0.5);
  * the exact two-sided sign test on w vs l (the §4 claim needs p < 0.05 pooled over three seeds AND all
    three seeds moving SR the same way — one seed can never satisfy it);
  * amendment A1's gate: seeds 1 and 2 go online only if |w - l| >= 6;
  * amendment A3's sensitivity: the same on the 104 tasks outside the 12 that `_slug` hid (H11). If the
    primary and the sensitivity set disagree on any decision, both are reported and nothing is claimed.
"""
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from awenv.metrics import official_sr  # noqa: E402

TASKS_FILE = os.path.join(HERE, "hostscripts", "aw116_tasks.txt")
ANSWER_ONLY_FILE = os.path.join(HERE, "hostscripts", "aw116_answer_only.txt")
# H11: the tasks `_slug`'s 24-character cut hid from every 116-task run before 2026-11-10.
SHADOWED = {
    "AudioRecorderRecordAudioWithFileName", "RecipeAddMultipleRecipesFromImage", "RecipeAddMultipleRecipesFromMarkor",
    "RecipeAddMultipleRecipesFromMarkor2", "RecipeDeleteDuplicateRecipes2", "RecipeDeleteDuplicateRecipes3",
    "RecipeDeleteMultipleRecipesWithConstraint", "RecipeDeleteMultipleRecipesWithNoise",
    "SimpleCalendarAddOneEventInTwoWeeks", "SimpleCalendarAddOneEventRelativeDay", "SimpleCalendarAddOneEventTomorrow",
    "SimpleCalendarDeleteEventsOnRelativeDay",
}


def _names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip() and not line.startswith("#")]


def load_results(path: str) -> Dict[str, Dict[str, Any]]:
    """task_name -> its result line. A task appearing twice is an error, never silently resolved."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            name = r["task_name"]
            if name in out:
                raise ValueError("task %s has two result lines in %s" % (name, path))
            out[name] = r
    return out


def _excluded(r: Dict[str, Any]) -> bool:
    return bool(r.get("setup_error") or r.get("unscored") or r.get("env_crash")) or r.get("score") is None


def _success(r: Dict[str, Any]) -> bool:
    return (r.get("score") or 0.0) > 0.5


def sign_test_two_sided(w: int, l: int) -> float:
    """Exact two-sided binomial sign test of w vs l under p = 1/2 (ties dropped)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def a1_runs_more_seeds(w: int, l: int) -> bool:
    """Amendment A1: seeds 1-2 are run online only if seed 0's paired difference is >= 6 tasks either way."""
    return abs(w - l) >= 6


def paired_counts(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]], tasks: Iterable[str]) -> Dict[str, Any]:
    w = l = both_s = both_f = 0
    excluded, missing = [], []
    for t in tasks:
        if t not in a or t not in b:
            missing.append(t)
            continue
        if _excluded(a[t]) or _excluded(b[t]):
            excluded.append(t)
            continue
        sa, sb = _success(a[t]), _success(b[t])
        w += sa and not sb
        l += sb and not sa
        both_s += sa and sb
        both_f += (not sa) and (not sb)
    return {"w": w, "l": l, "both_success": both_s, "both_fail": both_f,
            "n_both_scored": w + l + both_s + both_f, "excluded": sorted(excluded), "missing": sorted(missing)}


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def arm_summary(res: Dict[str, Dict[str, Any]], tasks: List[str], answer_only: set) -> Dict[str, Any]:
    eps = [res[t] for t in tasks if t in res]
    sr = official_sr(eps)
    full = sum(1 for r in eps if not _excluded(r) and _success(r))
    n = sr["n_tasks_scored"]
    tc = official_sr([res[t] for t in tasks if t in res and t not in answer_only])
    ao = official_sr([res[t] for t in tasks if t in res and t in answer_only])
    return {"sr": sr["sr"], "n_scored": n, "excluded": sr["excluded"], "missing": [t for t in tasks if t not in res],
            "full_successes": full, "wilson": wilson(full, n), "sr_tc": tc["sr"], "n_tc": tc["n_tasks_scored"],
            "sr_answer_only": ao["sr"], "n_answer_only": ao["n_tasks_scored"]}


def _fmt_arm(name: str, s: Dict[str, Any]) -> str:
    lo, hi = s["wilson"]
    pct = lambda v: "n/a" if v is None else "%.1f%%" % (100 * v)  # noqa: E731
    return ("| %s | %s | %d/%d | [%s, %s] | %s (n=%d) | %s (n=%d) | %s | %s |"
            % (name, pct(s["sr"]), s["full_successes"], s["n_scored"], pct(lo), pct(hi), pct(s["sr_tc"]), s["n_tc"],
               pct(s["sr_answer_only"]), s["n_answer_only"], ", ".join(s["excluded"]) or "-",
               ", ".join(s["missing"]) or "-"))


def analyse(r1: Dict[str, Dict[str, Any]], b0: Dict[str, Dict[str, Any]], label: str = "s0") -> Dict[str, Any]:
    suite = _names(TASKS_FILE)
    answer_only = set(_names(ANSWER_ONLY_FILE))
    assert len(suite) == 116 and len(answer_only) == 25 and SHADOWED <= set(suite)
    sets = {"primary (116)": suite, "sensitivity A3 (104, shadowed 12 excluded)": [t for t in suite if t not in SHADOWED]}
    out = {"label": label, "sets": {}}
    for set_name, tasks in sets.items():
        p = paired_counts(r1, b0, tasks)
        p["p_sign"] = sign_test_two_sided(p["w"], p["l"])
        p["a1_more_seeds"] = a1_runs_more_seeds(p["w"], p["l"])
        out["sets"][set_name] = {"paired": p, "r1": arm_summary(r1, tasks, answer_only),
                                 "b0prime": arm_summary(b0, tasks, answer_only)}
    decisions = {k: v["paired"]["a1_more_seeds"] for k, v in out["sets"].items()}
    out["a1_decisions_agree"] = len(set(decisions.values())) == 1
    return out


def render(out: Dict[str, Any]) -> str:
    lines = ["# R1 (%s) vs B0′ — pre-registered paired comparison" % out["label"], ""]
    for set_name, s in out["sets"].items():
        p = s["paired"]
        lines += ["## %s" % set_name, "",
                  "| arm | official SR | full successes | Wilson 95% | SR TC | SR answer-only | excluded | missing |",
                  "|---|---|---|---|---|---|---|---|",
                  _fmt_arm("R1 " + out["label"], s["r1"]), _fmt_arm("B0′", s["b0prime"]), "",
                  "Paired on %d tasks scored in both: **w = %d** (R1 only), **l = %d** (B0′ only), both succeed %d, "
                  "both fail %d. Exact two-sided sign test p = %.4f. Excluded in either arm: %s. Missing: %s."
                  % (p["n_both_scored"], p["w"], p["l"], p["both_success"], p["both_fail"], p["p_sign"],
                     ", ".join(p["excluded"]) or "-", ", ".join(p["missing"]) or "-"),
                  "A1: |w − l| = %d → seeds 1–2 online: **%s**." % (abs(p["w"] - p["l"]),
                                                                    "YES" if p["a1_more_seeds"] else "NO"), ""]
    lines.append("A1 decision identical on the primary and the A3 sensitivity set: **%s**%s" % (
        "yes" if out["a1_decisions_agree"] else "NO",
        "" if out["a1_decisions_agree"] else " — per A3, both are reported and nothing is claimed."))
    lines.append("")
    lines.append("§4 claim rule needs three seeds (pooled p < 0.05 and all seeds moving SR the same way); "
                 "one seed can only feed A1. Wording if no claim: \"no measurable difference at n = 116\", "
                 "never \"no effect\".")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("r1")
    ap.add_argument("b0prime")
    ap.add_argument("--seed-label", default="s0")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    out = analyse(load_results(args.r1), load_results(args.b0prime), args.seed_label)
    print(json.dumps(out, indent=2, sort_keys=True, default=str) if args.json else render(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
