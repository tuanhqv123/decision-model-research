#!/usr/bin/env python3
"""POST-HOC, NOT PRE-REGISTERED: where R1 s0's difference from B0' comes from. Written after the frozen analysis
(`paired_sr_s0.md`, `*_clustered_all.json`) was read; nothing here feeds a decision rule.

    python3 posthoc.py            (run from this directory; imports the frozen clustered.py unchanged)

1. Outcome buckets per arm (mutually exclusive): success (official score > 0.5) / premature done (declared done,
   not successful) / world right, never said done (checker would pass, no `status`) / lost or timeout / excluded.
2. The w and l tasks with what the other arm did, and the bucket transitions B0' -> R1.
3. First-encounter decisions (the frozen A2-bis definition) split into: repeat / `status` / a different action.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-11-10-b0-diagnosis"))
from clustered import same_action  # noqa: E402


def results(name):
    with open(os.path.join(HERE, name)) as fp:
        return {r["task_name"]: r for r in map(json.loads, fp)}


def bucket(r):
    if r.get("setup_error") or r.get("unscored") or r.get("env_crash") or r.get("score") is None:
        return "excluded"
    if (r["score"] or 0) > 0.5:
        return "success"
    if r["declared_done"]:
        return "premature_done"
    if (r.get("task_score") or 0) > 0.5:
        return "world_right_never_done"
    return "lost_or_timeout"


def first_encounters(export):
    with open(os.path.join(HERE, export)) as fp:
        eps = [e for e in json.load(fp) if not e.get("setup_error") and not e.get("unscored")]
    c = Counter()
    for e in eps:
        steps, prev_same = e["steps"], False
        for i in range(len(steps) - 1):  # same walk as clustered.episode_stats
            s, n = steps[i], steps[i + 1]
            if s["next_diff"] is None:
                prev_same = False
                continue
            same = s["next_diff"] < 0.002
            if same and not prev_same:
                c["repeat" if same_action(s, n) else ("status" if n["action"] == "status" else "different action")] += 1
            prev_same = same and same_action(s, n)
    acts = Counter(s["action"] for e in eps for s in e["steps"])
    return c, acts


def main():
    r1, b0 = results("r1s0_results.jsonl"), results("b0prime_results.jsonl")
    print("1. Outcome buckets")
    for name, arm in (("R1 s0", r1), ("B0'", b0)):
        c = Counter(bucket(r) for r in arm.values())
        done = c["success"] + c["premature_done"]
        steps = sorted(r["n_steps"] for r in arm.values())
        print("  %-6s %s | done declared %d, of them successful %d (%.1f%%) | median steps %d" % (
            name, dict(sorted(c.items())), done, c["success"], 100.0 * c["success"] / max(1, done),
            steps[len(steps) // 2]))
    print("2. Paired tasks")
    w = sorted(t for t in r1 if bucket(r1[t]) == "success" and bucket(b0[t]) not in ("success", "excluded"))
    l = sorted(t for t in r1 if bucket(b0[t]) == "success" and bucket(r1[t]) not in ("success", "excluded"))
    for t in w:
        print("  w %-42s B0' %-24s steps B0' %3d  R1 %3d" % (t, bucket(b0[t]), b0[t]["n_steps"], r1[t]["n_steps"]))
    for t in l:
        print("  l %-42s R1  %-24s steps R1  %3d  B0' %3d" % (t, bucket(r1[t]), r1[t]["n_steps"], b0[t]["n_steps"]))
    tr = Counter((bucket(b0[t]), bucket(r1[t])) for t in r1)
    print("  transitions B0' -> R1:", {"%s -> %s" % k: v for k, v in sorted(tr.items()) if k[0] != k[1]})
    print("3. First encounters of an unchanged screen")
    for name, export in (("R1 s0", "r1s0_export.json"), ("B0'", "b0prime_export.json")):
        c, acts = first_encounters(export)
        tot = sum(c.values())
        print("  %-6s n=%d: " % (name, tot) + ", ".join("%s %d (%.1f%%)" % (k, c[k], 100.0 * c[k] / tot)
                                                      for k in ("repeat", "status", "different action")))
        print("         action mix: %s" % dict(acts.most_common()))


if __name__ == "__main__":
    main()
