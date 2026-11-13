#!/usr/bin/env python3
"""POST-HOC, NOT PRE-REGISTERED: what R1 gets wrong on the 827 offline validation rows (137 AndroidControl episodes).

    python3 val_errors.py <dir with eval-v1_s{0,1,2}-V-val.jsonl>      (files: <train host>:acv2/runs/eval/)

Teacher-forced: every row is predicted from the TRUE history, so an episode counted "fully right" needs every step right.
These are AndroidControl episodes, not AndroidWorld tasks: the comparison with online success is a plausibility check only.
"""
import collections
import json
import statistics
import sys

d = sys.argv[1]
D = [[json.loads(l) for l in open("%s/eval-v1_s%d-V-val.jsonl" % (d, i))] for i in range(3)]
N = len(D[0])
G = lambda r: r["gold"]["action_type"]  # noqa: E731
last = {}
for r in D[0]:
    last[r["episode_id"]] = max(last.get(r["episode_id"], 0), r["step"])
print("rows", N, "episodes", len(last), "mean step acc %.2f%%" % (100 * sum(r["correct"] for x in D for r in x) / (3 * N)))
print("\nper gold type (3 seeds pooled): n per seed, step acc, type acc, grounding among spatial rows")
by = collections.defaultdict(list)
for x in D:
    for r in x:
        by[G(r)].append(r)
for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    sp = [r for r in rs if r["grounding"] is not None]
    print("  %-14s n=%3d step %.1f%% type %.1f%% grounding %s" % (
        t, len(rs) // 3, 100 * sum(r["correct"] for r in rs) / len(rs), 100 * sum(r["pred"] == t for r in rs) / len(rs),
        "%.1f%%" % (100 * sum(1 for r in sp if r["grounding"]) / len(sp)) if sp else "-"))
c = collections.Counter((G(r), r["pred"]) for x in D for r in x if r["pred"] != G(r))
tot = sum(c.values())
print("\nwrong-type confusions gold -> predicted (share of all type errors):")
for (g, p), n in c.most_common(10):
    print("  %-13s -> %-13s %3d rows/seed (%.0f%%)" % (g, p, n / 3, 100 * n / tot))
wrong = [r for x in D for r in x if not r["correct"]]
typ = sum(r["pred"] != G(r) for r in wrong)
print("\nwrong rows %d: wrong type %.0f%%, right type wrong target/value %.0f%%" % (len(wrong), 100 * typ / len(wrong), 100 * (len(wrong) - typ) / len(wrong)))
print("  right type, wrong detail, by gold type (3 seeds):", dict(collections.Counter(G(r) for r in wrong if r["pred"] == G(r))))
print("\nstep accuracy by position in the episode:")
for name, f in (("first step", lambda r: r["step"] == 0), ("middle", lambda r: 0 < r["step"] < last[r["episode_id"]]),
                ("last step", lambda r: r["step"] == last[r["episode_id"]] and r["step"] > 0)):
    rs = [r for x in D for r in x if f(r)]
    print("  %-10s n=%3d acc %.1f%%" % (name, len(rs) // 3, 100 * sum(r["correct"] for r in rs) / len(rs)))
cons = [sum(D[s][i]["correct"] for s in range(3)) for i in range(N)]
print("\nrows right in 0/1/2/3 of the 3 seeds:", [cons.count(k) for k in range(4)])
print("mean confidence right %.2f, wrong %.2f" % (statistics.mean(r["conf"] for x in D for r in x if r["correct"]),
                                                  statistics.mean(r["conf"] for x in D for r in x if not r["correct"])))
for s, x in enumerate(D):
    ep = collections.defaultdict(list)
    for r in x:
        ep[r["episode_id"]].append(r)
    b = collections.defaultdict(lambda: [0, 0])
    for rs in ep.values():
        n = len(rs)
        k = "1-2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"
        b[k][0] += 1
        b[k][1] += all(r["correct"] for r in rs)
    print("seed %d episodes with EVERY step right: %d/%d %s" % (s, sum(v[1] for v in b.values()), len(ep),
                                                              {k: "%d/%d" % (v[1], v[0]) for k, v in sorted(b.items())}))
    print("        premature `status` rows (gold not status): %d" % sum(1 for r in x if r["pred"] == "status" and G(r) != "status"))
