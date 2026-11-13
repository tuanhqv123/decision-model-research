"""Every AndroidWorld number of the v3 post-mortem, recomputed from the episode files (2026-11-11).
    python aw_facts.py benchmarks/results/2026-11-11-v3/androidworld"""
import collections as C, glob, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../aw"))
from awenv.metrics import official_sr  # the repo's implementation of AndroidWorld's suite_utils SR

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)

eps = []
for f in sorted(glob.glob(sys.argv[1] + "/aw116_v3_s0*/episodes/*.jsonl")):
    if ".abandoned." in f: continue
    L = [json.loads(l) for l in open(f)]
    head = next(x for x in L if x["type"] == "episode"); r = [x for x in L if x["type"] == "result"][-1]
    st = sorted([x for x in L if x["type"] == "step"], key=lambda s: s["i"])
    eps.append({"result": r, "task": r["task_name"], "ok": bool(r.get("success")), "unscored": bool(r.get("unscored") or r.get("checker_unavailable")),
                "done": bool(r.get("declared_done")), "budget": head["step_budget"], "steps": st,
                "lines": [s["history_line"] for s in st if s.get("history_line")]})
out = C.OrderedDict()
n = len(eps); k = sum(e["ok"] for e in eps)
out["tasks"] = n; out["successes"] = k
osr = official_sr([e["result"] for e in eps])
out["official_sr (repo awenv.metrics.official_sr)"] = osr
out["wilson95 over 116"] = "%.1f-%.1f" % wilson(k, n)
ans = [e for e in eps if e["task"] in set(open(sys.argv[2]).read().split())] if len(sys.argv) > 2 else []
if ans: out["answer-only tasks: successes"] = "%d/%d" % (sum(e["ok"] for e in ans), len(ans))
fails = [e for e in eps if not e["ok"]]
out["failed episodes"] = len(fails)
out["  declared done, not done"] = sum(e["done"] for e in fails)
out["  never declared done"] = sum(not e["done"] for e in fails)
b = C.OrderedDict((lab, [0, 0]) for lab in ("<=10", "11-20", "21-30", ">30"))
for e in eps:
    lab = "<=10" if e["budget"] <= 10 else "11-20" if e["budget"] <= 20 else "21-30" if e["budget"] <= 30 else ">30"
    b[lab][0] += e["ok"]; b[lab][1] += 1
out["success by step budget"] = {k2: "%d/%d" % tuple(v) for k2, v in b.items()}
tot = sum(len(e["lines"]) for e in eps); noc = sum(l.endswith("→ no change") for e in eps for l in e["lines"])
out["executed steps (history lines)"] = tot
out["steps that changed nothing"] = "%d/%d = %.1f%%" % (noc, tot, 100 * noc / tot)
same = nxt = 0
for e in eps:
    L = e["lines"]
    for i, l in enumerate(L[:-1]):
        if l.endswith("→ no change"):
            nxt += 1; same += L[i + 1].split(" →")[0] == l.split(" →")[0]
out["after a no-change step, next action identical"] = "%d/%d = %.1f%%" % (same, nxt, 100 * same / nxt)
def maxrun(L):
    run = best = 1 if L else 0
    for a, c in zip(L, L[1:]):
        run = run + 1 if a.split(" →")[0] == c.split(" →")[0] else 1; best = max(best, run)
    return best
out["failed episodes with one action repeated >=5 times in a row"] = "%d/%d" % (sum(maxrun(e["lines"]) >= 5 for e in fails), len(fails))
fo = [e["steps"][0]["action_executed"] for e in eps if e["steps"] and e["steps"][0].get("action_executed")]
out["first executed action is open_app"] = "%d/%d" % (sum(a["action_type"] == "open_app" for a in fo), n)
p50 = sorted(s["model_ms"] for e in eps for s in e["steps"] if s.get("model_ms"))
out["model latency ms p50 / p95 (all steps)"] = "%.1f / %.1f" % (p50[len(p50) // 2], p50[int(.95 * len(p50))])
for k2, v in out.items(): print("%-62s %s" % (k2, v))
