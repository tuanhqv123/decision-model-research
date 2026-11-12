"""How many AndroidWorld tasks can the v4 change (learn to act on "no change") plausibly add? (2026-11-12)
Per task on v3 seed 30 (no masking): outcome, failure kind, no-effect steps, budget, answer-only. A task is a LOOP failure
when it timed out (no status) and >= 30% of its steps changed nothing. Then: what did masking (v3m, same checkpoint, same
seed; masking replaces verified no-effect repeats by the model's next-best) do on exactly those tasks?
    python3 expected_gain.py V3_DIRS(+joined) V3M_DIRS(+joined) BUDGETS.json ANSWER_ONLY.txt"""
import collections as C, glob, json, os, sys
def load(dirs):
    out = {}
    for d in dirs.split("+"):
        for f in glob.glob(os.path.join(d, "episodes", "*.jsonl")):
            if ".abandoned." in f: continue
            L = [json.loads(l) for l in open(f)]; r = [x for x in L if x.get("type") == "result"]
            if not r or r[-1].get("setup_error") or r[-1].get("unscored"): continue
            st = [x for x in L if x.get("type") == "step"]
            ne = sum(1 for s in st if (s.get("action_executed") or {}).get("action_type") not in (None, "wait") and (s.get("history_line") or "").endswith("→ no change"))
            done = any(s.get("done") for s in st)
            out[r[-1]["task_name"]] = {"ok": bool(r[-1].get("success")), "steps": len(st), "noeff": ne, "said_done": done}
    return out
v3, v3m = load(sys.argv[1]), load(sys.argv[2])
B = json.load(open(sys.argv[3])); ans = {l.strip() for l in open(sys.argv[4]) if l.strip() and not l.startswith("#")}
kind = C.Counter(); loop = []
for t, e in v3.items():
    if e["ok"]: k = "success"
    elif t in ans: k = "fail: needs answer (out of reach)"
    elif e["said_done"]: k = "fail: premature status"
    elif e["noeff"] >= 0.3 * e["steps"]: k = "fail: LOOP (timeout, >=30% no-effect steps)"; loop.append(t)
    else: k = "fail: timeout, wanders (<30% no-effect)"
    kind[k] += 1
print("v3 seed 30, scored tasks:", len(v3))
for k, v in kind.most_common(): print("  %-48s %d" % (k, v))
lb = [t for t in loop if B[t]["max_steps"] <= 20]
print("loop failures with budget <= 20 steps:", len(lb), "| > 20:", len(loop) - len(lb))
both = [t for t in loop if t in v3m]
print("masking on the loop-failure tasks: rescued %d / %d -> %s" % (sum(v3m[t]["ok"] for t in both), len(both), [t for t in both if v3m[t]["ok"]]))
print("masking on those tasks, how they ended instead:", dict(C.Counter("success" if v3m[t]["ok"] else "premature status" if v3m[t]["said_done"] else "timeout" for t in both)))
lost = [t for t in v3 if v3[t]["ok"] and t in v3m and not v3m[t]["ok"]]
print("masking lost (v3 success -> fail):", lost)
