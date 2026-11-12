"""Self-play run (aw116_selfplay_s1001, in progress): labels from AndroidWorld's own checker, asked after every step.
    python selfplay_facts.py EPISODE_DIR"""
import collections as C, glob, json, os, sys
c = C.Counter(); ex = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    if ".abandoned." in f: continue
    L = [json.loads(l) for l in open(f)]
    res = [x for x in L if x.get("type") == "result"]
    if not res: continue
    r = res[-1]; c["episodes finished"] += 1; c["successes"] += bool(r.get("success"))
    st = sorted([x for x in L if x.get("type") == "step"], key=lambda s: s["i"])
    pr = [s.get("probe_score") for s in st]
    c["steps"] += len(st); c["steps with a checker answer"] += sum(p is not None for p in pr)
    c["steps with a checker error"] += sum("probe_error" in s for s in st); c["steps with full model input recorded"] += sum("state_in" in s for s in st)
    first = next((k for k, p in enumerate(pr) if p is not None and p >= 1), None)
    said = next((k for k, s in enumerate(st) if s.get("done") and s.get("action_executed") is None), None)
    if first is not None:
        c["episodes the checker says complete at some step"] += 1
        key = ("never declared done" if said is None else "declared done the next step" if said == first + 1
               else "declared done LATER than needed" if said > first + 1 else "other")
        c["  ... " + key] += 1
        if key.startswith("declared done LATER"): ex.append((r["task_name"], first + 1, said + 1))
    if said is not None and not r.get("success") and (said == 0 or not (pr[said - 1] or 0) >= 1):
        c["declared done while the checker said NOT complete"] += 1
for k, v in c.items(): print("%-58s %s" % (k, v))
print("late examples (task, complete after step, declared done at step):", ex)
