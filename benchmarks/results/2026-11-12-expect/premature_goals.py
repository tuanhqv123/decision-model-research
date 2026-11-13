"""What are the 35 premature-status tasks (v3, seed 30)? For each: goal, step of the status, budget, the last 3 history
lines. Then a coarse reading: is the goal multi-part (several items / 'and' / a list), and was the last action a
'save-like' click (Save/OK/Done/Add/Create) — i.e. the model stopped right after finishing ONE part. (2026-11-12)
    python3 premature_goals.py RUN_DIRS(+joined) BUDGETS.json"""
import collections as C, glob, json, os, re, sys
B = json.load(open(sys.argv[2])); c = C.Counter()
for d in sys.argv[1].split("+"):
    for f in sorted(glob.glob(os.path.join(d, "episodes", "*.jsonl"))):
        if ".abandoned." in f: continue
        L = [json.loads(l) for l in open(f)]; r = [x for x in L if x.get("type") == "result"]
        if not r or r[-1].get("success") or r[-1].get("setup_error") or r[-1].get("unscored"): continue
        st = sorted([x for x in L if x.get("type") == "step"], key=lambda s: s["i"])
        if not st or not st[-1].get("done"): continue
        t = r[-1]["task_name"]; goal = next((x.get("goal") for x in L if x.get("goal")), "") or ""
        hist = [s.get("history_line") for s in st[:-1]][-3:]
        multi = bool(re.search(r"\b(and|then|following|all|multiple|each|both)\b|,|:|\b\d+\b (items|expenses|recipes|notes)", goal, re.I))
        savey = bool(hist) and bool(re.search(r'"(save|ok|done|add|create|apply|set|confirm|delete)\b', (hist[-1] or ""), re.I))
        c["n"] += 1; c["multi-part goal"] += multi; c["last action save-like"] += savey; c["both"] += multi and savey
        print("- %-38s status@%2d/%3d multi=%d save=%d | %s" % (t, st[-1]["i"], B.get(t, {}).get("max_steps", 0), multi, savey, goal[:110].replace("\n", " ")))
        print("      last: %s" % " || ".join(h or "" for h in hist))
print(dict(c))
