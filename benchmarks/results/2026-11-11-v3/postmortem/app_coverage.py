"""AndroidWorld v3: success rate of tasks whose app has training rows vs tasks whose app has none.
Task -> primary app by AndroidWorld task-name prefix (the app the goal names first); app rows from train_facts.out."""
import ast, json, math, sys, os
here = os.path.dirname(os.path.abspath(__file__))
rows = ast.literal_eval(next(l for l in open(os.path.join(here, "train_facts.out")) if l.startswith("rows per AndroidWorld app")).split("app", 1)[1].strip())
PREFIX = [("AudioRecorder", "audio recorder"), ("Browser", "files"), ("Camera", "camera"), ("Clock", "clock"),
          ("Contacts", "contacts"), ("Expense", "pro expense"), ("Files", "files"), ("Markor", "markor"),
          ("Notes", "joplin"), ("OpenAppTaskEval", "contacts"), ("OsmAnd", "osmand"), ("Recipe", "broccoli"),
          ("Retro", "retro music"), ("SaveCopyOfReceipt", "simple gallery pro"), ("SimpleCalendar", "simple calendar pro"),
          ("SimpleDraw", "simple draw pro"), ("SimpleSms", "simple sms messenger"), ("SportsTracker", "open tracks sports tracker"),
          ("SystemCopyToClipboard", "clipper"), ("System", "settings"), ("Tasks", "tasks"), ("TurnO", "settings"), ("Vlc", "vlc")]
per = json.loads(open(os.path.join(here, "per_task.json")).read())
groups = {"app has training rows": [0, 0], "app has ZERO training rows": [0, 0]}
excl = set(json.loads(open(os.path.join(here, "excluded.json")).read()))
for task, score in per.items():
    app = next(a for p, a in PREFIX if task.startswith(p))
    g = "app has training rows" if rows.get(app, 0) > 0 else "app has ZERO training rows"
    groups[g][0] += score >= 1; groups[g][1] += 1
for g, (k, n) in groups.items(): print("%-28s %d/%d = %.1f%%" % (g, k, n, 100 * k / n))
(a, n1), (b, n2) = groups.values()
# one-sided Fisher exact test: P(X >= a) under the hypergeometric null
def C(n, k): return math.comb(n, k)
K = a + b; N = n1 + n2
p = sum(C(n1, x) * C(n2, K - x) for x in range(a, min(n1, K) + 1)) / C(N, K)
print("one-sided Fisher exact p = %.4f" % p)

# Confound check: covered-app tasks may simply be shorter. Same split inside each step-budget stratum.
import glob
budget = {}
for f in glob.glob(os.path.join(here, "../androidworld/aw116_v3_s0*/episodes/*.jsonl")):
    if ".abandoned." in f: continue
    h = json.loads(open(f).readline()); budget[h["task_name"]] = h["step_budget"]
for lo, hi in ((0, 10), (11, 20), (21, 999)):
    s = {"rows": [0, 0], "zero": [0, 0]}
    for task, score in per.items():
        if not lo <= budget[task] <= hi: continue
        app = next(a for p_, a in PREFIX if task.startswith(p_)); g = "rows" if rows.get(app, 0) > 0 else "zero"
        s[g][0] += score >= 1; s[g][1] += 1
    print("budget %3d-%-3d  app has rows %d/%d   app has zero rows %d/%d" % (lo, hi, *s["rows"], *s["zero"]))
