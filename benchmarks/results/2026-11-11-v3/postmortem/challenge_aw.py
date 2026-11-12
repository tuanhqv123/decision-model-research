"""Challenge questions on the AndroidWorld v3 traces (2026-11-11).
Q-A  Is a "no change" step really useless? Break no-change steps down by action and by what was clicked
     (a click that focuses a text field can leave the screen visually unchanged and still be right).
Q-B  At which step does the model declare done prematurely? (if it tracks the length of training demos, the
     'Progress: step n' line gives it a position cue)
    python challenge_aw.py ANDROIDWORLD_DIR"""
import collections as C, glob, json, sys
eps = []
for f in sorted(glob.glob(sys.argv[1] + "/aw116_v3_s0*/episodes/*.jsonl")):
    if ".abandoned." in f: continue
    L = [json.loads(l) for l in open(f)]
    r = [x for x in L if x["type"] == "result"][-1]
    if r.get("setup_error") or r.get("unscored"): continue
    eps.append((r, sorted([x for x in L if x["type"] == "step"], key=lambda s: s["i"])))
# Q-A
kind = C.Counter(); nxt_after_field = C.Counter()
for r, st in eps:
    for k, s in enumerate(st):
        ex = s.get("action_executed") or {}
        # the definition the MODEL sees: its own history line (ac_common.effect_changed, same as training)
        if not (s.get("history_line") or "").endswith("→ no change") or not ex: continue
        a = ex.get("action_type")
        if a in ("click", "long_press"):
            t = s.get("click_target") or {}
            fl = t.get("flags", "") if isinstance(t, dict) else ""
            lab = "click on an EDITABLE field" if "e" in str(fl) else (
                  "click on another interactive element" if s.get("click_on_interactive") else "click on a NON-interactive element")
            kind[lab] += 1
            if "e" in str(fl) and k + 1 < len(st):
                nxt_after_field[(st[k + 1].get("action_executed") or {}).get("action_type") or "status/none"] += 1
        else:
            kind[a] += 1
print("Q-A no-change steps by kind:", dict(kind.most_common()), "total", sum(kind.values()))
print("    next action after a no-change click on an editable field:", dict(nxt_after_field.most_common()))
# Q-B
prem = [len(st) for r, st in eps if r.get("declared_done") and not r.get("success")]
ok = [len(st) for r, st in eps if r.get("success")]
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print("Q-B step at which done was declared: premature (n=%d) median %d, IQR %d-%d | correct (n=%d) median %d" %
      (len(prem), q(prem, .5), q(prem, .25), q(prem, .75), len(ok), q(ok, .5)))
print("    premature done step histogram:", dict(sorted(C.Counter(prem).items())))
# Q-C  If done were read from the screen, the step of a premature done should grow with how long the task is
#      (official step budget = AndroidWorld's own length estimate). If it is a length prior, it should not.
pairs = []
for r, st in eps:
    if r.get("declared_done") and not r.get("success"):
        h = r.get("step_budget") or 0
        pairs.append((h, len(st)))
import glob as _g
budget = {}
for f in _g.glob(sys.argv[1] + "/aw116_v3_s0*/episodes/*.jsonl"):
    h = json.loads(open(f).readline()); budget[h["task_name"]] = h["step_budget"]
pairs = [(budget[r["task_name"]], len(st)) for r, st in eps if r.get("declared_done") and not r.get("success")]
for lo, hi in ((0, 10), (11, 20), (21, 999)):
    xs = sorted(s for b, s in pairs if lo <= b <= hi)
    if xs: print("Q-C budget %3d-%-3d: premature done n=%2d, median step %d, max %d" % (lo, hi, len(xs), xs[len(xs) // 2], xs[-1]))
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): rk[o[k]] = (i + j) / 2
        i = j + 1
    return rk
a, b = rank([p[0] for p in pairs]), rank([p[1] for p in pairs]); n = len(pairs)
ma, mb = sum(a) / n, sum(b) / n
rho = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
print("Q-C Spearman(step budget, premature-done step) = %.2f (n=%d)" % (rho, n))
# Q-D  Step-level mechanism for N1: is a click in a never-trained app more often a click that changes nothing?
import ast, os
here = os.path.dirname(os.path.abspath(__file__))
rows = ast.literal_eval(next(l for l in open(os.path.join(here, "train_facts.out")) if l.startswith("rows per AndroidWorld app")).split("app", 1)[1].strip())
PREFIX = eval(open(os.path.join(here, "app_coverage.py")).read().split("PREFIX = ", 1)[1].split("\nper =")[0])
g = {"rows": [0, 0], "zero": [0, 0]}
for r, st in eps:
    app = next(a for p, a in PREFIX if r["task_name"].startswith(p)); k = "rows" if rows.get(app, 0) > 0 else "zero"
    for s in st:
        if (s.get("action_executed") or {}).get("action_type") in ("click", "long_press") and s.get("history_line"):
            g[k][1] += 1; g[k][0] += s["history_line"].endswith("→ no change")
for k, (a, n) in g.items(): print("Q-D clicks that changed nothing, app %s: %d/%d = %.1f%%" % ("has rows " if k == "rows" else "zero rows", a, n, 100 * a / n))
(a1, n1), (a2, n2) = g["rows"], g["zero"]; p = (a1 + a2) / (n1 + n2)
print("Q-D two-proportion z = %.2f (pooled over steps; steps within an episode are not independent)" % ((a2 / n2 - a1 / n1) / (p * (1 - p) * (1 / n1 + 1 / n2)) ** .5))
# Q-E  Is repeating after a no-change step always a mistake? Split the identical repeats by action
#      (another `wait` while an app loads, or another scroll, can be legitimate; the same click on the same element
#      after it changed nothing is not), and ask whether the repeat changed anything the second time.
rep = C.Counter(); rep_changed = C.Counter()
for r, st in eps:
    L = [s for s in st if s.get("history_line")]
    for a_, b_ in zip(L, L[1:]):
        la, lb = a_["history_line"], b_["history_line"]
        if la.endswith("→ no change") and lb.split(" →")[0] == la.split(" →")[0]:
            act = la.split(" ")[0]; rep[act] += 1; rep_changed[act] += not lb.endswith("→ no change")
print("Q-E identical repeats after a no-change step, by action:", dict(rep.most_common()), "total", sum(rep.values()))
print("    ... of which the repeat DID change the screen:", dict(rep_changed.most_common()))
