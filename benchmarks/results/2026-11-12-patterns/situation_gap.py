"""Rank situations by how much AndroidWorld depends on them versus how much training covers them (2026-11-12).
  share_aw / share_train = how over-represented the situation is on AW relative to training ("coverage gap")
  bad_aw = (no-effect actions + premature status) / steps in that situation on AW = how badly v3 does there
  cost = bad steps on AW in that situation = where fixing it pays
Situations also collapse over the step bucket (second table) because the bucket splits small cells.
    python situation_gap.py situations_train.json situations_aw.json"""
import collections as C, json, sys
T, W = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def collapse(d, aw):
    o = C.defaultdict(lambda: C.Counter())
    for k, v in d.items():
        kk = "|".join(k.split("|")[:3]); o[kk]["n"] += v["n"]
        if aw: o[kk]["bad"] += v["no_effect"] + v["premature_status"]; o[kk]["prem"] += v["premature_status"]; o[kk]["noeff"] += v["no_effect"]
        for a, c in (v["act"] if aw else v["gold"]).items(): o[kk]["a:" + a] += c
    return o
t, w = collapse(T, False), collapse(W, True)
nt, nw = sum(v["n"] for v in t.values()), sum(v["n"] for v in w.values())
bad_all = sum(v["bad"] for v in w.values())
print("training rows %d | AW steps %d | AW bad steps %d (no-effect %d, premature status %d)" % (nt, nw, bad_all, sum(v["noeff"] for v in w.values()), sum(v["prem"] for v in w.values())))
print("\nsituation = previous action | its effect | was it a click on an editable field")
print("%-36s %7s %7s %7s %6s %6s %7s  %-34s %s" % ("situation", "train", "AW", "AW/tr", "badAW", "bad%", "of all", "training teaches (gold)", "v3 does on AW"))
rows = []
for k in set(t) | set(w):
    a, b = t.get(k, C.Counter()), w.get(k, C.Counter())
    st, sw = a["n"] / nt, b["n"] / nw
    rows.append((b["bad"], k, a, b, st, sw))
def top(c, n=3):
    tot = sum(v for x, v in c.items() if x.startswith("a:")) or 1
    return ", ".join("%s %.0f%%" % (x[2:], 100 * v / tot) for x, v in sorted(((x, v) for x, v in c.items() if x.startswith("a:")), key=lambda z: -z[1])[:n])
for bad, k, a, b, st, sw in sorted(rows, reverse=True)[:14]:
    print("%-36s %7d %7d %7s %6d %5.0f%% %6.0f%%  %-34s %s" % (k, a["n"], b["n"], ("%.1fx" % (sw / st)) if st else "inf", bad, 100 * bad / max(b["n"], 1), 100 * bad / bad_all, top(a), top(b)))
