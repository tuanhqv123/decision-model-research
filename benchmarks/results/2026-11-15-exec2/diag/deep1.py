"""Deep analysis of AC-Low false 'done' (exec1/exec2/DMA-2B), paired with the executor test on the same (ep, step)."""
import collections, json, sys

def clean(s): return " ".join((s or "").split()).rstrip(" .")
def said(p): return p["pred"] in ("status", "terminate")
def pt(p):
    pr = p.get("probs") or {}
    return float(pr.get("terminate", pr.get("status", 0.0)))

def load_preds(path): return {(str(p["episode_id"]), p["step"]): p for p in map(json.loads, open(path))}

def load_rows(path, keep):
    out = []
    for l in open(path):
        r = json.loads(l)
        out.append({k: r.get(k) for k in keep})
    return out

low = load_rows("actest/data/compact_actest_v4.jsonl", ["ep", "step", "local_goal", "local_done", "action", "history_v3", "goal"])
exe = load_rows("actest/data/compact_actest_exec.jsonl", ["ep", "step", "local_goal", "aug"])
P = {"dma": load_preds("actest/eval/eval-v4_s0_ckpt_2482_actest_v4_low-VA-val.jsonl"),
     "exec1": load_preds("gate_exec1/eval-exec1_actest_low-VA-val.jsonl"),
     "exec2": load_preds("gate_exec2/eval-exec2_actest_low-VA-val.jsonl")}
PN = {"exec1": load_preds("gate_exec1/eval-exec1_actest_low_noterm-VA-val.jsonl"),
      "exec2": load_preds("gate_exec2/eval-exec2_actest_low_noterm-VA-val.jsonl")}
PE = {"exec1": load_preds("gate_exec1/eval-exec1_actest_exec-VA-val.jsonl"),
      "exec2": load_preds("gate_exec2/eval-exec2_actest_exec-VA-val.jsonl")}

# position of each AC-Low step inside its instruction unit (same rule as build_exec_data.units)
low.sort(key=lambda r: (str(r["ep"]), r["step"]))
pos, unit_len, prev = {}, {}, None
cur = []
def close(u):
    for j, r in enumerate(u):
        pos[(str(r["ep"]), r["step"])] = (j, len(u))
for r in low:
    if prev is None or str(prev["ep"]) != str(r["ep"]) or clean(prev["local_goal"]) != clean(r["local_goal"]):
        if cur: close(cur)
        cur = []
    cur.append(r)
    if r.get("local_done"):
        close(cur); cur = []
    prev = r
if cur: close(cur)

exe_span = {(str(r["ep"]), r["step"]): r for r in exe if r["aug"] == "span"}
res = {}

# 1. overall + by position class
def cls(k):
    j, n = pos[k]
    return "first of unit" if j == 0 else "continuation"
tab = collections.defaultdict(lambda: collections.Counter())
for r in low:
    k = (str(r["ep"]), r["step"]); c = cls(k)
    for m, pm in P.items():
        p = pm[k]
        tab[(m, c)]["n"] += 1
        tab[(m, c)]["said done"] += said(p)
        tab[(m, c)]["correct"] += bool(p["correct"])
    for m, pm in PN.items():
        tab[(m + "_noterm", c)]["n"] += 1
        tab[(m + "_noterm", c)]["correct"] += bool(pm[k]["correct"])
res["1 by position"] = {"%s | %s" % kk: dict(v) for kk, v in sorted(tab.items())}

# 2. paired with the executor test: same screen/history; instruction text identical or not
t2 = collections.defaultdict(collections.Counter)
mism = []
for r in low:
    k = (str(r["ep"]), r["step"])
    e = exe_span.get(k)
    if e is None:
        t2["no executor row"]["n"] += 1; continue
    same = clean(e["local_goal"]) == clean(r["local_goal"])
    key = "%s, executor text %s" % (cls(k), "== AC-Low text" if same else "= several instructions")
    for m in ("exec1", "exec2"):
        a, b = P[m][k], PE[m].get(k)
        if b is None: continue
        t2[(m, key)]["n"] += 1
        t2[(m, key)]["AC-Low said done"] += said(a)
        t2[(m, key)]["executor said done"] += said(b)
        t2[(m, key)]["both"] += said(a) and said(b)
        if same:
            t2[(m, key)]["same pred"] += a["pred"] == b["pred"]
            if m == "exec2" and a["pred"] != b["pred"] and len(mism) < 8:
                mism.append({"k": k, "low": r["local_goal"], "exe": e["local_goal"], "low_pred": a["pred"], "exe_pred": b["pred"],
                             "pt_low": round(pt(a), 3), "pt_exe": round(pt(b), 3)})
res["2 paired with executor test"] = {(" | ".join(kk) if isinstance(kk, tuple) else kk): dict(v) for kk, v in t2.items()}
res["2b exec2 same-text prediction mismatches (examples)"] = mism

# 3. exec2 false done: gold action, position, unit length, p bins, last-of-unit
fd = [r for r in low if said(P["exec2"][(str(r["ep"]), r["step"])])]
def summ(rows, m):
    c = collections.Counter(); g = collections.Counter(); jpos = collections.Counter(); bins = collections.Counter()
    for r in rows:
        k = (str(r["ep"]), r["step"]); j, n = pos[k]; p = P[m][k]
        g[r["action"]["action_type"]] += 1
        jpos["j=%d of %d" % (j + 1, n) if n <= 4 else "j=%d of >4" % (j + 1)] += 1
        c["last step of unit"] += (j == n - 1)
        x = pt(p); bins["<0.5" if x < .5 else "0.5-0.8" if x < .8 else "0.8-0.95" if x < .95 else ">=0.95"] += 1
    return {"n": len(rows), "gold action": dict(g.most_common()), "position": dict(jpos.most_common(12)),
            "p(terminate)": dict(bins), **c}
res["3 exec2 false done"] = summ(fd, "exec2")
res["3 exec1 false done"] = summ([r for r in low if said(P["exec1"][(str(r["ep"]), r["step"])])], "exec1")

# base rates: gold action at continuation vs first
base = collections.defaultdict(collections.Counter)
for r in low:
    base[cls((str(r["ep"]), r["step"]))][r["action"]["action_type"]] += 1
res["3b gold action by position (all AC-Low)"] = {k: dict(v.most_common()) for k, v in base.items()}

# rate of false done per gold action x position
rate = collections.defaultdict(lambda: [0, 0, 0])
for r in low:
    k = (str(r["ep"]), r["step"]); kk = "%s | %s" % (cls(k), r["action"]["action_type"])
    rate[kk][0] += 1; rate[kk][1] += said(P["exec1"][k]); rate[kk][2] += said(P["exec2"][k])
res["3c false done per position x gold action [n, exec1, exec2]"] = dict(sorted(rate.items(), key=lambda x: -x[1][2]))

# 4. transitions exec1 -> exec2 on AC-Low correctness
tr = collections.Counter()
for r in low:
    k = (str(r["ep"]), r["step"]); a, b = P["exec1"][k], P["exec2"][k]
    tr["%s->%s" % ("ok" if a["correct"] else ("done" if said(a) else "err"), "ok" if b["correct"] else ("done" if said(b) else "err"))] += 1
res["4 exec1->exec2 (ok / said done / other error)"] = dict(tr.most_common())

# 5. what terminate costs: steps lost vs noterm
for m in ("exec1", "exec2"):
    lost = sum(1 for r in low if PN[m][(str(r["ep"]), r["step"])]["correct"] and said(P[m][(str(r["ep"]), r["step"])]))
    res["5 %s: false done on steps the noterm run gets right" % m] = lost
need = -(-int(0.883 * 8444 * 1000) // 1000)
res["5 need correct for 88.3"] = need

json.dump(res, open("gate_exec2/deep1.json", "w"), indent=1)
print(json.dumps(res, indent=1))
