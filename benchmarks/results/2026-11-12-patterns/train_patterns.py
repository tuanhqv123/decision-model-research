"""What does the v3 TRAINING data teach in the three situations where v3 fails on AndroidWorld? (2026-11-12)
Counts only, from the training file itself (compact_awlike_full_v3.jsonl, 27,454 rows, 4,197 episodes).
  S1 right after a click on an editable field     -> what is the next gold action, what did the history line say?
  S2 right after a step that changed nothing      -> is the gold the SAME action again (repeat) or something else?
  S3 the `status` row                              -> what came just before it; how long were the episodes?
    python train_patterns.py CODE_DIR DATA.jsonl"""
import collections as C, json, statistics as S, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]
ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def effect(line): return "no change" if line.endswith("→ no change") else "changed" if line.endswith("→ screen changed") else "none"
def target_el(r):
    """canonical element the gold point/box falls in (smallest), or None"""
    t = r.get("target")
    if not t: return None
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2
    best = None
    for e in (A.canonical_element(x) for x in r.get("elements") or []):
        b = e["bbox"]
        if b and b["x_min"] <= cx <= b["x_max"] and b["y_min"] <= cy <= b["y_max"]:
            if best is None or (b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"]) < (best["bbox"]["x_max"]-best["bbox"]["x_min"])*(best["bbox"]["y_max"]-best["bbox"]["y_min"]): best = e
    return best
print("rows %d, episodes %d" % (len(rows), len(ep)))
# S1
nxt = C.Counter(); eff = C.Counter(); foc = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if r["action"]["action_type"] != "click" or s + 1 not in st: continue
        te = target_el(r)
        if not te or not te["flags"]["editable"]: continue
        n = st[s + 1]; nxt[n["action"]["action_type"]] += 1
        h = n.get("history_v3") or []; eff[effect(h[-1]) if h else "none"] += 1
        foc[any(A.canonical_element(x)["flags"]["focused"] and A.canonical_element(x)["flags"]["editable"] for x in n.get("elements") or [])] += 1
print("\nS1 after a click on an editable field: n=%d" % sum(nxt.values()))
print("  next gold action:", dict(nxt.most_common()))
print("  history line of that click says:", dict(eff))
print("  next screen has a focused editable element:", dict(foc))
# S2
rep = C.Counter(); kinds = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        h = r.get("history_v3") or []
        if not h or effect(h[-1]) != "no change" or s - 1 not in st: continue
        p = st[s - 1]; pa, ga = p["action"]["action_type"], r["action"]["action_type"]
        same_type = pa == ga
        same_el = same_type and pa in ("click", "long_press", "type_text") and target_el(p) is not None and target_el(r) is not None \
                  and A._effect_key(target_el(p)) == A._effect_key(target_el(r))
        rep["n"] += 1; rep["same type"] += same_type; rep["same type AND same element (exact repeat)"] += same_el
        kinds[(pa, ga)] += 1
print("\nS2 right after a no-change step: n=%d" % rep["n"])
print("  ", {k: "%d (%.0f%%)" % (v, 100 * v / rep["n"]) for k, v in rep.items() if k != "n"})
print("  (previous -> gold) top:", kinds.most_common(8))
# S3
prev = C.Counter(); peff = C.Counter(); L = []
for e, st in ep.items():
    last = max(st); r = st[last]; assert r["action"]["action_type"] == "status"
    L.append(last + 1)
    if last - 1 in st: prev[st[last - 1]["action"]["action_type"]] += 1
    h = r.get("history_v3") or []; peff[effect(h[-1]) if h else "none (status at step 0)"] += 1
print("\nS3 status rows: n=%d" % len(L))
print("  episode length (rows incl. status): median %d, p75 %d, p90 %d, max %d" % (S.median(L), sorted(L)[int(.75*len(L))], sorted(L)[int(.9*len(L))], max(L)))
print("  share of episodes with > 10 rows: %.1f%%, > 15: %.1f%%" % (100*sum(l > 10 for l in L)/len(L), 100*sum(l > 15 for l in L)/len(L)))
print("  action just before status:", dict(prev.most_common()))
print("  effect of that action (history line):", dict(peff))
