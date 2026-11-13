"""Two 'effect' definitions live in the training rows (2026-11-12): the row field `effect` (of THIS row's action) and the
text the NEXT row's history_v3 shows for it ("→ screen changed / no change"). Do they agree? Which code wrote each?
    python effect_defs.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
c = C.Counter(); ex = []
for e, st in ep.items():
    for s, r in st.items():
        if s + 1 not in st: continue
        h = (st[s + 1].get("history_v3") or [""])[-1]
        hv = "no change" if h.endswith("→ no change") else "changed" if h.endswith("→ screen changed") else "none"
        # recompute with the serving function on the two element lists
        rc = A.effect_changed([A.canonical_element(x) for x in r.get("elements") or []], [A.canonical_element(x) for x in st[s + 1].get("elements") or []])
        c[(r["action"]["action_type"], "field effect=%s" % r.get("effect"), "history says " + hv, "recomputed " + ("changed" if rc else "no change"))] += 1
for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
    if v >= 3 or "no change" in str(k): print(v, k)
