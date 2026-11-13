"""In the training data, when the gold REPEATS an action whose history said "no change", did the repeat work? (2026-11-12)
Effect of the repeat = the history line of the row after it (same serving function). Also pixels as a second witness.
    python retry_outcome.py CODE_DIR DATA.jsonl"""
import collections as C, json, sys
sys.path.insert(0, sys.argv[1]); import ac_common as A
rows = [json.loads(l) for l in open(sys.argv[2])]; ep = C.defaultdict(dict)
for r in rows: ep[r["ep"]][r["step"]] = r
def tl(r):
    t = r.get("target")
    if not t: return None
    cx, cy = (t["x_min"] + t["x_max"]) / 2, (t["y_min"] + t["y_max"]) / 2; best = None
    for e in map(A.canonical_element, r.get("elements") or []):
        b = e["bbox"]
        if b and b["x_min"] <= cx <= b["x_max"] and b["y_min"] <= cy <= b["y_max"] and (best is None or (b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"]) < (best["bbox"]["x_max"]-best["bbox"]["x_min"])*(best["bbox"]["y_max"]-best["bbox"]["y_min"])): best = e
    return best
c = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if s - 1 not in st: continue
        p = st[s - 1]; h = (r.get("history_v3") or [""])[-1]
        if not h.endswith("→ no change"): continue
        pa, g = p["action"]["action_type"], r["action"]["action_type"]
        rep = g == pa and ((pa == "scroll" and p["action"].get("direction") == r["action"].get("direction")) or
                           (pa in ("click", "long_press") and tl(p) is not None and tl(r) is not None and A._effect_key(tl(p)) == A._effect_key(tl(r))))
        if not rep: continue
        c["repeats after no change"] += 1
        if s + 1 in st:
            h2 = (st[s + 1].get("history_v3") or [""])[-1]
            c["  repeat then changed the screen" if h2.endswith("→ screen changed") else "  repeat ALSO no change"] += 1
        else: c["  repeat is the last row (no outcome)"] += 1
        c["  repeat pixels >=1%%"] += (r.get("pixel_change") or 0) >= 0.01
for k, v in c.items(): print(k, v)
