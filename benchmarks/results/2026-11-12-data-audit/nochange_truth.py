"""The label the MODEL reads ("→ no change" in the next row's history_v3, = ac_common.effect_changed, the serving function):
is it true, and what does the data do after it? (2026-11-12)
  1. pixels changed after actions the history calls "no change", by action
  2. after such an action: gold action, same element again?, same step instruction (annotator retry)?
  3. hand-checkable categories: why did a 'no change' click have no change?
    python nochange_truth.py CODE_DIR DATA.jsonl"""
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
def pb(p): return "<1%" if p is None or p < .01 else "1-5%" if p < .05 else "5-20%" if p < .2 else ">=20%"
px = C.defaultdict(C.Counter); nxt = C.defaultdict(C.Counter); kind = C.Counter()
for e, st in ep.items():
    for s, r in st.items():
        if s + 1 not in st or not (st[s + 1].get("history_v3") or [""])[-1].endswith("→ no change"): continue
        a = r["action"]["action_type"]; n = st[s + 1]; px[a][pb(r.get("pixel_change"))] += 1
        pe, ne = tl(r), tl(n)
        same_el = pe is not None and ne is not None and A._effect_key(pe) == A._effect_key(ne)
        same_dir = a == "scroll" and n["action"]["action_type"] == "scroll" and r["action"].get("direction") == n["action"].get("direction")
        retry = (r.get("local_goal") or "").strip().lower() == (n.get("local_goal") or "").strip().lower()
        g = n["action"]["action_type"]
        lab = ("REPEAT same element" if same_el and g == a else "REPEAT same direction" if same_dir else g) + (" (same instruction)" if retry else "")
        nxt[a][lab] += 1
        if a == "click" and pe is not None:
            f = pe["flags"]
            kind["editable field" if f["editable"] else "checkable/selected" if f["checkable"] or f["selected"] else "keypad digit/char" if len(pe["label"]) == 1 else "unlabelled" if not pe["label"] else "labelled clickable" if f["clickable"] else "labelled non-clickable"] += 1
print("1. pixels changed after an action the history calls 'no change'")
for a in sorted(px): n = sum(px[a].values()); print("   %-14s n=%4d  %s" % (a, n, "  ".join("%s %.0f%%" % (b, 100 * px[a][b] / n) for b in ("<1%", "1-5%", "5-20%", ">=20%"))))
print("\n2. what the data does right after it (gold)")
for a in sorted(nxt): n = sum(nxt[a].values()); print("   %-14s n=%4d  %s" % (a, n, "; ".join("%s %d" % kv for kv in nxt[a].most_common(7))))
print("\n3. clicks called 'no change': what was clicked", dict(kind.most_common()))
